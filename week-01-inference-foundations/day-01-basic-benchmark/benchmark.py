from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilbert/distilgpt2"
PROMPT = "Artificial intelligence infrastructure is important because"
MAX_NEW_TOKENS = 50
NUMBER_OF_MEASURED_RUNS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"


@dataclass
class RunResult:
    run_number: int
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    output_tokens_per_second: float
    process_memory_mb_before: float
    process_memory_mb_after: float


def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def process_memory_mb() -> float:
    """Return the current process resident memory in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def synchronize_device(device: torch.device) -> None:
    """
    Wait for queued CUDA operations to finish.

    CUDA operations are asynchronous. Without synchronization, elapsed time
    may not represent the complete generation workload.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    run_number: int,
) -> tuple[RunResult, str]:
    """Generate text once and collect basic performance measurements."""
    encoded_input = tokenizer(prompt, return_tensors="pt")
    encoded_input = {
        name: tensor.to(device)
        for name, tensor in encoded_input.items()
    }

    input_token_count = encoded_input["input_ids"].shape[1]
    memory_before = process_memory_mb()

    synchronize_device(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize_device(device)
    end_time = time.perf_counter()

    memory_after = process_memory_mb()

    total_output_length = output_ids.shape[1]
    generated_token_count = total_output_length - input_token_count
    generation_seconds = end_time - start_time

    if generation_seconds <= 0:
        raise RuntimeError("Measured generation time must be positive.")

    result = RunResult(
        run_number=run_number,
        input_tokens=input_token_count,
        output_tokens=generated_token_count,
        generation_seconds=round(generation_seconds, 4),
        output_tokens_per_second=round(
            generated_token_count / generation_seconds,
            4,
        ),
        process_memory_mb_before=round(memory_before, 2),
        process_memory_mb_after=round(memory_after, 2),
    )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result, generated_text


def save_results(
    model_name: str,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    results: list[RunResult],
    generated_text: str,
) -> Path:
    """Save benchmark metadata and measurements to a JSON file."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model_name = model_name.replace("/", "--")
    output_path = (
        RESULTS_DIRECTORY
        / f"day-01-{safe_model_name}-{timestamp}.json"
    )

    throughput_values = [
        result.output_tokens_per_second for result in results
    ]
    latency_values = [
        result.generation_seconds for result in results
    ]

    payload = {
        "experiment": "day-01-basic-llm-inference-benchmark",
        "created_at_utc": timestamp,
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "operating_system": platform.platform(),
        },
        "configuration": {
            "model_name": model_name,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "measured_runs": len(results),
            "decoding": "greedy",
        },
        "runs": [asdict(result) for result in results],
        "summary": {
            "mean_generation_seconds": round(
                statistics.mean(latency_values),
                4,
            ),
            "mean_output_tokens_per_second": round(
                statistics.mean(throughput_values),
                4,
            ),
        },
        "sample_output": generated_text,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    device = select_device()

    print(f"Loading model: {MODEL_NAME}")
    print(f"Selected device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("\nRunning one warm-up generation...")

    run_generation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=PROMPT,
        max_new_tokens=10,
        run_number=0,
    )

    print("Warm-up completed.\n")

    measured_results: list[RunResult] = []
    last_generated_text = ""

    for run_number in range(1, NUMBER_OF_MEASURED_RUNS + 1):
        result, generated_text = run_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            run_number=run_number,
        )

        measured_results.append(result)
        last_generated_text = generated_text

        print(
            f"Run {run_number}: "
            f"{result.output_tokens} output tokens, "
            f"{result.generation_seconds:.4f} seconds, "
            f"{result.output_tokens_per_second:.4f} tokens/second"
        )

    result_path = save_results(
        model_name=MODEL_NAME,
        device=device,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        results=measured_results,
        generated_text=last_generated_text,
    )

    print("\nGenerated text:")
    print(last_generated_text)

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()