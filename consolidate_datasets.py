#!/usr/bin/env python3
"""
consolidate_datasets.py - Main orchestration script for combining math datasets.

Processes 5 datasets sequentially, converts each to the unified schema:
{
    "question": "...",
    "solution": "...",
    "answer": "...",
    "reasoning": "step by step explanation"
}

Key optimizations:
- PARALLEL batch processing: multiple API calls sent concurrently
- SymPy/regex for answer extraction (NO LLM tokens spent on answers)
- LLM used ONLY for reasoning generation
- Large buffers (30 items) flushed in parallel sub-batches

Outputs to final_dataset.jsonl (streaming), then converts to final_dataset.json.
Deletes source files after each dataset is processed to save disk space.
Supports resuming from a checkpoint.
"""

import json
import os
import sys
import time
import re
import traceback
from pathlib import Path
from datetime import datetime

from answer_extractor import (
    extract_answer, extract_boxed, extract_hash_answer, extract_answer_is,
    safe_eval_expression,
)
from llm_helpers import (
    parallel_batch_generate_reasoning, ensure_ollama_running,
    REASONING_BATCH_SIZE, MAX_CONCURRENT_REQUESTS,
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/aniketpatel/Desktop/SOTA_MATH")
OUTPUT_JSONL = BASE_DIR / "final_dataset.jsonl"
OUTPUT_JSON = BASE_DIR / "final_dataset.json"
PROGRESS_FILE = BASE_DIR / ".progress"
LOG_FILE = BASE_DIR / "consolidation.log"

DATASETS = [
    "gsm8k",
    "math",
    "metamathqa",
    "numinamath_cot",
    "stackmathqa",
]

PROGRESS_INTERVAL = 500

# Buffer size: collect this many items before flushing in parallel
# With 3 concurrent requests of 10 items each = 30 items processed per flush
PARALLEL_BUFFER_SIZE = REASONING_BATCH_SIZE * MAX_CONCURRENT_REQUESTS  # 30


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Progress / Checkpointing ──────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_datasets": [], "current_dataset": None, "current_offset": 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Output Writing ─────────────────────────────────────────────────────────────

def append_records(records: list[dict], output_file: Path):
    with open(output_file, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_record(record: dict) -> bool:
    for field in ["question", "solution", "answer", "reasoning"]:
        if field not in record or not record[field] or not str(record[field]).strip():
            return False
    return True


# ── SymPy-based Answer Extraction (NO LLM) ────────────────────────────────────

def extract_answer_smart(solution: str, question: str = "") -> str:
    """
    Extract the final answer from a solution using regex and SymPy.
    NO LLM calls — purely programmatic.

    Priority order:
    1. \\boxed{} pattern
    2. #### pattern (GSM8K style)
    3. "The answer is:" pattern
    4. Last numeric/expression value in solution
    5. SymPy evaluation of expressions
    """
    # Try standard patterns
    answer = extract_boxed(solution)
    if answer:
        return answer

    answer = extract_hash_answer(solution)
    if answer:
        return answer

    answer = extract_answer_is(solution)
    if answer:
        return answer

    # Try to find "= <answer>" at the end of lines or sentences
    # Matches patterns like "x = 42", "= 3/4", "equals 7"
    eq_patterns = [
        r'=\s*\\boxed\{([^}]+)\}',           # = \boxed{...}
        r'(?:is|equals|=)\s*\$([^$]+)\$',     # is $...$ or = $...$
        r'(?:^|\n)[^\n]*=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$',  # line ending with = number
        r'\\therefore\s*(.+?)(?:\.|$)',         # \therefore ...
        r'(?:answer|result|value)\s*(?:is|=|:)\s*([^\n.]+)',  # "answer is ..."
    ]
    for pattern in eq_patterns:
        match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip().rstrip('.')
            if candidate:
                return candidate

    # Try to find the last standalone number/expression in the solution
    # Look for numbers at the end of the text
    numbers = re.findall(r'(?<![a-zA-Z])([+-]?\d+(?:[.,]\d+)*(?:/\d+)?)\s*[\.\n]?\s*$', solution)
    if numbers:
        return numbers[-1].replace(',', '')

    # Try SymPy on the question if it looks like a direct computation
    if question:
        result = safe_eval_expression(question)
        if result:
            return result

    # Last resort: find the last number mentioned anywhere
    all_numbers = re.findall(r'(?<![a-zA-Z])(\d+(?:\.\d+)?(?:/\d+)?)', solution)
    if all_numbers:
        return all_numbers[-1]

    return ""


# ── Batch Flush ────────────────────────────────────────────────────────────────

def flush_buffer(buffer: list[dict], output_file: Path) -> tuple[int, int]:
    """
    Flush a buffer of records that have question, solution, answer.
    Generates reasoning in parallel batches, validates, and writes.
    Returns (written_count, skipped_count).
    """
    if not buffer:
        return 0, 0

    # Generate reasoning for all items in parallel
    reasoning_items = [
        {"question": r["question"], "solution": r["solution"], "answer": r["answer"]}
        for r in buffer
    ]
    reasonings = parallel_batch_generate_reasoning(reasoning_items)

    written = 0
    skipped = 0
    valid_records = []
    for record, reasoning in zip(buffer, reasonings):
        record["reasoning"] = reasoning
        if validate_record(record):
            valid_records.append(record)
            written += 1
        else:
            skipped += 1

    if valid_records:
        append_records(valid_records, output_file)

    return written, skipped


# ── Dataset Processors ─────────────────────────────────────────────────────────

def process_gsm8k(progress: dict):
    """Process GSM8K: question + answer (embedded solution + #### final_answer)."""
    dataset_name = "gsm8k"
    filepath = BASE_DIR / "gsm8k.json"

    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Loading {dataset_name}...")
    with open(filepath) as f:
        content = f.read()

    decoder = json.JSONDecoder()
    all_entries = []
    idx = 0
    while idx < len(content):
        content_stripped = content[idx:].lstrip()
        if not content_stripped:
            break
        idx = len(content) - len(content_stripped)
        obj, end = decoder.raw_decode(content, idx)
        if isinstance(obj, list):
            all_entries.extend(obj)
        else:
            all_entries.append(obj)
        idx += end

    total = len(all_entries)
    log(f"  {dataset_name}: {total} entries loaded.")

    start_offset = progress.get("current_offset", 0) if progress.get("current_dataset") == dataset_name else 0
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(all_entries):
        if i < start_offset:
            continue

        try:
            question = entry.get("question", "").strip()
            answer_field = entry.get("answer", "").strip()

            if not question or not answer_field:
                skipped += 1
                continue

            if "####" in answer_field:
                parts = answer_field.rsplit("####", 1)
                solution = parts[0].strip()
                answer = parts[1].strip()
            else:
                solution = answer_field
                answer = extract_answer_smart(answer_field, question)

            if not answer:
                skipped += 1
                continue

            buffer.append({"question": question, "solution": solution, "answer": answer})

            if len(buffer) >= PARALLEL_BUFFER_SIZE:
                w, s = flush_buffer(buffer, OUTPUT_JSONL)
                processed += w
                skipped += s
                buffer = []

        except Exception as e:
            skipped += 1
            log(f"  Error at index {i}: {e}")

        if (i + 1) % PROGRESS_INTERVAL == 0:
            log(f"  {dataset_name}: {i+1}/{total} ({processed} written, {skipped} skipped)")
            progress["current_dataset"] = dataset_name
            progress["current_offset"] = i + 1
            save_progress(progress)

    if buffer:
        w, s = flush_buffer(buffer, OUTPUT_JSONL)
        processed += w
        skipped += s

    log(f"  {dataset_name} COMPLETE: {processed} written, {skipped} skipped out of {total}")
    filepath.unlink()
    log(f"  Deleted {filepath}")

    progress["completed_datasets"].append(dataset_name)
    progress["current_dataset"] = None
    progress["current_offset"] = 0
    save_progress(progress)


def process_math(progress: dict):
    """Process MATH: problem + solution + level + type."""
    dataset_name = "math"
    filepath = BASE_DIR / "math.json"

    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Loading {dataset_name}...")
    with open(filepath) as f:
        all_entries = json.load(f)

    total = len(all_entries)
    log(f"  {dataset_name}: {total} entries loaded.")

    start_offset = progress.get("current_offset", 0) if progress.get("current_dataset") == dataset_name else 0
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(all_entries):
        if i < start_offset:
            continue

        try:
            question = entry.get("problem", "").strip()
            solution = entry.get("solution", "").strip()

            if not question or not solution:
                skipped += 1
                continue

            answer = extract_answer_smart(solution, question)
            if not answer:
                skipped += 1
                continue

            buffer.append({"question": question, "solution": solution, "answer": answer})

            if len(buffer) >= PARALLEL_BUFFER_SIZE:
                w, s = flush_buffer(buffer, OUTPUT_JSONL)
                processed += w
                skipped += s
                buffer = []

        except Exception as e:
            skipped += 1
            log(f"  Error at index {i}: {e}")

        if (i + 1) % PROGRESS_INTERVAL == 0:
            log(f"  {dataset_name}: {i+1}/{total} ({processed} written, {skipped} skipped)")
            progress["current_dataset"] = dataset_name
            progress["current_offset"] = i + 1
            save_progress(progress)

    if buffer:
        w, s = flush_buffer(buffer, OUTPUT_JSONL)
        processed += w
        skipped += s

    log(f"  {dataset_name} COMPLETE: {processed} written, {skipped} skipped out of {total}")
    filepath.unlink()
    log(f"  Deleted {filepath}")

    progress["completed_datasets"].append(dataset_name)
    progress["current_dataset"] = None
    progress["current_offset"] = 0
    save_progress(progress)


def process_metamathqa(progress: dict):
    """Process MetaMathQA: query + response."""
    dataset_name = "metamathqa"
    filepath = BASE_DIR / "MetaMathQA-395K.json"

    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Loading {dataset_name}...")
    with open(filepath) as f:
        all_entries = json.load(f)

    total = len(all_entries)
    log(f"  {dataset_name}: {total} entries loaded.")

    start_offset = progress.get("current_offset", 0) if progress.get("current_dataset") == dataset_name else 0
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(all_entries):
        if i < start_offset:
            continue

        try:
            question = entry.get("query", "").strip()
            solution = entry.get("response", "").strip()

            if not question or not solution:
                skipped += 1
                continue

            answer = extract_answer_smart(solution, question)
            if not answer:
                skipped += 1
                continue

            buffer.append({"question": question, "solution": solution, "answer": answer})

            if len(buffer) >= PARALLEL_BUFFER_SIZE:
                w, s = flush_buffer(buffer, OUTPUT_JSONL)
                processed += w
                skipped += s
                buffer = []

        except Exception as e:
            skipped += 1
            log(f"  Error at index {i}: {e}")

        if (i + 1) % PROGRESS_INTERVAL == 0:
            log(f"  {dataset_name}: {i+1}/{total} ({processed} written, {skipped} skipped)")
            progress["current_dataset"] = dataset_name
            progress["current_offset"] = i + 1
            save_progress(progress)

    if buffer:
        w, s = flush_buffer(buffer, OUTPUT_JSONL)
        processed += w
        skipped += s

    log(f"  {dataset_name} COMPLETE: {processed} written, {skipped} skipped out of {total}")
    filepath.unlink()
    log(f"  Deleted {filepath}")

    progress["completed_datasets"].append(dataset_name)
    progress["current_dataset"] = None
    progress["current_offset"] = 0
    save_progress(progress)


def process_numinamath_cot(progress: dict):
    """Process NuminaMath COT: problem + solution."""
    dataset_name = "numinamath_cot"
    filepath = BASE_DIR / "numinamath_cot.json"

    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Loading {dataset_name}...")
    with open(filepath) as f:
        all_entries = json.load(f)

    total = len(all_entries)
    log(f"  {dataset_name}: {total} entries loaded.")

    start_offset = progress.get("current_offset", 0) if progress.get("current_dataset") == dataset_name else 0
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(all_entries):
        if i < start_offset:
            continue

        try:
            question = entry.get("problem", "").strip()
            solution = entry.get("solution", "").strip()

            if not question or not solution:
                skipped += 1
                continue

            answer = extract_answer_smart(solution, question)
            if not answer:
                skipped += 1
                continue

            buffer.append({"question": question, "solution": solution, "answer": answer})

            if len(buffer) >= PARALLEL_BUFFER_SIZE:
                w, s = flush_buffer(buffer, OUTPUT_JSONL)
                processed += w
                skipped += s
                buffer = []

        except Exception as e:
            skipped += 1
            log(f"  Error at index {i}: {e}")

        if (i + 1) % PROGRESS_INTERVAL == 0:
            log(f"  {dataset_name}: {i+1}/{total} ({processed} written, {skipped} skipped)")
            progress["current_dataset"] = dataset_name
            progress["current_offset"] = i + 1
            save_progress(progress)

    if buffer:
        w, s = flush_buffer(buffer, OUTPUT_JSONL)
        processed += w
        skipped += s

    log(f"  {dataset_name} COMPLETE: {processed} written, {skipped} skipped out of {total}")
    filepath.unlink()
    log(f"  Deleted {filepath}")

    progress["completed_datasets"].append(dataset_name)
    progress["current_dataset"] = None
    progress["current_offset"] = 0
    save_progress(progress)


def process_stackmathqa(progress: dict):
    """Process StackMathQA: Q + A (JSONL). Answer extracted via SymPy/regex."""
    dataset_name = "stackmathqa"
    filepath = BASE_DIR / "StackMathQA"

    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {dataset_name} (streaming JSONL, parallel batches)...")

    start_offset = progress.get("current_offset", 0) if progress.get("current_dataset") == dataset_name else 0
    processed = 0
    skipped = 0
    line_num = 0
    buffer = []

    with open(filepath) as f:
        for line in f:
            line_num += 1
            if line_num <= start_offset:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                question = entry.get("Q", "").strip()
                solution = entry.get("A", "").strip()

                if not question or not solution:
                    skipped += 1
                    continue

                answer = extract_answer_smart(solution, question)
                if not answer:
                    # For conceptual Q&A, use a summary marker
                    answer = "See solution"

                buffer.append({"question": question, "solution": solution, "answer": answer})

                if len(buffer) >= PARALLEL_BUFFER_SIZE:
                    w, s = flush_buffer(buffer, OUTPUT_JSONL)
                    processed += w
                    skipped += s
                    buffer = []

            except json.JSONDecodeError:
                skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    log(f"  Error at line {line_num}: {e}")

            if line_num % PROGRESS_INTERVAL == 0:
                log(f"  {dataset_name}: {line_num} lines ({processed} written, {skipped} skipped)")
                progress["current_dataset"] = dataset_name
                progress["current_offset"] = line_num
                save_progress(progress)

    if buffer:
        w, s = flush_buffer(buffer, OUTPUT_JSONL)
        processed += w
        skipped += s

    log(f"  {dataset_name} COMPLETE: {processed} written, {skipped} skipped out of {line_num} lines")
    filepath.unlink()
    log(f"  Deleted {filepath}")

    progress["completed_datasets"].append(dataset_name)
    progress["current_dataset"] = None
    progress["current_offset"] = 0
    save_progress(progress)


# ── Final Conversion ───────────────────────────────────────────────────────────

def convert_jsonl_to_json():
    log("Converting JSONL to JSON array format...")

    if not OUTPUT_JSONL.exists():
        log("  No JSONL file found!")
        return

    total_lines = 0
    with open(OUTPUT_JSONL) as f:
        for _ in f:
            total_lines += 1
    log(f"  Total records: {total_lines}")

    with open(OUTPUT_JSON, "w") as out:
        out.write("[\n")
        with open(OUTPUT_JSONL) as inp:
            for i, line in enumerate(inp):
                line = line.strip()
                if not line:
                    continue
                if i > 0:
                    out.write(",\n")
                out.write(line)
                if (i + 1) % 1000000 == 0:
                    log(f"  Converting: {i+1}/{total_lines}")
        out.write("\n]")

    log(f"  JSON conversion complete: {OUTPUT_JSON}")
    OUTPUT_JSONL.unlink()
    log(f"  Deleted intermediate {OUTPUT_JSONL}")


# ── Main ───────────────────────────────────────────────────────────────────────

PROCESSOR_MAP = {
    "gsm8k": process_gsm8k,
    "math": process_math,
    "metamathqa": process_metamathqa,
    "numinamath_cot": process_numinamath_cot,
    "stackmathqa": process_stackmathqa,
}


def main():
    log("=" * 70)
    log("MATH DATASET CONSOLIDATION PIPELINE")
    log("Azure OpenAI GPT-4o-mini | Parallel batches | SymPy answer extraction")
    log("=" * 70)

    if not ensure_ollama_running():
        log("ERROR: Cannot connect to Azure OpenAI.")
        sys.exit(1)

    progress = load_progress()
    log(f"Progress: completed={progress['completed_datasets']}, "
        f"current={progress.get('current_dataset')}, "
        f"offset={progress.get('current_offset', 0)}")

    for dataset_name in DATASETS:
        if dataset_name in progress["completed_datasets"]:
            log(f"Skipping {dataset_name} (already completed)")
            continue

        log(f"\n{'='*50}")
        log(f"PROCESSING: {dataset_name}")
        log(f"{'='*50}")

        processor = PROCESSOR_MAP[dataset_name]
        start_time = time.time()

        try:
            processor(progress)
        except KeyboardInterrupt:
            log(f"\nInterrupted during {dataset_name}. Progress saved. Run again to resume.")
            sys.exit(0)
        except Exception as e:
            log(f"FATAL ERROR processing {dataset_name}: {e}")
            log(traceback.format_exc())
            log("Progress saved. Fix the issue and run again to resume.")
            sys.exit(1)

        elapsed = time.time() - start_time
        log(f"  Time for {dataset_name}: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    log(f"\n{'='*50}")
    log("FINAL CONVERSION: JSONL → JSON")
    log(f"{'='*50}")
    convert_jsonl_to_json()

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    log(f"\n{'='*70}")
    log("PIPELINE COMPLETE!")
    log(f"Output: {OUTPUT_JSON}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
