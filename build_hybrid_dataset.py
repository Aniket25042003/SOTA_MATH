#!/usr/bin/env python3
"""
build_hybrid_dataset.py - Create two training files:

1. final_dataset.jsonl (100K hardest problems WITH reasoning)
   - Appends to existing 8,822 GSM8K entries
   - All math.json (12,500), all AIME (933)
   - Random ~50K from numinamath, ~28K from StackMathQA

2. training_base.jsonl (everything else WITHOUT reasoning)
   - All MetaMathQA (395K)
   - Remaining numinamath (~809K)
   - Remaining StackMathQA (~1.57M)

Answer extraction: SymPy/regex only (no LLM tokens for answers)
LLM (GPT-4o-mini): Used ONLY for reasoning + AIME solutions
"""

import json
import csv
import sys
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

from answer_extractor import extract_boxed, extract_hash_answer, extract_answer_is, safe_eval_expression
from llm_helpers import (
    parallel_batch_generate_reasoning, ensure_ollama_running,
    REASONING_BATCH_SIZE, MAX_CONCURRENT_REQUESTS,
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/aniketpatel/Desktop/SOTA_MATH")
REASONING_FILE = BASE_DIR / "final_dataset.jsonl"      # 100K with reasoning
BASE_FILE = BASE_DIR / "training_base.jsonl"            # ~2.78M without reasoning
LOG_FILE = BASE_DIR / "hybrid_build.log"
PROGRESS_FILE = BASE_DIR / ".hybrid_progress"

EXISTING_REASONING_COUNT = 8822  # GSM8K already in final_dataset.jsonl
TARGET_REASONING_TOTAL = 100000
PARALLEL_BUFFER_SIZE = REASONING_BATCH_SIZE * MAX_CONCURRENT_REQUESTS  # 30

RANDOM_SEED = 42  # Reproducible sampling

# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Progress ───────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_steps": [], "reasoning_count": EXISTING_REASONING_COUNT}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Utilities ──────────────────────────────────────────────────────────────────

import re

def extract_answer_smart(solution: str, question: str = "") -> str:
    """Extract answer using regex/SymPy only — NO LLM."""
    answer = extract_boxed(solution)
    if answer:
        return answer

    answer = extract_hash_answer(solution)
    if answer:
        return answer

    answer = extract_answer_is(solution)
    if answer:
        return answer

    # "= <number>" patterns
    eq_patterns = [
        r'=\s*\\boxed\{([^}]+)\}',
        r'(?:is|equals|=)\s*\$([^$]+)\$',
        r'(?:^|\n)[^\n]*=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$',
        r'\\therefore\s*(.+?)(?:\.|$)',
        r'(?:answer|result|value)\s*(?:is|=|:)\s*([^\n.]+)',
    ]
    for pattern in eq_patterns:
        match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip().rstrip('.')
            if candidate:
                return candidate

    # Last number at end of solution
    numbers = re.findall(r'(?<![a-zA-Z])([+-]?\d+(?:[.,]\d+)*(?:/\d+)?)\s*[\.\n]?\s*$', solution)
    if numbers:
        return numbers[-1].replace(',', '')

    # SymPy on question
    if question:
        result = safe_eval_expression(question)
        if result:
            return result

    # Any last number
    all_numbers = re.findall(r'(?<![a-zA-Z])(\d+(?:\.\d+)?(?:/\d+)?)', solution)
    if all_numbers:
        return all_numbers[-1]

    return ""


def validate_reasoning_record(record: dict) -> bool:
    for field in ["question", "solution", "answer", "reasoning"]:
        if field not in record or not record[field] or not str(record[field]).strip():
            return False
    return True


def validate_base_record(record: dict) -> bool:
    for field in ["question", "solution", "answer"]:
        if field not in record or not record[field] or not str(record[field]).strip():
            return False
    return True


def append_records(records: list[dict], filepath: Path):
    with open(filepath, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def flush_reasoning_buffer(buffer: list[dict]) -> tuple[int, int]:
    """Generate reasoning in parallel, validate, write to reasoning file. Returns (written, skipped)."""
    if not buffer:
        return 0, 0

    items = [{"question": r["question"], "solution": r["solution"], "answer": r["answer"]} for r in buffer]
    reasonings = parallel_batch_generate_reasoning(items)

    valid = []
    written = 0
    skipped = 0
    for record, reasoning in zip(buffer, reasonings):
        record["reasoning"] = reasoning
        if validate_reasoning_record(record):
            valid.append(record)
            written += 1
        else:
            skipped += 1

    if valid:
        append_records(valid, REASONING_FILE)

    return written, skipped


# ── Step 1: math.json (all 12,500 → reasoning) ───────────────────────────────

def process_math(progress: dict):
    step = "math"
    if step in progress["completed_steps"]:
        log(f"Skipping {step} (already done)")
        return

    filepath = BASE_DIR / "math.json"
    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {step} (all → reasoning file)...")
    with open(filepath) as f:
        entries = json.load(f)

    total = len(entries)
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(entries):
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
            w, s = flush_reasoning_buffer(buffer)
            processed += w
            skipped += s
            buffer = []

        if (i + 1) % 500 == 0:
            log(f"  {step}: {i+1}/{total} ({processed} written, {skipped} skipped)")

    if buffer:
        w, s = flush_reasoning_buffer(buffer)
        processed += w
        skipped += s

    log(f"  {step} COMPLETE: {processed} written, {skipped} skipped")
    progress["completed_steps"].append(step)
    progress["reasoning_count"] += processed
    save_progress(progress)


# ── Step 2: AIME CSV (all 933 → reasoning, generate solution via LLM) ────────

def process_aime(progress: dict):
    step = "aime"
    if step in progress["completed_steps"]:
        log(f"Skipping {step} (already done)")
        return

    filepath = BASE_DIR / "AIME_Dataset_1983_2024.csv"
    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {step} (all → reasoning file, generating solutions)...")

    entries = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("Question", "").strip()
            answer = str(row.get("Answer", "")).strip()
            if question and answer:
                entries.append({"question": question, "answer": answer})

    total = len(entries)
    log(f"  {step}: {total} entries loaded.")

    # For AIME, we need to generate both solution AND reasoning via LLM.
    # We'll use the reasoning prompt but request solution-style output.
    # Since these are competition problems, the "reasoning" IS the solution.
    processed = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(entries):
        # Use the reasoning as the solution (AIME has no solution field)
        buffer.append({
            "question": entry["question"],
            "solution": f"The final answer is {entry['answer']}.",  # Placeholder, will be replaced by reasoning
            "answer": entry["answer"],
        })

        if len(buffer) >= PARALLEL_BUFFER_SIZE:
            # For AIME: reasoning becomes both solution AND reasoning
            items = [{"question": r["question"], "solution": r["solution"], "answer": r["answer"]} for r in buffer]
            reasonings = parallel_batch_generate_reasoning(items)

            valid = []
            for record, reasoning in zip(buffer, reasonings):
                record["solution"] = reasoning  # Use reasoning as solution too
                record["reasoning"] = reasoning
                if validate_reasoning_record(record):
                    valid.append(record)
                    processed += 1
                else:
                    skipped += 1
            if valid:
                append_records(valid, REASONING_FILE)
            buffer = []

        if (i + 1) % 100 == 0:
            log(f"  {step}: {i+1}/{total} ({processed} written)")

    # Flush remaining
    if buffer:
        items = [{"question": r["question"], "solution": r["solution"], "answer": r["answer"]} for r in buffer]
        reasonings = parallel_batch_generate_reasoning(items)
        valid = []
        for record, reasoning in zip(buffer, reasonings):
            record["solution"] = reasoning
            record["reasoning"] = reasoning
            if validate_reasoning_record(record):
                valid.append(record)
                processed += 1
            else:
                skipped += 1
        if valid:
            append_records(valid, REASONING_FILE)

    log(f"  {step} COMPLETE: {processed} written, {skipped} skipped")
    progress["completed_steps"].append(step)
    progress["reasoning_count"] += processed
    save_progress(progress)


# ── Step 3: numinamath_cot (random ~50K → reasoning, rest → base) ─────────────

def process_numinamath(progress: dict):
    step = "numinamath"
    if step in progress["completed_steps"]:
        log(f"Skipping {step} (already done)")
        return

    filepath = BASE_DIR / "numinamath_cot.json"
    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {step} (split into reasoning + base)...")
    with open(filepath) as f:
        entries = json.load(f)

    total = len(entries)

    # Use the reasoning_count from BEFORE numinamath started for deterministic selection
    # This ensures the same indices are selected even when resuming
    pre_numina_reasoning = progress.get("pre_numina_reasoning_count", progress["reasoning_count"])
    if "pre_numina_reasoning_count" not in progress:
        progress["pre_numina_reasoning_count"] = progress["reasoning_count"]
        save_progress(progress)

    remaining_for_reasoning = TARGET_REASONING_TOTAL - pre_numina_reasoning

    # Allocate ~65% of remaining to numinamath, ~35% to stackmathqa
    numina_reasoning_count = min(int(remaining_for_reasoning * 0.65), total)
    log(f"  {step}: {total} entries. Selecting {numina_reasoning_count} for reasoning.")

    # Random sample indices
    rng = random.Random(RANDOM_SEED)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    reasoning_indices = set(all_indices[:numina_reasoning_count])

    # Check for incremental resume point
    start_index = progress.get("numinamath_index", 0)
    reasoning_processed = progress.get("numinamath_reasoning_written", 0)
    base_written = progress.get("numinamath_base_written", 0)
    reasoning_skipped = 0
    base_skipped = 0
    reasoning_buffer = []
    base_buffer = []

    if start_index > 0:
        log(f"  Resuming from entry {start_index} (reasoning: {reasoning_processed}, base: {base_written})")

    for i, entry in enumerate(entries):
        # Skip already-processed entries
        if i < start_index:
            continue

        question = entry.get("problem", "").strip()
        solution = entry.get("solution", "").strip()

        if not question or not solution:
            base_skipped += 1
            continue

        answer = extract_answer_smart(solution, question)

        if i in reasoning_indices:
            if not answer:
                reasoning_skipped += 1
                continue
            reasoning_buffer.append({"question": question, "solution": solution, "answer": answer})
            if len(reasoning_buffer) >= PARALLEL_BUFFER_SIZE:
                w, s = flush_reasoning_buffer(reasoning_buffer)
                reasoning_processed += w
                reasoning_skipped += s
                reasoning_buffer = []
        else:
            if not answer:
                base_skipped += 1
                continue
            base_buffer.append({"question": question, "solution": solution, "answer": answer})
            if len(base_buffer) >= 1000:
                append_records(base_buffer, BASE_FILE)
                base_written += len(base_buffer)
                base_buffer = []

        if (i + 1) % 5000 == 0:
            log(f"  {step}: {i+1}/{total} (reasoning: {reasoning_processed}, base: {base_written})")
            # Save incremental progress every 5000 entries
            progress["numinamath_index"] = i + 1
            progress["numinamath_reasoning_written"] = reasoning_processed
            progress["numinamath_base_written"] = base_written
            progress["reasoning_count"] = pre_numina_reasoning + reasoning_processed
            save_progress(progress)

    # Flush
    if reasoning_buffer:
        w, s = flush_reasoning_buffer(reasoning_buffer)
        reasoning_processed += w
        reasoning_skipped += s
    if base_buffer:
        append_records(base_buffer, BASE_FILE)
        base_written += len(base_buffer)

    log(f"  {step} COMPLETE: reasoning={reasoning_processed} written/{reasoning_skipped} skipped, base={base_written}/{base_skipped} skipped")
    progress["completed_steps"].append(step)
    progress["reasoning_count"] = pre_numina_reasoning + reasoning_processed
    # Clean up incremental keys
    for key in ["numinamath_index", "numinamath_reasoning_written", "numinamath_base_written", "pre_numina_reasoning_count"]:
        progress.pop(key, None)
    save_progress(progress)


# ── Step 4: StackMathQA (random ~28K → reasoning, rest → base) ────────────────

def process_stackmathqa(progress: dict):
    step = "stackmathqa"
    if step in progress["completed_steps"]:
        log(f"Skipping {step} (already done)")
        return

    filepath = BASE_DIR / "StackMathQA"
    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {step} (split into reasoning + base)...")

    # Count total lines first
    total = 0
    with open(filepath) as f:
        for _ in f:
            total += 1

    remaining_for_reasoning = TARGET_REASONING_TOTAL - progress["reasoning_count"]
    stack_reasoning_count = min(remaining_for_reasoning, total)
    log(f"  {step}: {total} entries. Selecting {stack_reasoning_count} for reasoning.")

    # Random sample line numbers
    rng = random.Random(RANDOM_SEED + 1)  # Different seed than numinamath
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    reasoning_indices = set(all_indices[:stack_reasoning_count])

    # Process
    reasoning_processed = 0
    reasoning_skipped = 0
    base_written = 0
    base_skipped = 0
    reasoning_buffer = []
    base_buffer = []
    line_num = 0

    with open(filepath) as f:
        for line in f:
            i = line_num
            line_num += 1

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                base_skipped += 1
                continue

            question = entry.get("Q", "").strip()
            solution = entry.get("A", "").strip()

            if not question or not solution:
                base_skipped += 1
                continue

            answer = extract_answer_smart(solution, question)

            if i in reasoning_indices:
                if not answer:
                    answer = "See solution"
                reasoning_buffer.append({"question": question, "solution": solution, "answer": answer})
                if len(reasoning_buffer) >= PARALLEL_BUFFER_SIZE:
                    w, s = flush_reasoning_buffer(reasoning_buffer)
                    reasoning_processed += w
                    reasoning_skipped += s
                    reasoning_buffer = []
            else:
                if not answer:
                    answer = "See solution"
                base_buffer.append({"question": question, "solution": solution, "answer": answer})
                if len(base_buffer) >= 1000:
                    append_records(base_buffer, BASE_FILE)
                    base_written += len(base_buffer)
                    base_buffer = []

            if (line_num) % 10000 == 0:
                log(f"  {step}: {line_num}/{total} (reasoning: {reasoning_processed}, base: {base_written})")

    # Flush
    if reasoning_buffer:
        w, s = flush_reasoning_buffer(reasoning_buffer)
        reasoning_processed += w
        reasoning_skipped += s
    if base_buffer:
        append_records(base_buffer, BASE_FILE)
        base_written += len(base_buffer)

    log(f"  {step} COMPLETE: reasoning={reasoning_processed}/{reasoning_skipped} skipped, base={base_written}/{base_skipped} skipped")
    progress["completed_steps"].append(step)
    progress["reasoning_count"] += reasoning_processed
    save_progress(progress)


# ── Step 5: MetaMathQA (all 395K → base, no LLM) ─────────────────────────────

def process_metamathqa(progress: dict):
    step = "metamathqa"
    if step in progress["completed_steps"]:
        log(f"Skipping {step} (already done)")
        return

    filepath = BASE_DIR / "MetaMathQA-395K.json"
    if not filepath.exists():
        log(f"  {filepath} not found, skipping.")
        return

    log(f"Processing {step} (all → base file, no LLM)...")
    with open(filepath) as f:
        entries = json.load(f)

    total = len(entries)
    written = 0
    skipped = 0
    buffer = []

    for i, entry in enumerate(entries):
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

        if len(buffer) >= 1000:
            append_records(buffer, BASE_FILE)
            written += len(buffer)
            buffer = []

        if (i + 1) % 50000 == 0:
            log(f"  {step}: {i+1}/{total} ({written} written)")

    if buffer:
        append_records(buffer, BASE_FILE)
        written += len(buffer)

    log(f"  {step} COMPLETE: {written} written, {skipped} skipped")
    progress["completed_steps"].append(step)
    save_progress(progress)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("HYBRID DATASET BUILDER")
    log("100K reasoning (GPT-4o-mini) + ~2.78M base (SymPy only)")
    log("=" * 70)

    if not ensure_ollama_running():
        log("ERROR: Cannot connect to Azure OpenAI.")
        sys.exit(1)

    progress = load_progress()
    log(f"Progress: completed={progress['completed_steps']}, reasoning_count={progress['reasoning_count']}")

    steps = [
        ("math.json (12,500 → reasoning)", process_math),
        ("AIME (933 → reasoning)", process_aime),
        ("numinamath (~50K reasoning + ~809K base)", process_numinamath),
        ("StackMathQA (~28K reasoning + ~1.57M base)", process_stackmathqa),
        ("MetaMathQA (395K → base)", process_metamathqa),
    ]

    for step_name, step_fn in steps:
        log(f"\n{'='*50}")
        log(f"STEP: {step_name}")
        log(f"{'='*50}")

        start_time = time.time()
        try:
            step_fn(progress)
        except KeyboardInterrupt:
            log(f"\nInterrupted. Progress saved. Run again to resume.")
            sys.exit(0)
        except Exception as e:
            log(f"FATAL ERROR: {e}")
            log(traceback.format_exc())
            log("Progress saved. Fix and re-run.")
            sys.exit(1)

        elapsed = time.time() - start_time
        log(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Final stats
    reasoning_count = 0
    if REASONING_FILE.exists():
        with open(REASONING_FILE) as f:
            for _ in f:
                reasoning_count += 1

    base_count = 0
    if BASE_FILE.exists():
        with open(BASE_FILE) as f:
            for _ in f:
                base_count += 1

    log(f"\n{'='*70}")
    log("HYBRID BUILD COMPLETE!")
    log(f"  Reasoning file: {REASONING_FILE} ({reasoning_count} records)")
    log(f"  Base file:      {BASE_FILE} ({base_count} records)")
    log(f"{'='*70}")

    # Cleanup progress
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == "__main__":
    main()
