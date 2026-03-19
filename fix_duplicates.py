#!/usr/bin/env python3
"""
fix_duplicates.py - Remove duplicate numinamath entries caused by process restart.

Run 1 processed numinamath entries 0-690,000 before being stopped.
Run 2 restarted from entry 0 and reached ~75,000, creating duplicates.
Both runs used the same deterministic seed, so duplicated entries are identical.

This script deduplicates both final_dataset.jsonl and training_base.jsonl
by removing duplicate questions (keeping the first occurrence).
"""

import json
from pathlib import Path

BASE_DIR = Path("/Users/aniketpatel/Desktop/SOTA_MATH")
REASONING_FILE = BASE_DIR / "final_dataset.jsonl"
BASE_FILE = BASE_DIR / "training_base.jsonl"


def deduplicate_jsonl(filepath: Path, label: str):
    """Remove duplicate entries from a JSONL file, keeping first occurrence."""
    if not filepath.exists():
        print(f"  {label}: File not found, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Deduplicating: {filepath.name}")
    print(f"{'='*60}")

    # Read all lines
    with open(filepath, "r") as f:
        lines = f.readlines()

    original_count = len(lines)
    print(f"  Original line count: {original_count:,}")

    # Deduplicate by question field
    seen_questions = set()
    unique_lines = []
    duplicates_removed = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Keep malformed lines as-is
            unique_lines.append(line)
            continue

        question = record.get("question", "")
        if question in seen_questions:
            duplicates_removed += 1
        else:
            seen_questions.add(question)
            unique_lines.append(line)

        if (i + 1) % 100000 == 0:
            print(f"    Processed {i+1:,}/{original_count:,} lines...")

    print(f"  Duplicates found and removed: {duplicates_removed:,}")
    print(f"  Unique entries remaining: {len(unique_lines):,}")

    if duplicates_removed == 0:
        print(f"  No duplicates found in {filepath.name}. No changes made.")
        return

    # Backup original
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    filepath.rename(backup_path)
    print(f"  Backup saved to: {backup_path.name}")

    # Write deduplicated file
    with open(filepath, "w") as f:
        for line in unique_lines:
            f.write(line + "\n")

    print(f"  ✓ Wrote {len(unique_lines):,} unique entries to {filepath.name}")


def update_progress():
    """Update .hybrid_progress to reflect Run 1's numinamath progress."""
    progress_file = BASE_DIR / ".hybrid_progress"

    # Read current progress
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed_steps": ["math", "aime"], "reasoning_count": 22255}

    # From Run 1 logs: numinamath reached entry 690,000 with reasoning=40,410
    # We need the build script to resume from entry 690,000
    progress["numinamath_index"] = 690000
    progress["numinamath_reasoning_written"] = 40410
    progress["numinamath_base_written"] = 649000

    # Update reasoning_count to include Run 1's numinamath reasoning
    # 22,255 (math+aime) + 40,410 (numinamath from Run 1)
    progress["reasoning_count"] = 22255 + 40410

    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\n  ✓ Updated .hybrid_progress:")
    print(f"    - numinamath_index: 690,000 (resume from here)")
    print(f"    - reasoning_count: {progress['reasoning_count']:,}")
    print(f"    - numinamath_reasoning_written: 40,410")
    print(f"    - numinamath_base_written: 649,000")


def main():
    print("=" * 60)
    print("DUPLICATE ENTRY FIXER")
    print("=" * 60)

    # Step 1: Deduplicate both files
    deduplicate_jsonl(REASONING_FILE, "Reasoning (final_dataset.jsonl)")
    deduplicate_jsonl(BASE_FILE, "Base (training_base.jsonl)")

    # Step 2: Update progress file for resumption
    update_progress()

    print(f"\n{'='*60}")
    print("DONE! Next steps:")
    print("  1. Verify the deduplicated files look correct")
    print("  2. Re-run build_hybrid_dataset.py to resume from entry 690,000")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
