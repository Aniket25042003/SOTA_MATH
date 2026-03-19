"""llm_helpers.py - LLM integration for generating missing dataset fields.

Uses Azure OpenAI GPT-4o-mini with:
- Batch processing (multiple problems per API call to share system prompts)
- Parallel requests (multiple batches sent concurrently via ThreadPoolExecutor)
- LLM used ONLY for reasoning generation (answer extraction uses SymPy/regex)
"""

import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

AZURE_ENDPOINT = "MY_AZURE_ENDPOINT"
AZURE_API_KEY = "MY_AZURE_API_KEY"
AZURE_API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4o-mini"

# Batch sizes — tuned for 200K TPM limit with concurrency
REASONING_BATCH_SIZE = 10   # problems per API call
ANSWER_BATCH_SIZE = 15      # problems per API call (only used as fallback now)
SOLUTION_BATCH_SIZE = 10

# Concurrency — how many API calls to run in parallel
# With 200K TPM and ~12K tokens/batch, we can do ~16 batches/min
# With 5 concurrent: 5 batches every ~5s = 60 batches/min (but TPM-limited to ~16)
# So 3 concurrent is safe: 3 * 12K = 36K tokens every ~5s = ~432K/min BUT
# each batch takes ~5s, so real rate is 3 batches/5s = 36 batches/min * 12K = 432K/min > 200K
# Let's use 3 concurrent to stay safe with some headroom for bursts
MAX_CONCURRENT_REQUESTS = 3

MAX_RETRIES = 5
INITIAL_BACKOFF = 5
MAX_BACKOFF = 60

# Initialize Azure OpenAI client (thread-safe)
client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)


# ── Core API Call ──────────────────────────────────────────────────────────────

def _call_azure(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str | None:
    """Call Azure OpenAI GPT-4o-mini with retry and rate limit handling."""
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate' in error_str or 'throttl' in error_str:
                print(f"  [Azure] Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                print(f"  [Azure] Error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(INITIAL_BACKOFF)

    return None


def _parse_json_array(text: str, expected_count: int) -> list[str]:
    """Parse a JSON array from LLM output, with fallback heuristics."""
    # Try direct JSON parse
    try:
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if isinstance(result, list) and len(result) == expected_count:
                return [str(item) for item in result]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try parsing as numbered items
    items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\n*$)', text, re.DOTALL)
    if len(items) == expected_count:
        return [item.strip() for item in items]

    # Try splitting by separators
    for sep in ['---', '===', '***']:
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) == expected_count:
            return parts

    # Fallback: return the whole text for all items
    return [text] * expected_count


# ── System Prompts ─────────────────────────────────────────────────────────────

REASONING_SYSTEM_PROMPT = """You are a mathematics tutor. You will receive multiple math problems with their solutions and answers.

For EACH problem, produce a detailed pedagogical step-by-step reasoning that explains:
1. What the problem is asking
2. What approach/method to use and why
3. Each step with explanations of WHY each step is taken
4. How the final answer is reached

Your reasoning should be educational — a student should understand not just WHAT was done, but WHY.

CRITICAL: Return your response as a JSON array of strings, one reasoning per problem, in the same order as the input. Example:
["Reasoning for problem 1...", "Reasoning for problem 2...", ...]"""


# ── Single Batch Functions ─────────────────────────────────────────────────────

def _process_one_reasoning_batch(items: list[dict]) -> list[str]:
    """Process a single batch of reasoning items. Called by ThreadPoolExecutor."""
    if not items:
        return []

    parts = []
    for i, item in enumerate(items):
        parts.append(
            f"--- Problem {i+1} ---\n"
            f"Question: {item['question']}\n"
            f"Solution: {item['solution']}\n"
            f"Final Answer: {item['answer']}"
        )
    user_prompt = "\n\n".join(parts)

    result = _call_azure(REASONING_SYSTEM_PROMPT, user_prompt, max_tokens=4096)
    if result:
        return _parse_json_array(result, len(items))

    # Fallback: use solutions as reasoning
    return [item.get("solution", "") for item in items]


# ── Parallel Batch Functions (public API) ──────────────────────────────────────

def parallel_batch_generate_reasoning(all_items: list[dict]) -> list[str]:
    """
    Generate reasoning for a large list of items using parallel API calls.
    Splits into sub-batches of REASONING_BATCH_SIZE and runs MAX_CONCURRENT_REQUESTS
    in parallel via ThreadPoolExecutor.

    Each item: {"question": ..., "solution": ..., "answer": ...}
    Returns list of reasoning strings in the same order.
    """
    if not all_items:
        return []

    # Split into sub-batches
    sub_batches = []
    for i in range(0, len(all_items), REASONING_BATCH_SIZE):
        sub_batches.append(all_items[i:i + REASONING_BATCH_SIZE])

    # Process sub-batches in parallel
    all_results = [None] * len(sub_batches)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        future_to_idx = {
            executor.submit(_process_one_reasoning_batch, batch): idx
            for idx, batch in enumerate(sub_batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_results[idx] = future.result()
            except Exception as e:
                print(f"  [Parallel] Batch {idx} failed: {e}")
                # Fallback: use solutions
                all_results[idx] = [
                    item.get("solution", "") for item in sub_batches[idx]
                ]

    # Flatten results in order
    flat_results = []
    for batch_results in all_results:
        flat_results.extend(batch_results)

    return flat_results


# ── Legacy single-item wrappers (backward compat) ─────────────────────────────

def batch_generate_reasoning(items: list[dict]) -> list[str]:
    """Generate reasoning for a batch (delegates to parallel version)."""
    return parallel_batch_generate_reasoning(items)


def batch_extract_answers(items: list[dict]) -> list[str]:
    """Legacy — no longer used for answer extraction (use SymPy instead)."""
    return [""] * len(items)


def generate_reasoning(question: str, solution: str, answer: str) -> str:
    results = parallel_batch_generate_reasoning([{"question": question, "solution": solution, "answer": answer}])
    return results[0] if results else solution


def extract_answer_with_llm(question: str, solution: str) -> str:
    """Legacy fallback — should rarely be called now."""
    return ""


def generate_solution_with_llm(question: str, answer: str) -> str:
    """Legacy fallback."""
    return f"The answer to this problem is {answer}."


# ── Health Check ───────────────────────────────────────────────────────────────

def ensure_ollama_running():
    """Verify Azure OpenAI connectivity."""
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Reply with just 'ok'"}],
            max_tokens=5,
        )
        reply = response.choices[0].message.content.strip()
        print(f"Azure OpenAI GPT-4o-mini connected. Test reply: '{reply}'")
        return True
    except Exception as e:
        print(f"Azure OpenAI connection failed: {e}")
        return False
