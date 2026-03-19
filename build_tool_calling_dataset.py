#!/usr/bin/env python3
"""
build_tool_calling_dataset.py - Generate Phase 4 tool-calling dataset.

Uses Azure AI Phi-4-mini-instruct to generate multi-turn conversations where
the model reasons through math problems, calling tools (calculator, cheatsheet)
as needed. Tool responses are computed locally for correctness.

Output: tool_calling_dataset.jsonl (~10K examples)
"""

import json
import math
import os
import re
import sys
import time
import random
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/aniketpatel/Desktop/SOTA_MATH")
OUTPUT_FILE = BASE_DIR / "tool_calling_dataset.jsonl"
LOG_FILE = BASE_DIR / "tool_calling_build.log"
PROGRESS_FILE = BASE_DIR / ".tool_calling_progress"

NEBIUS_ENDPOINT = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_API_KEY = "MY_NEBIUS_API_KEY"
DEPLOYMENT_NAME = "meta-llama/Llama-3.3-70B-Instruct-fast"

MAX_CONCURRENT_REQUESTS = 3
BATCH_SIZE = 5  # problems per API call
MAX_RETRIES = 5
INITIAL_BACKOFF = 5
MAX_BACKOFF = 60

RANDOM_SEED = 42
PROGRESS_INTERVAL = 50

# Target counts per category
CATEGORY_COUNTS = {
    "arithmetic": 1600,
    "verification": 1200,
    "cheatsheet_lookup": 1200,
    "multi_tool": 1600,
    "no_tool": 2400,
    "multi_step_calc": 1000,
    "formula_then_compute": 1000,
}

client = OpenAI(
    base_url=NEBIUS_ENDPOINT,
    api_key=NEBIUS_API_KEY,
)

rng = random.Random(RANDOM_SEED)

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
    return {"completed_categories": {}, "total_written": 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Math Cheatsheet Knowledge Base ─────────────────────────────────────────────

CHEATSHEET = {
    "quadratic_formula": {
        "title": "Quadratic Formula",
        "content": "For ax² + bx + c = 0, the solutions are: x = (-b ± √(b² - 4ac)) / (2a). The discriminant D = b² - 4ac determines the nature of roots: D > 0 → two distinct real roots, D = 0 → one repeated real root, D < 0 → two complex conjugate roots."
    },
    "pythagorean_theorem": {
        "title": "Pythagorean Theorem",
        "content": "In a right triangle with legs a and b and hypotenuse c: a² + b² = c². Common Pythagorean triples: (3,4,5), (5,12,13), (8,15,17), (7,24,25). The converse also holds: if a² + b² = c², the triangle is right-angled."
    },
    "distance_formula": {
        "title": "Distance Formula",
        "content": "The distance between two points (x₁, y₁) and (x₂, y₂) in the plane is: d = √((x₂-x₁)² + (y₂-y₁)²). In 3D with points (x₁,y₁,z₁) and (x₂,y₂,z₂): d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)."
    },
    "area_of_triangle": {
        "title": "Area of a Triangle",
        "content": "Area = ½ × base × height. Using coordinates: Area = ½|x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|. Heron's formula: Area = √(s(s-a)(s-b)(s-c)) where s = (a+b+c)/2. Using trig: Area = ½ab·sin(C)."
    },
    "circle_formulas": {
        "title": "Circle Formulas",
        "content": "Circumference = 2πr = πd. Area = πr². Arc length = rθ (θ in radians). Sector area = ½r²θ. Equation of circle centered at (h,k): (x-h)² + (y-k)² = r²."
    },
    "sphere_formulas": {
        "title": "Sphere Formulas",
        "content": "Surface Area = 4πr². Volume = (4/3)πr³. For a hemisphere: Surface Area = 3πr² (including base), Volume = (2/3)πr³."
    },
    "cylinder_formulas": {
        "title": "Cylinder Formulas",
        "content": "Lateral Surface Area = 2πrh. Total Surface Area = 2πr(r+h). Volume = πr²h."
    },
    "cone_formulas": {
        "title": "Cone Formulas",
        "content": "Slant height l = √(r²+h²). Lateral Surface Area = πrl. Total Surface Area = πr(r+l). Volume = (1/3)πr²h."
    },
    "arithmetic_sequences": {
        "title": "Arithmetic Sequences",
        "content": "nth term: aₙ = a₁ + (n-1)d. Sum of n terms: Sₙ = n/2 × (2a₁ + (n-1)d) = n/2 × (a₁ + aₙ). The common difference d = aₙ₊₁ - aₙ."
    },
    "geometric_sequences": {
        "title": "Geometric Sequences",
        "content": "nth term: aₙ = a₁ × r^(n-1). Sum of n terms: Sₙ = a₁(1-rⁿ)/(1-r) for r≠1. Infinite sum (|r|<1): S∞ = a₁/(1-r). Common ratio r = aₙ₊₁/aₙ."
    },
    "permutations_combinations": {
        "title": "Permutations and Combinations",
        "content": "Permutations (order matters): P(n,r) = n!/(n-r)!. Combinations (order doesn't matter): C(n,r) = n!/(r!(n-r)!). With repetition: permutations = nʳ, combinations = C(n+r-1, r)."
    },
    "probability_basics": {
        "title": "Probability Basics",
        "content": "P(A) = favorable outcomes / total outcomes. P(A or B) = P(A) + P(B) - P(A and B). P(A and B) = P(A) × P(B|A). For independent events: P(A and B) = P(A) × P(B). Complement: P(not A) = 1 - P(A)."
    },
    "binomial_theorem": {
        "title": "Binomial Theorem",
        "content": "(a+b)ⁿ = Σ C(n,k) × a^(n-k) × b^k for k=0 to n. The (k+1)th term is C(n,k) × a^(n-k) × b^k. Pascal's Triangle gives the coefficients."
    },
    "logarithm_rules": {
        "title": "Logarithm Rules",
        "content": "log_b(xy) = log_b(x) + log_b(y). log_b(x/y) = log_b(x) - log_b(y). log_b(xⁿ) = n·log_b(x). Change of base: log_b(x) = ln(x)/ln(b). log_b(1) = 0. log_b(b) = 1. b^(log_b(x)) = x."
    },
    "exponent_rules": {
        "title": "Exponent Rules",
        "content": "a^m × a^n = a^(m+n). a^m / a^n = a^(m-n). (a^m)^n = a^(mn). (ab)^n = a^n × b^n. a^0 = 1. a^(-n) = 1/a^n. a^(1/n) = ⁿ√a."
    },
    "trigonometric_identities": {
        "title": "Trigonometric Identities",
        "content": "sin²θ + cos²θ = 1. tan θ = sin θ / cos θ. sin(A±B) = sinA·cosB ± cosA·sinB. cos(A±B) = cosA·cosB ∓ sinA·sinB. Double angle: sin(2θ) = 2sinθ·cosθ, cos(2θ) = cos²θ - sin²θ = 2cos²θ - 1 = 1 - 2sin²θ."
    },
    "derivative_rules": {
        "title": "Derivative Rules",
        "content": "Power rule: d/dx[xⁿ] = nxⁿ⁻¹. Product rule: d/dx[fg] = f'g + fg'. Quotient rule: d/dx[f/g] = (f'g - fg')/g². Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x). d/dx[eˣ] = eˣ. d/dx[ln x] = 1/x. d/dx[sin x] = cos x. d/dx[cos x] = -sin x."
    },
    "integration_rules": {
        "title": "Integration Rules",
        "content": "∫xⁿ dx = xⁿ⁺¹/(n+1) + C (n≠-1). ∫1/x dx = ln|x| + C. ∫eˣ dx = eˣ + C. ∫sin x dx = -cos x + C. ∫cos x dx = sin x + C. Integration by parts: ∫u dv = uv - ∫v du."
    },
    "compound_interest": {
        "title": "Compound Interest Formula",
        "content": "A = P(1 + r/n)^(nt) where A = final amount, P = principal, r = annual rate (decimal), n = compounds per year, t = years. Continuous compounding: A = Pe^(rt). Simple interest: I = Prt."
    },
    "mean_median_mode": {
        "title": "Mean, Median, Mode",
        "content": "Mean = sum of values / count. Median = middle value when sorted (average of two middle values if even count). Mode = most frequent value. For grouped data: mean = Σ(f×x)/Σf."
    },
    "standard_deviation": {
        "title": "Standard Deviation",
        "content": "Population σ = √(Σ(xᵢ - μ)²/N). Sample s = √(Σ(xᵢ - x̄)²/(n-1)). Variance = σ². For a dataset, ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ (normal distribution)."
    },
    "slope_intercept": {
        "title": "Linear Equations (Slope-Intercept)",
        "content": "Slope-intercept form: y = mx + b. Point-slope form: y - y₁ = m(x - x₁). Slope: m = (y₂-y₁)/(x₂-x₁). Parallel lines have equal slopes. Perpendicular lines have slopes m₁·m₂ = -1. x-intercept: set y=0. y-intercept: set x=0."
    },
    "systems_of_equations": {
        "title": "Systems of Linear Equations",
        "content": "Methods: Substitution (solve one equation for a variable, substitute into the other), Elimination (add/subtract equations to eliminate a variable), Cramer's Rule (using determinants). A system has 0, 1, or infinitely many solutions."
    },
    "matrix_operations": {
        "title": "Matrix Operations",
        "content": "Addition: [A+B]ᵢⱼ = Aᵢⱼ + Bᵢⱼ. Scalar multiplication: [cA]ᵢⱼ = c·Aᵢⱼ. Matrix multiplication: [AB]ᵢⱼ = Σ Aᵢₖ·Bₖⱼ. Det of 2×2: ad-bc. Inverse of 2×2: (1/det)[[d,-b],[-c,a]]."
    },
    "absolute_value": {
        "title": "Absolute Value Properties",
        "content": "|x| = x if x≥0, -x if x<0. |ab| = |a|·|b|. |a/b| = |a|/|b|. |a+b| ≤ |a| + |b| (triangle inequality). |a-b| is the distance between a and b on the number line."
    },
    "factoring_techniques": {
        "title": "Factoring Techniques",
        "content": "Difference of squares: a²-b² = (a-b)(a+b). Perfect square: a²±2ab+b² = (a±b)². Sum/difference of cubes: a³±b³ = (a±b)(a²∓ab+b²). Factor by grouping for 4-term polynomials. Always look for GCF first."
    },
    "number_theory_basics": {
        "title": "Number Theory Basics",
        "content": "Divisibility rules: by 2 (even last digit), by 3 (digit sum divisible by 3), by 5 (ends in 0 or 5), by 9 (digit sum divisible by 9). GCD via Euclidean algorithm. LCM(a,b) = |ab|/GCD(a,b). Fundamental theorem: every integer > 1 is a unique product of primes."
    },
    "modular_arithmetic": {
        "title": "Modular Arithmetic",
        "content": "a ≡ b (mod n) means n divides (a-b). (a+b) mod n = ((a mod n) + (b mod n)) mod n. (a×b) mod n = ((a mod n) × (b mod n)) mod n. Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p if gcd(a,p)=1."
    },
    "work_rate_problems": {
        "title": "Work Rate Problems",
        "content": "If A does a job in a hours and B in b hours: combined rate = 1/a + 1/b. Time together = 1/(1/a + 1/b) = ab/(a+b). For pipes: filling rate - draining rate = net rate."
    },
    "percentage_problems": {
        "title": "Percentage Problems",
        "content": "Percent = (part/whole) × 100. Percent change = ((new-old)/old) × 100. Finding X% of Y: (X/100) × Y. If increased by P%, new value = original × (1 + P/100). Successive discounts: multiply the (1 - discount) factors."
    },
}


# ── Safe Calculator ────────────────────────────────────────────────────────────

SAFE_MATH_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "log": math.log, "log10": math.log10,
    "log2": math.log2, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "pi": math.pi, "e": math.e, "factorial": math.factorial,
    "ceil": math.ceil, "floor": math.floor, "gcd": math.gcd,
    "comb": math.comb, "perm": math.perm,
}


def safe_calc(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        expr = expression.strip()
        # Replace common math notations
        expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH_NAMES)
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return f"{result:.10g}"
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def lookup_cheatsheet(topic: str) -> str:
    """Look up a topic in the cheatsheet."""
    topic_key = topic.lower().strip().replace(" ", "_").replace("-", "_")
    # Direct match
    if topic_key in CHEATSHEET:
        entry = CHEATSHEET[topic_key]
        return f"{entry['title']}: {entry['content']}"
    # Fuzzy match
    for key, entry in CHEATSHEET.items():
        if topic_key in key or key in topic_key:
            return f"{entry['title']}: {entry['content']}"
    # Keyword search
    for key, entry in CHEATSHEET.items():
        title_words = entry["title"].lower().split()
        topic_words = topic.lower().split()
        if any(w in title_words for w in topic_words):
            return f"{entry['title']}: {entry['content']}"
    return f"No exact match found for '{topic}'. Available topics include: quadratic formula, Pythagorean theorem, circle formulas, derivative rules, probability basics, and more."


# ── Nebius Token Factory Call ──────────────────────────────────────────────────

def call_nebius(system_prompt: str, user_prompt: str, prior_messages: list[dict] = None, max_tokens: int = 4096) -> str | None:
    backoff = INITIAL_BACKOFF
    messages = [{"role": "system", "content": system_prompt}]
    
    if prior_messages:
        messages.extend(prior_messages)
    else:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        })

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</tool_call>"]
            )
            raw_output = response.choices[0].message.content.strip()
            stop_reason = response.choices[0].finish_reason
            
            # Re-attach the stop token if it stopped because of it
            if stop_reason == "stop" and "<tool_call>" in raw_output and not raw_output.endswith("</tool_call>"):
                 raw_output += "</tool_call>"
            elif stop_reason == "stop" and "</tool_call>" not in raw_output and "<tool_call>" in raw_output:
                 raw_output += "</tool_call>"
                 
            return raw_output
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate' in error_str or 'throttl' in error_str:
                log(f"  [Nebius] Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                log(f"  [Nebius] Error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(INITIAL_BACKOFF)
    return None


# ── System Prompt for Generation ───────────────────────────────────────────────

TOOL_SYSTEM_PROMPT = """You are a highly skilled math expert with access to external tools. \
Use the available tools to verify calculations and look up concepts/formulas when needed.

Available Tools:
1. calculator — Evaluate a mathematical expression accurately.
   Usage: <tool_call>{"name": "calculator", "arguments": {"expression": "YOUR_EXPR"}}</tool_call>
   The system will respond with: <tool_response>{"result": "VALUE"}</tool_response>

2. math_cheatsheet — Look up a mathematical concept, formula, or theorem.
   Usage: <tool_call>{"name": "math_cheatsheet", "arguments": {"topic": "YOUR_TOPIC"}}</tool_call>
   The system will respond with: <tool_response>{"content": "RELEVANT_INFO"}</tool_response>

For each problem: reason through it, use tools as needed, provide the solution, and state the final answer."""

GENERATION_SYSTEM_PROMPT = """You are a training data generator for a math AI model. \
Your job is to write the ASSISTANT's response to a math problem. The assistant has access to tools.

The assistant's response must follow this EXACT format:

**Reasoning:**
[3-5 sentences of deep analysis: identify the problem type, relevant mathematical concepts, \
multiple possible approaches, justify the chosen approach, mention potential pitfalls or edge cases. \
Explain WHY tools are needed (or not) and what you expect the results to be.]

**Step 1:** [Step title]
[Detailed mathematical work. Show your reasoning BEFORE calling any tool. \
Explain what you expect and why.]
[If using a tool, include the tool call on its own line]
[After receiving a tool result, interpret it: verify it makes sense, connect to broader problem]

**Step 2:** [Step title]
[Continue with detailed work...]

... (use 3-6 steps)

**Solution:**
[Comprehensive 2-4 sentence summary connecting all steps and key insights]

**Answer:** [Clear final answer]

CRITICAL RULES:
1. Tool calls MUST be exactly: <tool_call>{"name": "TOOL_NAME", "arguments": {"PARAM": "VALUE"}}</tool_call>
2. For calculator: the expression must be valid Python math (use ** for power, math.sqrt(), etc.)
3. For cheatsheet: use a clear topic name
4. If you need a tool, output exactly ONE <tool_call>...</tool_call> and STOP exactly after it so we can inject the result. Do NOT write [TOOL_RESULT].
5. THINK DEEPLY. Before each step explain the MATH, not just "I'll calculate this"
6. PREDICT results before calling calculator: "I expect approximately X because..."
7. INTERPRET results after receiving them: "This confirms our expectation because..."
8. DISCUSS alternatives briefly: "We could also use method X, but Y is more efficient here"
9. Make the reasoning EDUCATIONAL — a student should understand WHY each step is taken
10. Keep total response 500-1500 tokens long"""


# ── Problem Generators ─────────────────────────────────────────────────────────

def generate_arithmetic_problems(count: int) -> list[dict]:
    """Generate problems requiring calculator for complex arithmetic."""
    problems = []
    templates = [
        lambda: {"q": f"Calculate {rng.randint(100,9999)} × {rng.randint(100,9999)}.", "tools": ["calculator"]},
        lambda: {"q": f"What is {rng.randint(1000,99999)} ÷ {rng.randint(2,999)}? Give the exact decimal result.", "tools": ["calculator"]},
        lambda: {"q": f"Compute {rng.randint(2,25)}^{rng.randint(3,8)}.", "tools": ["calculator"]},
        lambda: {"q": f"Find the square root of {rng.randint(100,99999)}. Round to 4 decimal places.", "tools": ["calculator"]},
        lambda: {"q": f"Calculate ({rng.randint(10,999)} + {rng.randint(10,999)}) × ({rng.randint(10,999)} - {rng.randint(10,500)}).", "tools": ["calculator"]},
        lambda: {"q": f"What is {rng.randint(2,15)}! (factorial)?", "tools": ["calculator"]},
        lambda: {"q": f"Evaluate {rng.randint(100,999)} × {rng.randint(100,999)} + {rng.randint(100,999)} × {rng.randint(10,99)}.", "tools": ["calculator"]},
        lambda: {"q": f"A factory produces {rng.randint(100,999)} items per hour. How many items in {rng.randint(10,99)} hours and {rng.randint(10,59)} minutes?", "tools": ["calculator"]},
        lambda: {"q": f"Find the value of ({rng.randint(2,50)}³ - {rng.randint(2,50)}²) / {rng.randint(2,20)}.", "tools": ["calculator"]},
        lambda: {"q": f"Calculate the sum: {rng.randint(1000,9999)} + {rng.randint(1000,9999)} + {rng.randint(1000,9999)} + {rng.randint(1000,9999)}.", "tools": ["calculator"]},
    ]
    for i in range(count):
        t = templates[i % len(templates)]()
        problems.append({"question": t["q"], "category": "arithmetic", "expected_tools": t["tools"]})
    return problems


def generate_verification_problems(count: int) -> list[dict]:
    """Generate problems where model solves manually then verifies with calculator."""
    problems = []
    templates = [
        lambda: f"Solve step by step: {rng.randint(2,9)}x + {rng.randint(1,20)} = {rng.randint(25,100)}. Verify your answer by substituting back.",
        lambda: f"A store has a {rng.randint(15,50)}% off sale. An item costs ${rng.randint(50,500):.2f}. What is the sale price and how much do you save? Verify your calculations.",
        lambda: f"If you invest ${rng.randint(1000,50000)} at {rng.randint(2,12)}% annual interest compounded monthly for {rng.randint(1,10)} years, what is the final amount? Show work and verify.",
        lambda: f"Calculate the area and perimeter of a rectangle with length {rng.randint(5,50)}.{rng.randint(1,9)} cm and width {rng.randint(3,30)}.{rng.randint(1,9)} cm. Verify your answers.",
        lambda: f"A train travels {rng.randint(100,500)} km in {rng.randint(2,8)} hours {rng.randint(10,59)} minutes. What is its average speed in km/h? Verify by calculating distance = speed × time.",
        lambda: f"Find the GCD and LCM of {rng.randint(20,200)} and {rng.randint(20,200)}. Verify that GCD × LCM = product of the two numbers.",
    ]
    for i in range(count):
        t = templates[i % len(templates)]
        problems.append({"question": t(), "category": "verification", "expected_tools": ["calculator"]})
    return problems


def generate_cheatsheet_problems(count: int) -> list[dict]:
    """Generate problems requiring formula lookups."""
    problems = []
    topics = list(CHEATSHEET.keys())
    template_map = {
        "quadratic_formula": lambda: f"Solve the equation {rng.randint(1,5)}x² + {rng.randint(-20,20)}x + {rng.randint(-30,30)} = 0 using the quadratic formula.",
        "pythagorean_theorem": lambda: f"A right triangle has legs of length {rng.randint(3,30)} and {rng.randint(4,40)}. Find the hypotenuse.",
        "circle_formulas": lambda: f"Find the area and circumference of a circle with radius {rng.randint(1,50)}.{rng.randint(0,9)}.",
        "arithmetic_sequences": lambda: f"In an arithmetic sequence, the first term is {rng.randint(1,20)} and the common difference is {rng.randint(1,10)}. Find the {rng.randint(10,50)}th term and the sum of the first {rng.randint(10,50)} terms.",
        "geometric_sequences": lambda: f"A geometric sequence has first term {rng.randint(1,10)} and common ratio {rng.randint(2,5)}. Find the {rng.randint(5,12)}th term and the sum of the first {rng.randint(5,12)} terms.",
        "compound_interest": lambda: f"Calculate the compound interest on ${rng.randint(1000,100000)} at {rng.randint(2,15)}% per annum for {rng.randint(1,20)} years, compounded quarterly.",
        "permutations_combinations": lambda: f"In how many ways can you choose {rng.randint(2,6)} items from {rng.randint(8,20)} items? Calculate both permutations and combinations.",
        "derivative_rules": lambda: f"Find the derivative of f(x) = {rng.randint(1,10)}x^{rng.randint(2,5)} + {rng.randint(1,10)}x^{rng.randint(1,3)} - {rng.randint(1,20)}x + {rng.randint(1,50)}.",
        "logarithm_rules": lambda: f"Simplify: log₂({2**rng.randint(3,10)}) + log₂({2**rng.randint(1,5)}). Then verify using logarithm rules.",
        "trigonometric_identities": lambda: f"If sin(θ) = {rng.randint(1,9)}/{rng.randint(10,20)}, find cos(θ), tan(θ), and sin(2θ). Use trigonometric identities.",
        "probability_basics": lambda: f"A bag contains {rng.randint(3,10)} red, {rng.randint(2,8)} blue, and {rng.randint(1,6)} green balls. What is the probability of drawing 2 red balls in succession without replacement?",
        "standard_deviation": lambda: f"Find the mean, variance, and standard deviation of the dataset: {', '.join(str(rng.randint(10,100)) for _ in range(rng.randint(5,8)))}.",
        "sphere_formulas": lambda: f"Find the surface area and volume of a sphere with radius {rng.randint(1,25)}.",
        "area_of_triangle": lambda: f"Find the area of a triangle with sides {rng.randint(5,20)}, {rng.randint(5,20)}, and {rng.randint(5,20)} using Heron's formula.",
        "percentage_problems": lambda: f"A product's price increased by {rng.randint(10,40)}% and then decreased by {rng.randint(10,30)}%. If the original price was ${rng.randint(50,500)}, what is the final price?",
    }
    for i in range(count):
        topic = topics[i % len(topics)]
        if topic in template_map:
            q = template_map[topic]()
        else:
            entry = CHEATSHEET[topic]
            q = f"Explain and apply the concept of {entry['title'].lower()} to solve a practical problem."
        problems.append({"question": q, "category": "cheatsheet_lookup", "expected_tools": ["math_cheatsheet"], "topic_hint": topic})
    return problems


def generate_multi_tool_problems(count: int) -> list[dict]:
    """Generate problems needing both cheatsheet lookup AND calculator."""
    problems = []
    templates = [
        lambda: f"Using the quadratic formula, solve {rng.randint(2,8)}x² - {rng.randint(5,30)}x + {rng.randint(1,20)} = 0. Give exact and decimal answers.",
        lambda: f"A cone has radius {rng.randint(3,20)} cm and height {rng.randint(5,30)} cm. Find its volume, lateral surface area, and total surface area.",
        lambda: f"Find the {rng.randint(15,30)}th term and sum of the first {rng.randint(15,30)} terms of a geometric sequence with a₁ = {rng.randint(2,10)} and r = {rng.randint(2,4)}.",
        lambda: f"Calculate the compound interest on a principal of ${rng.randint(5000,50000)} at {rng.randint(3,12)}.{rng.randint(0,9)}% annual rate compounded monthly for {rng.randint(2,10)} years. What is the total interest earned?",
        lambda: f"A right triangle has one leg of {rng.randint(5,25)} cm. The hypotenuse is {rng.randint(30,50)} cm. Find the other leg, the area, and the perimeter.",
        lambda: f"Find the area and perimeter of a circular sector with radius {rng.randint(5,30)} cm and central angle {rng.randint(30,270)}°.",
        lambda: f"Using Heron's formula, find the area of a triangle with sides {rng.randint(10,30)}, {rng.randint(10,30)}, and {rng.randint(10,30)}. Then find the height from the longest side.",
        lambda: f"A cylindrical tank has radius {rng.randint(2,15)} m and height {rng.randint(5,25)} m. Find the volume in cubic meters and how many liters of water it can hold.",
    ]
    for i in range(count):
        t = templates[i % len(templates)]
        problems.append({"question": t(), "category": "multi_tool", "expected_tools": ["math_cheatsheet", "calculator"]})
    return problems


def generate_no_tool_problems(count: int) -> list[dict]:
    """Generate simple problems that don't need tools."""
    problems = []
    templates = [
        lambda: f"What is {rng.randint(1,20)} + {rng.randint(1,20)}?",
        lambda: f"Simplify the fraction {rng.randint(2,12)*rng.randint(2,5)}/{rng.randint(2,12)*rng.randint(2,5)}.",
        lambda: f"Is {rng.randint(1,100)*2+1} an even or odd number? Explain why.",
        lambda: f"What is {rng.randint(1,12)} × {rng.randint(1,12)}?",
        lambda: f"Solve for x: x + {rng.randint(1,20)} = {rng.randint(21,50)}.",
        lambda: f"What is {rng.randint(10,99)} - {rng.randint(1,50)}?",
        lambda: f"Convert {rng.randint(1,7)}/{rng.randint(2,8)} to a decimal.",
        lambda: f"What is 10% of {rng.randint(1,20) * 10}?",
        lambda: f"How many sides does a {rng.choice(['triangle', 'pentagon', 'hexagon', 'octagon'])} have?",
        lambda: f"Factor the expression x² - {rng.randint(2,12)**2}.",
        lambda: f"What is the next number in the sequence: {', '.join(str(i*rng.randint(2,5)+rng.randint(0,3)) for i in range(4))}?",
        lambda: f"Solve: {rng.randint(2,9)}x = {rng.randint(2,9)*rng.randint(2,12)}.",
    ]
    for i in range(count):
        t = templates[i % len(templates)]
        problems.append({"question": t(), "category": "no_tool", "expected_tools": []})
    return problems


def generate_multi_step_calc_problems(count: int) -> list[dict]:
    """Generate problems requiring multiple calculator calls."""
    problems = []
    templates = [
        lambda: f"A store sells {rng.randint(3,8)} different products at prices ${rng.randint(10,99)}.{rng.randint(10,99)}, ${rng.randint(10,99)}.{rng.randint(10,99)}, and ${rng.randint(10,99)}.{rng.randint(10,99)}. A customer buys {rng.randint(1,5)} of the first, {rng.randint(1,5)} of the second, and {rng.randint(1,5)} of the third. What is the total before tax, the {rng.randint(5,12)}% tax amount, and the final total?",
        lambda: f"Calculate: ({rng.randint(100,999)} × {rng.randint(10,99)}) + ({rng.randint(100,999)} × {rng.randint(10,99)}) - ({rng.randint(10,99)}² × {rng.randint(2,9)}). Break this into parts.",
        lambda: f"A company's revenue was ${rng.randint(100,999)}K in Q1, ${rng.randint(100,999)}K in Q2, ${rng.randint(100,999)}K in Q3, and ${rng.randint(100,999)}K in Q4. Calculate total annual revenue, average quarterly revenue, and the percentage each quarter contributed.",
        lambda: f"Convert {rng.randint(100,999)} square meters to square feet (1 m = 3.28084 ft), then to square yards, then calculate how many rooms of {rng.randint(10,30)} sq yards each can fit.",
        lambda: f"Find the surface area and volume of a rectangular prism with dimensions {rng.randint(3,20)} × {rng.randint(3,20)} × {rng.randint(3,20)} cm. Then find the length of the space diagonal.",
    ]
    for i in range(count):
        t = templates[i % len(templates)]
        problems.append({"question": t(), "category": "multi_step_calc", "expected_tools": ["calculator"]})
    return problems


def generate_formula_then_compute_problems(count: int) -> list[dict]:
    """Generate problems that first need a formula lookup, then computation."""
    problems = []
    templates = [
        lambda: f"A ball is thrown upward with initial velocity {rng.randint(10,50)} m/s. Using the kinematic equations, find the maximum height and time to reach it (g = 9.8 m/s²).",
        lambda: f"Find the binomial coefficient C({rng.randint(8,20)}, {rng.randint(2,6)}) and use it to find the {rng.randint(2,6)}th term in the expansion of (x + {rng.randint(2,5)})^{rng.randint(8,15)}.",
        lambda: f"Using the distance formula, find the distance between ({rng.randint(-20,20)}, {rng.randint(-20,20)}) and ({rng.randint(-20,20)}, {rng.randint(-20,20)}). Then find the midpoint.",
        lambda: f"Use the compound interest formula to find how long it takes for ${rng.randint(1000,10000)} to double at {rng.randint(3,10)}% annual interest compounded yearly.",
        lambda: f"Apply Heron's formula to find the area of a triangle with sides {rng.randint(7,15)}, {rng.randint(8,18)}, and {rng.randint(9,20)}.",
        lambda: f"Use the arithmetic sequence formulas to find the sum of all multiples of {rng.randint(3,9)} between {rng.randint(1,50)} and {rng.randint(200,500)}.",
    ]
    for i in range(count):
        t = templates[i % len(templates)]
        problems.append({"question": t(), "category": "formula_then_compute", "expected_tools": ["math_cheatsheet", "calculator"]})
    return problems


# ── Conversation Builder ───────────────────────────────────────────────────────

def build_generation_prompt(problem: dict) -> str:
    """Build the user prompt for GPT-4o-mini to generate a conversation."""
    category = problem["category"]
    tools = problem["expected_tools"]

    tool_instruction = ""
    if not tools:
        tool_instruction = """IMPORTANT: This is a SIMPLE problem. Do NOT use any tools. \
Solve it entirely through reasoning. The problem is straightforward enough that \
tools are unnecessary — demonstrate this by noting why tools aren't needed."""
    elif "math_cheatsheet" in tools and "calculator" in tools:
        tool_instruction = """Use BOTH tools in your solution:
- Look up the relevant formula/concept with math_cheatsheet FIRST
- Then use calculator for the numerical computations
Each tool call must be on its own line."""
    elif "math_cheatsheet" in tools:
        topic = problem.get("topic_hint", "")
        tool_instruction = f"""Look up the relevant formula using math_cheatsheet (topic hint: "{topic}"). \
Then apply the formula to solve the problem. You may also use calculator if helpful."""
    elif "calculator" in tools:
        tool_instruction = """Use the calculator tool for computations. For multi-step problems, \
make SEPARATE calculator calls for each computation step."""

    return f"""Write the assistant's response to this math problem.

PROBLEM: {problem['question']}

TOOL GUIDANCE: {tool_instruction}

Remember:
- Start with **Reasoning:** (3-5 sentences of deep analysis)
- Use 3-6 numbered steps with detailed math explanations
- Tool calls: <tool_call>{{"name": "...", "arguments": {{"...": "..."}}}}</tool_call>
- Output ONE tool call at a time and STOP. The system will provide the result.
- End with **Solution:** summary and **Answer:** final answer
- Be EDUCATIONAL — explain the WHY, not just the WHAT
- Predict results before tool calls, interpret after
- Target 500-1500 tokens total"""


def process_llm_output(raw_output: str, category: str) -> list[dict] | None:
    """Parse LLM output into multi-turn conversation messages.

    Splits at tool_call boundaries, evaluates tools, injects tool_response messages.
    Returns list of message dicts or None if parsing fails.
    """
    if not raw_output:
        return None

    messages = []
    # Split the output into segments around tool calls and [TOOL_RESULT] markers
    # Pattern: everything before <tool_call>...</tool_call>\n[TOOL_RESULT]
    segments = re.split(r'(<tool_call>.*?</tool_call>)\s*\n?\s*\[TOOL_RESULT\]', raw_output, flags=re.DOTALL)

    current_text = ""
    for i, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue

        if segment.startswith("<tool_call>"):
            # This is a tool call — save any preceding text as assistant message
            if current_text.strip():
                full_msg = current_text.strip()
                if not full_msg.endswith("\n"):
                    full_msg += "\n"
                full_msg += segment
                messages.append({"role": "assistant", "content": full_msg})
                current_text = ""
            else:
                messages.append({"role": "assistant", "content": segment})

            # Extract and evaluate the tool call
            try:
                call_match = re.search(r'<tool_call>(.*?)</tool_call>', segment, re.DOTALL)
                if call_match:
                    call_json = json.loads(call_match.group(1))
                    tool_name = call_json.get("name", "")
                    tool_args = call_json.get("arguments", {})

                    if tool_name == "calculator":
                        result = safe_calc(tool_args.get("expression", ""))
                        tool_response = json.dumps({"result": result})
                    elif tool_name == "math_cheatsheet":
                        result = lookup_cheatsheet(tool_args.get("topic", ""))
                        tool_response = json.dumps({"content": result})
                    else:
                        tool_response = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    messages.append({
                        "role": "ipython",
                        "content": f"<tool_response>{tool_response}</tool_response>"
                    })
            except (json.JSONDecodeError, KeyError) as e:
                # Invalid tool call — skip this example
                return None
        else:
            current_text += segment + "\n"

    # Any remaining text is the final assistant message
    if current_text.strip():
        messages.append({"role": "assistant", "content": current_text.strip()})

    # Validation
    if not messages:
        return None

    # For no_tool category, merge all into single assistant message
    if category == "no_tool":
        all_text = "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")
        messages = [{"role": "assistant", "content": all_text}]

    # Verify tool_call/ipython pairing
    for j, msg in enumerate(messages):
        if msg["role"] == "ipython":
            if j == 0 or messages[j-1]["role"] != "assistant":
                return None

    return messages


def build_full_conversation(question: str, assistant_messages: list[dict]) -> list[dict]:
    """Build the complete conversation with system + user + assistant turns."""
    conversation = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    conversation.extend(assistant_messages)
    return conversation


# ── Batch Processing ───────────────────────────────────────────────────────────

def process_single_problem(problem: dict) -> dict | None:
    """Generate a single tool-calling example interactively."""
    prompt = build_generation_prompt(problem)
    
    max_turns = 8
    
    # We maintain two lists:
    # 1. messages_payload: what is sent to Nebius (assistant, user for tool responses)
    # 2. assistant_messages: what goes into the final dataset (assistant, ipython)
    messages_payload = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    assistant_messages = []
    
    for turn in range(max_turns):
        raw_output = call_nebius(GENERATION_SYSTEM_PROMPT, "", prior_messages=messages_payload, max_tokens=3000)
        
        if not raw_output:
            return None
            
        messages_payload.append({"role": "assistant", "content": raw_output})
        assistant_messages.append({"role": "assistant", "content": raw_output})
        
        call_match = re.search(r'<tool_call>(.*?)</tool_call>', raw_output, re.DOTALL)
        if call_match:
            try:
                call_json = json.loads(call_match.group(1))
                tool_name = call_json.get("name", "")
                tool_args = call_json.get("arguments", {})
                
                if tool_name == "calculator":
                    result = safe_calc(tool_args.get("expression", ""))
                    tool_response = json.dumps({"result": result})
                elif tool_name == "math_cheatsheet":
                    result = lookup_cheatsheet(tool_args.get("topic", ""))
                    tool_response = json.dumps({"content": result})
                else:
                    tool_response = json.dumps({"error": f"Unknown tool: {tool_name}"})
                
                tool_message = f"<tool_response>{tool_response}</tool_response>"
                
                # API expects user role for tool responses if it doesn't natively support tool role properly
                # We'll inject it as user to keep the conversation going
                messages_payload.append({"role": "user", "content": [{"type": "text", "text": tool_message}]})
                
                # But for our dataset, we use ipython
                assistant_messages.append({"role": "ipython", "content": tool_message})
                
            except (json.JSONDecodeError, KeyError) as e:
                # Invalid tool call — skip this example
                return None
        else:
            # Reached final answer
            break
            
    if not assistant_messages:
        return None

    conversation = build_full_conversation(problem["question"], assistant_messages)

    # Extract answer from the last assistant message
    last_msg = [m for m in conversation if m["role"] == "assistant"][-1]["content"]
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)', last_msg, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    if not answer:
        return None

    # Calculate total assistant token estimate
    total_text = " ".join(m["content"] for m in conversation if m["role"] == "assistant")
    word_count = len(total_text.split())

    # Determine tools actually used
    tools_used = []
    for msg in conversation:
        if "calculator" in msg.get("content", "") and "<tool_call>" in msg.get("content", ""):
            if "calculator" not in tools_used:
                tools_used.append("calculator")
        if "math_cheatsheet" in msg.get("content", "") and "<tool_call>" in msg.get("content", ""):
            if "math_cheatsheet" not in tools_used:
                tools_used.append("math_cheatsheet")

    return {
        "question": problem["question"],
        "conversations": conversation,
        "answer": answer,
        "tools_used": tools_used,
        "category": problem["category"],
        "word_count": word_count,
    }


def process_batch(problems: list[dict]) -> list[dict]:
    """Process a batch of problems in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        future_to_problem = {
            executor.submit(process_single_problem, p): p for p in problems
        }
        for future in as_completed(future_to_problem):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                log(f"  Error processing problem: {e}")
    return results


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("PHASE 4: TOOL-CALLING DATASET BUILDER")
    log("Azure AI Phi-4-mini-instruct | Multi-turn conversations")
    log("=" * 70)

    # Test connectivity
    try:
        test = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Reply with just 'ok'"}],
            max_tokens=5,
        )
        log(f"Azure OpenAI connected: {test.choices[0].message.content.strip()}")
    except Exception as e:
        log(f"ERROR: Cannot connect to Azure OpenAI: {e}")
        sys.exit(1)

    progress = load_progress()
    log(f"Progress: {progress}")

    # Generate all problems
    all_generators = {
        "arithmetic": generate_arithmetic_problems,
        "verification": generate_verification_problems,
        "cheatsheet_lookup": generate_cheatsheet_problems,
        "multi_tool": generate_multi_tool_problems,
        "no_tool": generate_no_tool_problems,
        "multi_step_calc": generate_multi_step_calc_problems,
        "formula_then_compute": generate_formula_then_compute_problems,
    }

    for category, count in CATEGORY_COUNTS.items():
        if category in progress["completed_categories"]:
            log(f"Skipping {category} (already done: {progress['completed_categories'][category]} written)")
            continue

        log(f"\n{'='*50}")
        log(f"CATEGORY: {category} (target: {count})")
        log(f"{'='*50}")

        generator = all_generators[category]
        # Generate more problems than needed to account for failures
        problems = generator(int(count * 1.3))
        rng.shuffle(problems)

        written = 0
        failed = 0
        batch_size = MAX_CONCURRENT_REQUESTS * 2  # Process 6 at a time
        start_time = time.time()

        for batch_start in range(0, len(problems), batch_size):
            if written >= count:
                break

            batch = problems[batch_start:batch_start + batch_size]
            results = process_batch(batch)

            # Write results
            valid = []
            for r in results:
                if written >= count:
                    break
                valid.append(r)
                written += 1

            if valid:
                with open(OUTPUT_FILE, "a") as f:
                    for r in valid:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            failed += len(batch) - len(results)

            if (batch_start + batch_size) % PROGRESS_INTERVAL == 0 or written >= count:
                elapsed = time.time() - start_time
                rate = written / max(elapsed, 1) * 60
                log(f"  {category}: {written}/{count} written, {failed} failed "
                    f"({elapsed:.0f}s, {rate:.1f}/min)")

        log(f"  {category} COMPLETE: {written} written, {failed} failed")
        progress["completed_categories"][category] = written
        progress["total_written"] += written
        save_progress(progress)

    # Final stats
    total = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for _ in f:
                total += 1

    log(f"\n{'='*70}")
    log("TOOL-CALLING DATASET BUILD COMPLETE!")
    log(f"  Output: {OUTPUT_FILE}")
    log(f"  Total records: {total}")

    # Category breakdown
    cats = {}
    with open(OUTPUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("category", "unknown")
            cats[cat] = cats.get(cat, 0) + 1
    for cat, cnt in sorted(cats.items()):
        log(f"    {cat}: {cnt}")

    # Word count stats
    word_counts = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            word_counts.append(r.get("word_count", 0))
    if word_counts:
        log(f"  Avg word count: {sum(word_counts)/len(word_counts):.0f}")
        log(f"  Min/Max: {min(word_counts)}/{max(word_counts)}")
    log(f"{'='*70}")

    # Cleanup progress
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == "__main__":
    main()
