import os
import re
import json
from vllm import LLM, SamplingParams
from agent_tools import calculator, math_cheatsheet

# Initialize vLLM with BF16 precision for RTX 3060
# (Assuming the model is downloaded or using HF hub path)
MODEL_PATH = "Aniket200325/SOTA_MATH-phase4"

print("Loading model via vLLM...")
# Note: trust_remote_code=True is usually needed for Llama-3 tokenizers
try:
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,  # Single RTX 3060
        max_model_len=4096,
        gpu_memory_utilization=0.9
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to a mock LLM for testing the script without GPU if needed
    llm = None

# Sampling params for generation (greedy decoding is usually best for math/tools)
sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, stop=["</tool_call>", "<|eot_id|>"])

# System Prompt with explicit Infinite Loop guardrails
SYSTEM_PROMPT = """You are an expert mathematical AI assistant.
You have access to the following tools:
1. calculator — Evaluate a mathematical expression accurately.
   Usage: <tool_call>{"name": "calculator", "arguments": {"expression": "YOUR_EXPR"}}</tool_call>
   The system will respond with: <tool_response>{"result": "VALUE"}</tool_response>

2. math_cheatsheet — Look up a mathematical concept, formula, or theorem.
   Usage: <tool_call>{"name": "math_cheatsheet", "arguments": {"topic": "YOUR_TOPIC"}}</tool_call>
   The system will respond with: <tool_response>{"content": "RELEVANT_INFO"}</tool_response>

CRITICAL INSTRUCTIONS:
- You must ONLY use the exact JSON format specified above for tools.
- Do NOT hallucinate tools. Only use 'calculator' and 'math_cheatsheet'.
- After you receive a <tool_response>, carefully analyze the result.
- Avoid repeating the same tool call if you already have the answer. DO NOT loop infinitely. If a tool fails, try a different approach.
- Once you reach the final answer, state it clearly using "**Answer:** [your final answer]".
"""

def extract_tool_call(text: str):
    """Parses <tool_call>{...json...} if it exists at the end of the generated text."""
    # We look for the last tool call generated in this turn
    match = re.search(r'<tool_call>(.*?)$', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        # Clean up any trailing tags if the model generated them
        if json_str.endswith("</tool_call>"):
             json_str = json_str[:-12]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def run_agent_turn(messages: list) -> tuple[str, bool]:
    """Runs a single generation turn, executes a tool if called, and returns the response."""
    if llm is None:
        raise RuntimeError("vLLM model not loaded.")
        
    # We apply the chat template. Note vLLM handles this automatically if tokenizer has it.
    # But since we have a custom tool flow, we might need to manually format or rely on HF chat template.
    # Llama-3.1 standard chat template:
    prompt = ""
    for msg in messages:
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    
    # Prompt the assistant to continue
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    
    is_done = True
    
    # Check if a tool call was made
    if "<tool_call>" in generated_text:
        is_done = False
        tool_call_data = extract_tool_call(generated_text)
        
        # Append the assistant's generation (including the <tool_call> tag)
        if not generated_text.endswith("</tool_call>"):
             generated_text += "</tool_call>"
        messages.append({"role": "assistant", "content": generated_text})
        
        if tool_call_data and "name" in tool_call_data and "arguments" in tool_call_data:
            tool_name = tool_call_data["name"]
            args = tool_call_data["arguments"]
            
            tool_result = ""
            if tool_name == "calculator" and "expression" in args:
                res = calculator(args["expression"])
                tool_result = f'<tool_response>{{"result": "{res}"}}</tool_response>'
            elif tool_name == "math_cheatsheet" and "topic" in args:
                res = math_cheatsheet(args["topic"])
                tool_result = f'<tool_response>{{"content": "{res}"}}</tool_response>'
            else:
                tool_result = f'<tool_response>{{"error": "Invalid tool or arguments."}}</tool_response>'
                
            messages.append({"role": "ipython", "content": tool_result})
        else:
            # Failed to parse JSON
            messages.append({"role": "ipython", "content": '<tool_response>{"error": "Failed to parse tool JSON."}</tool_response>'})
            
    else:
        # No tool call, must be the final answer
        messages.append({"role": "assistant", "content": generated_text})
        
    return generated_text, is_done

def solve_math_problem(question: str, max_iterations=15) -> list:
    """Runs the full agent loop for a single math problem."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    for i in range(max_iterations):
        _, is_done = run_agent_turn(messages)
        if is_done:
            break
            
    return messages

if __name__ == "__main__":
    test_q = "What is the area of a right triangle with legs of length 3 and 4?"
    print(f"Testing Agent Loop with question: {test_q}")
    # history = solve_math_problem(test_q)
    # for msg in history:
    #     print(f"--- {msg['role'].upper()} ---")
    #     print(msg['content'])
    print("Agent script initialized successfully.")
