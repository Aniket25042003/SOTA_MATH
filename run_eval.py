import os
from langsmith import evaluate, Client
from agent_eval import solve_math_problem
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangSmith Client
client = Client()
DATASET_NAME = "Math_Agent_Eval_Top_500"

# Set up Azure OpenAI Judge Model (for evaluating correctness)
# Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
try:
    judge_llm = AzureChatOpenAI(
        api_version="2024-12-01-preview",
        azure_deployment="gpt-4o-mini",
        temperature=0.0
    )
except Exception as e:
    judge_llm = None
    print("Warning: AzureChatOpenAI not configured properly:", e)

def math_agent_target(inputs: dict) -> dict:
    """The function that runs the VLLM agent and returns the final prediction."""
    question = inputs["question"]
    
    # Run loop
    messages = solve_math_problem(question, max_iterations=20)
    
    # Extract the final answer and total tool usage
    tool_calls_count = sum(1 for msg in messages if msg["role"] == "ipython")
    final_output = messages[-1]["content"] if messages else ""
    
    # Simple regex to extract what follows "**Answer:**" 
    # Or just return the whole block for the Judge LLM
    final_answer = ""
    ans_idx = final_output.rfind("**Answer:**")
    if ans_idx != -1:
        final_answer = final_output[ans_idx + len("**Answer:**"):].strip()
    else:
        ans_idx = final_output.rfind("Answer:")
        if ans_idx != -1:
            final_answer = final_output[ans_idx + len("Answer:"):].strip()
        else:
            final_answer = final_output # Fallback
            
    return {
        "prediction": final_answer,
        "full_reasoning_trace": final_output,
        "tool_calls": tool_calls_count,
        "history": messages
    }

def correctness_evaluator(run, example) -> dict:
    """A custom evaluator using GPT-4o-mini to grade math equivalency."""
    if judge_llm is None:
        return {"key": "correctness", "score": 0, "comment": "Judge LLM not configured."}
        
    ground_truth = example.outputs.get("ground_truth", "")
    prediction = run.outputs.get("prediction", "")
    
    # Prompt the judge
    prompt = PromptTemplate.from_template(
        "You are an expert math grader.\n\n"
        "Question: {question}\n\n"
        "Ground Truth Solution/Answer: {ground_truth}\n\n"
        "Student's Predicted Final Answer: {prediction}\n\n"
        "Are the Student's prediction and the Ground Truth materially equivalent? "
        "Reply exactly with 'YES' or 'NO' followed by a brief reason."
    )
    
    chain = prompt | judge_llm | StrOutputParser()
    try:
        response = chain.invoke({
            "question": example.inputs["question"],
            "ground_truth": ground_truth,
            "prediction": prediction
        })
        
        score = 1 if response.strip().upper().startswith("YES") else 0
        return {"key": "correctness", "score": score, "comment": response}
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"Eval failed: {e}"}

def tool_usage_evaluator(run, example) -> dict:
    """Evaluates how many tool calls were made."""
    count = run.outputs.get("tool_calls", 0)
    return {"key": "tool_calls_made", "score": count}

def evaluate_model():
    print(f"Starting evaluation on dataset '{DATASET_NAME}'...")
    
    experiment_results = evaluate(
        math_agent_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, tool_usage_evaluator],
        experiment_prefix="sota-math-phase4-eval",
        max_concurrency=1, # vLLM runs locally, so concurrency should be 1 unless batching
    )
    
    print("Evaluation launched! View results in the LangSmith UI.")

if __name__ == "__main__":
    evaluate_model()
