# SOTA MATH Phase 4 Agent Evaluation

This directory contains the scripts required to evaluate your Phase 4 tool-calling Math LLM using **vLLM**, **Langchain**, and **LangSmith**.

## Prerequisites (for your RTX 3060 Rig)

1. **Python Environment**:
   ```bash
   pip install vllm langchain langchain-openai langsmith datasets wikipedia sympy
   ```

2. **Environment Variables**:
   You must set these before running the scripts to allow LangSmith metrics tracking and GPT-4o-mini judging.
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY="your-langsmith-api-key"
   
   # For the GPT-4o-mini Evaluator
   export AZURE_OPENAI_API_KEY="your-azure-key"
   export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
   ```

## Step 1: Create the Evaluation Dataset
Run the setup script. This will download a perfectly balanced 500-question mix from GSM8k, MATH, and NuminaMath-CoT (AIME/AMC) and push it to your LangSmith project.
```bash
python3 setup_eval.py
```

## Step 2: Run the Agent Evaluation Loop
Run the evaluation script. This will load your `Aniket200325/SOTA_MATH-phase4` model onto your RTX 3060 using vLLM in `bfloat16`. 
It will then iterate through the dataset, execute `<tool_call>` outputs locally using the robust Python tools (`agent_tools.py`), and have GPT-4o-mini judge the final answers.
```bash
python3 run_eval.py
```

## How It Works
- `agent_tools.py`: Contains a highly robust `calculator` (based on SymPy) and a `math_cheatsheet` (which uses the Wikipedia API to look up concepts).
- `agent_eval.py`: The core vLLM inference loop. It applies your system prompt strictly and prevents the model from looping infinitely by enforcing a max iteration cap per problem conceptually, while giving it the freedom to take as many sequential tool-call steps as it needs.
- `run_eval.py`: Orchestrates the dataset run with LangSmith, using two Custom Evaluators:
  - **Tool Usage**: Counts how many times the model successfully utilized the calculator/cheatsheet.
  - **Correctness**: Uses `gpt-4o-mini` to do an equivalency check between the true answer and the model's generated answer.
