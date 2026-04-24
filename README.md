# Agentic Data Analyst

An autonomous AI agent that analyses CSV datasets by dynamically writing 
and executing Python code, interpreting results, and delivering 
business-friendly insights — without any human intervention.

## How it Works
User Question
↓
Agent calls get_dataset_info → understands the data
↓
Agent writes Python code → executes via run_python_code tool
↓
Agent interprets results → iterates if needed
↓
Final business insight delivered
## Architecture
- **ReAct Loop** — Reason → Act → Observe → Repeat
- **Tool Calling** — Agent dynamically selects and calls the right tool
- **Self-Correcting** — If a tool fails, agent adapts and tries alternative approach

## Tools Available
| Tool | Purpose |
|---|---|
| `get_dataset_info` | Understand schema, dtypes, nulls |
| `get_summary_statistics` | Statistical overview of all columns |
| `get_correlation_matrix` | Find relationships between numeric features |
| `get_value_counts` | Distribution of categorical columns |
| `run_python_code` | Execute custom Python for deep analysis |

## Tech Stack
- LangChain Core for tool calling and message orchestration
- Groq API (LLaMA 3.3 70B) for high-quality reasoning
- Pandas, NumPy, Matplotlib, Seaborn for data analysis
- Python-dotenv for secure API management

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
GROQ_API_KEY=your_key_here
```bash
cd src
python main.py
```

## Sample Questions You Can Ask
- "What are the top 3 factors driving customer churn?"
- "What is the average monthly charge for churned vs non-churned customers?"
- "Show me the correlation between tenure and monthly charges"
- "Which customer segment has the highest churn rate?"

## Sample Output
Question: What is the average monthly charge for churned vs non-churned customers?
🔧 Calling tool: get_dataset_info
🔧 Calling tool: run_python_code → df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
🔧 Calling tool: run_python_code → df[df['Churn'] == 'No']['MonthlyCharges'].mean()
FINAL ANALYSIS:
Churned customers pay $74.44/month on average vs $61.27 for
non-churned — a 21% premium suggesting price sensitivity is
a key churn driver.