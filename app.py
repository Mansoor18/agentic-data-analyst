import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import sys
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Load API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page config ──
st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# ── Global dataframe ──
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# ── Tools ──
def get_dataset_info_fn():
    df = st.session_state.df
    if df is None:
        return "No dataset loaded."
    info = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"]
    info.append("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isnull().sum()
        info.append(f"  - {col}: {dtype} ({nulls} nulls)")
    info.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(info)

def run_python_code_fn(code):
    df = st.session_state.df
    if df is None:
        return "No dataset loaded."
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec_globals = {
            "df": df, "pd": pd, "np": np,
            "plt": plt, "st": st, "os": os
        }
        exec(code, exec_globals)
        output = buffer.getvalue()
        return output if output else "Code executed successfully."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

def get_summary_statistics_fn():
    df = st.session_state.df
    if df is None:
        return "No dataset loaded."
    return df.describe(include='all').to_string()

def get_correlation_matrix_fn():
    df = st.session_state.df
    if df is None:
        return "No dataset loaded."
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.corr().to_string()

def get_value_counts_fn(column):
    df = st.session_state.df
    if df is None:
        return "No dataset loaded."
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"
    counts = df[column].value_counts()
    pct = df[column].value_counts(normalize=True) * 100
    result = pd.DataFrame({'count': counts, 'percentage': pct.round(2)})
    return result.to_string()

# ── Tool definitions for LLM ──
TOOLS = [
    {
        "name": "get_dataset_info",
        "description": "Get basic information about the dataset including shape, columns, dtypes and missing values. Always call this first.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "run_python_code",
        "description": "Execute Python code for data analysis. The dataframe is available as 'df'. Always print your results.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"]
        }
    },
    {
        "name": "get_summary_statistics",
        "description": "Get summary statistics for all columns.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_correlation_matrix",
        "description": "Get correlation matrix for numeric columns.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_value_counts",
        "description": "Get value counts for a specific column.",
        "input_schema": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Column name"}},
            "required": ["column"]
        }
    }
]

TOOL_FN_MAP = {
    "get_dataset_info": lambda args: get_dataset_info_fn(),
    "run_python_code": lambda args: run_python_code_fn(args.get("code", "")),
    "get_summary_statistics": lambda args: get_summary_statistics_fn(),
    "get_correlation_matrix": lambda args: get_correlation_matrix_fn(),
    "get_value_counts": lambda args: get_value_counts_fn(args.get("column", ""))
}

SYSTEM_PROMPT = """You are an expert Data Analyst agent.
You have access to tools to analyse a dataset.
Always start by calling get_dataset_info to understand the data.
Then use other tools to answer the user's question thoroughly.
Write and execute Python code for deeper analysis when needed.
Always explain results in plain, business-friendly English."""

def run_agent(question, log_container):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    # Bind tools
    llm_with_tools = llm.bind_tools(
        [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        } for t in TOOLS]
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    logs = []
    final_answer = ""

    for i in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_answer = response.content
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            log_msg = f"🔧 **{tool_name}**"
            if tool_name == "run_python_code":
                log_msg += f"\n```python\n{tool_args.get('code', '')}\n```"
            logs.append(log_msg)

            # Update log display
            with log_container:
                for log in logs:
                    st.markdown(log)

            # Execute tool
            if tool_name in TOOL_FN_MAP:
                result = TOOL_FN_MAP[tool_name](tool_args)
            else:
                result = f"Tool '{tool_name}' not found"

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            ))

    return final_answer, logs

# ── Header ──
st.title("🤖 Agentic Data Analyst")
st.markdown("*Upload any CSV — ask questions in plain English — watch the agent think*")
st.divider()

# ── Sidebar ──
st.sidebar.header("📊 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.chat_history = []
    st.session_state.agent_logs = []
    st.sidebar.success(f"✅ Loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

    st.sidebar.subheader("Dataset Preview")
    st.sidebar.dataframe(st.session_state.df.head(3), use_container_width=True)

    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Sample questions ──
if st.session_state.df is not None:
    st.subheader("💬 Ask anything about your data")

    sample_questions = [
        "What are the top 3 factors driving customer churn?",
        "What is the average monthly charge for churned vs non-churned customers?",
        "Show me the distribution of tenure across different contract types",
        "Which customer segment has the highest churn rate?"
    ]

    selected_q = st.selectbox("Try a sample question or write your own:",
                               ["Write my own..."] + sample_questions)

    if selected_q == "Write my own...":
        question = st.chat_input("Ask a question about your dataset...")
    else:
        question = None
        if st.button(f"▶ Run: {selected_q}", use_container_width=True):
            question = selected_q

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Process question
    if question:
        st.session_state.chat_history.append({
            "role": "user", "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            st.markdown("**🔄 Agent is working...**")
            log_container = st.container()
            with st.spinner("Analysing..."):
                answer, logs = run_agent(question, log_container)

            st.divider()
            st.markdown("**📊 Analysis Result:**")
            st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer
        })

else:
    st.info("👈 Upload a CSV file to get started")
    st.subheader("About this App")
    st.markdown("""
    This app implements a **ReAct (Reason → Act → Observe → Repeat)** agentic loop:

    | Step | What happens |
    |---|---|
    | 1️⃣ Reason | Agent decides which tool to use |
    | 2️⃣ Act | Agent calls the tool |
    | 3️⃣ Observe | Agent reads the output |
    | 4️⃣ Repeat | Agent iterates until question is answered |

    **Available Tools:**
    - `get_dataset_info` — understand schema and data types
    - `run_python_code` — write and execute custom analysis
    - `get_summary_statistics` — statistical overview
    - `get_correlation_matrix` — feature relationships
    - `get_value_counts` — category distributions

    **Powered by:** LLaMA 3.3 70B via Groq API
    """)