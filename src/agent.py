from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from tools import (get_dataset_info, run_python_code,
                   get_summary_statistics, get_correlation_matrix,
                   get_value_counts, set_dataframe)
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

TOOLS = [
    get_dataset_info,
    run_python_code,
    get_summary_statistics,
    get_correlation_matrix,
    get_value_counts
]

SYSTEM_PROMPT = """You are an expert Data Analyst agent with deep knowledge 
of Python, Pandas, statistics, and data visualization.

You have access to the following tools:
- get_dataset_info: Always call this FIRST to understand the dataset
- get_summary_statistics: Get statistical summary of columns
- get_correlation_matrix: Find correlations between numeric columns
- get_value_counts: Understand distribution of categorical columns
- run_python_code: Execute Python code for custom analysis

Your workflow:
1. ALWAYS start by calling get_dataset_info
2. Explore the data systematically
3. Write and execute Python code for deeper analysis
4. Provide clear, business-friendly insights
5. Always explain what the numbers mean, not just what they are

Be thorough, precise, and always interpret results in plain English."""

def create_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # upgrade to 70B
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    ).bind_tools(TOOLS)
    return llm

def run_agent_loop(llm, question, max_iterations=10):
    from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    for i in range(max_iterations):
        response = llm.invoke(messages)
        messages.append(response)
        
        # If no tool calls — final answer
        if not response.tool_calls:
            return response.content
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"\n🔧 Calling tool: {tool_name}")
            print(f"   Args: {tool_args}")
            
            # Find and run the tool
            tool_map = {t.name: t for t in TOOLS}
            if tool_name in tool_map:
                try:
                    if tool_args:
                        first_arg = list(tool_args.values())[0]
                        result = tool_map[tool_name].invoke(first_arg)
                    else:
                        result = tool_map[tool_name].invoke({})
                except Exception as e:
                    result = f"Tool error: {str(e)}"
            else:
                result = f"Tool '{tool_name}' not found"
            
            print(f"   Result preview: {str(result)[:200]}...")
            
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            ))
    
    return "Max iterations reached."

def analyse(csv_path, question):
    print(f"\n{'='*60}")
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    set_dataframe(df)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nQuestion: {question}")
    print(f"{'='*60}\n")

    llm = create_agent()
    result = run_agent_loop(llm, question)
    
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS:")
    print(f"{'='*60}")
    print(result)
    return result