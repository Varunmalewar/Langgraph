from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import SecretStr,BaseModel , Field 
from typing import TypedDict,NotRequired,Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import operator
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
)


search_tool = DuckDuckGoSearchRun(region = "us-en")

@tool 
def calculator(first_num : float, second_num : float , operation :str )->dict :
    """Perform a basic arithmetic operation on two numbers . Supported operations : add , sub , mul , div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool 
def get_stock_price(symbol : str)->dict:
    """Fetch latest stock price for a given symbol (e.g., AAPL for Apple Inc.) using Alpha Vantage API key in the url."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.environ['ALPHA_VANTAGE_API_KEY']}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return {"error": f"Failed to fetch stock price for {symbol}. Status code: {r.status_code}"}



tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state : ChatState) :
    """LLm node that may answer or requests a tool call"""
    # take user query from state 
    messages = state['messages']


    # send tp the llm 
    response =  llm_with_tools.invoke(messages)

    # response store state 
    return{
        'messages':[response]
    }

tool_node = ToolNode(tools)

config1 = {
    "configurable":{
        "thread_id": "thread-1"
    }
}

# Use an ABSOLUTE path anchored to this file's folder, so the same chatbot.db is
# used no matter which directory you launch `streamlit run` from.
# (A relative path silently creates a NEW empty DB when the working directory differs,
#  which makes all previously saved threads "disappear".)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chatbot.db")

connection = sqlite3.connect(database=DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn = connection)
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

workflow = graph.compile(checkpointer=checkpointer)



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


def delete_all_threads():
    # Wipe every saved conversation from the checkpointer database.
    # This SqliteSaver version uses the tables 'checkpoints' and 'writes'
    # ('checkpoint_migrations' only stores schema info, so it is kept).
    cursor = connection.cursor()
    tables = [row[0] for row in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    for table in tables:
        if table == "writes" or (table.startswith("checkpoint") and table != "checkpoint_migrations"):
            cursor.execute(f"DELETE FROM {table}")
    connection.commit()


# from IPython.display import Image
# png_bytes = workflow.get_graph().draw_mermaid_png()
# with open(os.path.join(BASE_DIR, "graph.png"), "wb") as f:
#     f.write(png_bytes)
# print("Graph saved to:", os.path.join(BASE_DIR, "graph.png"))
