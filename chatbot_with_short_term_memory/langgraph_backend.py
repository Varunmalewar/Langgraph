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

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state : ChatState) :
    # take user query from state 
    messages = state['messages']


    # send tp the llm 
    response = llm.invoke(messages)

    # response store state 
    return{
        'messages':[response]
    }

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

graph.add_edge(START,'chat_node')
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



