from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import SecretStr,BaseModel , Field 
from typing import TypedDict,NotRequired,Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import operator
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # ram wali memory

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
checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow = graph.compile(checkpointer=checkpointer)



