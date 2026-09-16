import asyncio
import os
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

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


def build_graph():
    async def chat_node(state : ChatState) :
        """LLm node that may answer or requests a tool call"""
        # take user query from state 
        messages = state['messages']


        # send to the llm (ainvoke = async version of invoke)
        response = await llm_with_tools.ainvoke(messages)

        # response store state 
        return{
            'messages':[response]
        }

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node('chat_node',chat_node)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,'chat_node')
    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools','chat_node')
    graph.add_edge('chat_node',END)

    chatbot = graph.compile()
    return chatbot





async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke({"messages":[HumanMessage(content = "Find the modulus of 132354 and 23 and give answer like a cricket commentator")]})

    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())