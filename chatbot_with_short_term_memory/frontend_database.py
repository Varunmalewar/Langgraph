import streamlit as st 

from langchain_core.messages import HumanMessage
import streamlit as st
from langgraph_backend import workflow
import uuid # har bar ek nayi id generate kar sakte ho 
from langgraph_backend import retrieve_all_threads, delete_all_threads, workflow

#***********************************utility function ********************************
def generate_thread_id():
    thread_id = str(uuid.uuid4())  # Generate a unique thread ID using UUID
    return thread_id

def reset_chat():
    st.session_state.thread_id = generate_thread_id()  # Generate a new thread ID
    add_thread(st.session_state.thread_id)  # Add the new thread ID to the list of chat threads
    st.session_state.message_history.clear()  # Clear the message history IN PLACE (keeps the same list object so the UI updates on this same run)

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)  # Add the new thread ID to the list of chat threads
    if thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[thread_id] = "New Chat"  # Give the thread a default name until the first message arrives

def get_thread_title(thread_id):
    return st.session_state.chat_titles.get(thread_id, "New Chat")

def load_conversation(thread_id):
    return workflow.get_state(config={
        "configurable": {
            "thread_id": thread_id  # use the parameter, not the current session's thread_id
        }
    })  # Load the conversation state for the given thread ID

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))  # keep only the "text" field of each block
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)  # fallback for any other type

#****************************************session setup ****************************


# st.session_state -> dict
if "message_history" not in st.session_state:
    st.session_state.message_history = []

message_history = st.session_state.message_history

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread_id()  # Generate a new thread ID

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()  # Initialize an empty list to store chat threads

if 'chat_titles' not in st.session_state:
    st.session_state.chat_titles = {}  # Map of thread_id -> display name shown in the sidebar

add_thread(st.session_state.thread_id)  # Add the current thread ID to the list of chat threads

#*************************** Side bar UI ************************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()  # Reset the chat and generate a new thread ID

if st.sidebar.button("🗑️ Clear All Chats"):
    delete_all_threads()  # Permanently wipe every saved conversation from the database
    st.session_state.chat_threads = []  # Empty the sidebar thread list
    st.session_state.chat_titles = {}  # Forget all thread titles
    st.session_state.message_history.clear()  # Clear the currently displayed chat
    st.session_state.thread_id = generate_thread_id()  # Start fresh with a brand-new thread
    st.rerun()  # Rerun so the sidebar and chat area reflect the empty state immediately

st.sidebar.title("My Conversations")


for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(get_thread_title(thread_id), key=f"btn_{thread_id}", help=f"Thread: {thread_id}"):  # show meaningful name, keep thread_id as unique key + tooltip
        st.session_state.thread_id = thread_id  # Set the selected thread ID in session state
        snapshot = load_conversation(thread_id)  # get_state() returns a StateSnapshot object
        messages = snapshot.values.get("messages", [])  # extract the actual list of LangChain messages from the snapshot
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": extract_text(message.content)})
            else:
                temp_messages.append({"role": "assistant", "content": extract_text(message.content)})
        # Update the message history IN PLACE so the render loop below (which holds a
        # reference to the same list) sees the loaded conversation on this same run
        st.session_state.message_history.clear()
        st.session_state.message_history.extend(temp_messages)

# Build the config AFTER the sidebar, so it reflects the thread the user just selected
# (otherwise a message sent right after switching threads would go to the old thread)
config1 = {
    "configurable":{
        "thread_id": st.session_state.thread_id
    }
}

#***************************Main UI ****************************************


for message in message_history :
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here")

if user_input:

    # Name the thread after its first user message (only if it's still unnamed)
    if st.session_state.chat_titles.get(st.session_state.thread_id, "New Chat") == "New Chat":
        st.session_state.chat_titles[st.session_state.thread_id] = (
            user_input[:30] + "..." if len(user_input) > 30 else user_input
        )

    #first add the message to message
    st.session_state.message_history.append({"role":"user","content":user_input})
    with st.chat_message('user'):
        st.text(user_input)



    
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.text for message_chunk, meta_data in workflow.stream(
                {
                    'messages': [HumanMessage(content=user_input)]
                },
                config=config1,
                stream_mode = "messages"
            )
        )
    st.session_state.message_history.append({"role":"assistant","content":ai_message})


