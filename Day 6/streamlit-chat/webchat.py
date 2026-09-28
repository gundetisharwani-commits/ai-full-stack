import streamlit as st
from ollama import chat

st.title("YOUR AICHATBOT")
# messages=[]
system_message="you are a shakespherean tutor.you explain in 35-40words shortly and clearly"
# messages.append({
#     "role":"system",
#     "content":system_message
# })
if "messages" not in st.session_state:
    st.session_state.messages=[
        { 
        "role":"system",
        "content":system_message
        }
    ]
#DISPLAY PREVIOUS MESSAGES
for message in st.session_state.messages:
    #dont display system message
    if message["role"]!="system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
question=st.chat_input("Ask me a question...")
if question:
    with st.chat_message('user'):
        st.write(question)
        st.session_state.messages.append({
            "role":".user",
            "content":question
        })
        with st.spinner("thinking....💭"):
            response=chat(model="gemma3:1b",messages=st.session_state.messages)
            answer=response.message.content
            with st.chat_message('assistant'):
                st.write(answer)
            st.session_state.messages.append({
                "role":"assistant",
                "content":answer
            })
