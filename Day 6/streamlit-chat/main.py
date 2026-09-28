import streamlit as st
from ollama import chat
st.title("MY AI BOT: Synora")
name="sharwani"
st.write(f"welcome {name} to my AI bot")
question = st.text_input("Ask me a question")
if st.button("ask"):
    st.write(f"you asked: {question}")
if question:
    response=chat(model="gemma3:1b",messages=[{
        "role":"system", 
        "content":"you are a friendly tutor.you explain in 35-40words shortly and clearly" #can give constraints to the model 
    },
    {
        "role":"user",
        "content":question

    }])
    answer=response.message.content
    st.write(f"Answer: {answer}")
clear=st.checkbox("clear")
if clear:
    st.write("Thankyou")
#dropdown
# options=["c++","python","java","javascript"]
# choice=st.selectbox("select your favorite programming language",options)
# st.write(f'you selected: {choice}')
