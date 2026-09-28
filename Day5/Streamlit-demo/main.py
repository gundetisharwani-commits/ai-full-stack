import streamlit as st
st.title("Welcome to streamlit!")
#text box component
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")
# agree=st.checkbox("i agree")
# if agree:
#     st.write("thankyou!")