import openai
import streamlit as st

with st.sidebar:
    openai_api_key =""
 
st.title("💬 Chatbot")
st.caption("🚀 A chatbot powered by Gen AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# If user enters a prompt
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Call the OpenAI API to get a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    # Extract the assistant's reply from the API response
    msg = response.choices[0].message["content"]

    # Append assistant message to session state and display it
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
