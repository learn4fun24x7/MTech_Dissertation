import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# from ConversationalAIAgent import graph_builder
from AgenticWorkflow import graph_builder

st.set_page_config(page_title="Sanjeevani — Virtual Care Assistant", layout="centered")

st.title("🩺 Sanjeevani — Virtual Care Assistant")

st.info("""
Welcome to **Sanjeevani**

I can help you with:
- Generic healthcare related questions
- Symptom assessment and finding specialists
- Booking doctor appointments
- Ordering Medicines
- Medication reminders

Please describe your health concern.
""")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# User input
user_input = st.chat_input("Type Here...")

if user_input:

    response = graph_builder.invoke({'messages': user_input}, config=CONFIG, debug=True)
    print(response)

    # Render full conversation
    for msg in response["messages"]:

        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                clean_text = msg.content.replace("\\n", "\n")
                st.markdown(clean_text, unsafe_allow_html=False)

        elif isinstance(msg, AIMessage):

            # Skip empty messages
            if not msg.content or msg.content.strip() == "":
                continue

            with st.chat_message("assistant"):
                clean_text = msg.content.replace("\\n", "\n")
                st.markdown(clean_text, unsafe_allow_html=False)