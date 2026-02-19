import streamlit as st
from connect_memory_with_llm import build_qa_chain

# PAGE CONFIG 
st.set_page_config(
    page_title="Beyond Addiction Support Bot",
    page_icon="🚭",
    layout="centered"
)

# TITLE 
st.title("🚭 Beyond Addiction Support Bot")
st.caption("Counselling & guidance to help you quit drug abuse")

#  DISCLAIMER 
st.info(
    "This chatbot provides guidance and emotional support for quitting substance abuse. "
    "It does not provide medical advice. For medical concerns, consult a healthcare professional."
)

# LOAD QA CHAIN 
@st.cache_resource
def load_chain():
    return build_qa_chain()

qa_chain = load_chain()

# SESSION STATE 
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY CHAT HISTORY 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#USER INPUT 
user_input = st.chat_input("Ask about quitting smoking, drug abuse...")

if user_input:

    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(user_input)
            except Exception as e:
                response = "⚠️ Sorry, something went wrong. Please try again."

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


with st.sidebar:
    st.header("💡 Quick Help")
    st.write("**Try asking:**")
    st.write("- How can I manage drug cravings safely?")
    st.write("- How long does recovery from narcotics dependence take?")
    st.write("- What are healthy ways to cope with urges to use?")
    st.write("- When should someone seek medical help ?")


    st.divider()

    st.write("📞 **India Quitline:**")
    st.write("1800-11-2356")

    st.divider()
    st.caption("Stay strong — every step counts.")
