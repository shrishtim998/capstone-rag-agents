
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="E-commerce Customer Support",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ E-commerce Customer Support")
st.caption("Ask questions about orders, payments, refunds, delivery, and accounts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("How can I help you today?")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                answer = result.get(
                    "answer",
                    "Sorry, I could not find an answer.",
                )
                st.markdown(answer)

                if result.get("sources"):
                    with st.expander("Answer source"):
                        st.json(result["sources"])

                if result.get("needs_escalation"):
                    st.warning("This question may require human support.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except requests.RequestException as exc:
                st.error(
                    "Could not connect to the support API. "
                    "Check that FastAPI is running."
                )
                st.caption(str(exc))

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
