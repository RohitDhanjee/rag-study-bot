import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CompArch Study Bot", page_icon="🖥️")
st.title("🖥️ Computer Architecture Study Bot")
st.caption("Answers from the notes with page citations. Ask in English or Roman Urdu.")

try:
    for key, value in st.secrets.items():        # Streamlit Cloud secrets -> environment variables
        os.environ.setdefault(key, str(value))
except Exception:
    pass                                          # no secrets file locally; .env is used instead

with st.spinner("Loading the AI model (first load takes 1-2 minutes)..."):
    if not Path("index/vectors.npy").exists():   # the index is not in git, so build it on first run
        import build_index  # noqa: F401
    from rag import answer

if "messages" not in st.session_state:          # chat history survives reruns
    st.session_state.messages = []

for m in st.session_state.messages:              # redraw the previous chat
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the notes..."):
            start = time.time()
            reply, results = answer(question)
            latency = time.time() - start
        st.markdown(reply)
        with st.expander(f"Sources ({latency:.1f}s)"):
            for _, c in results:
                st.markdown(f"**Page {c['page']}:** {c['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": reply})
