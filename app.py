import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CompArch Study Bot", page_icon="🖥️")
st.title("🖥️ Computer Architecture Study Bot")
st.caption("Notes se jawab, page number ke saath. English ya Roman Urdu mein poochiye.")

try:
    for key, value in st.secrets.items():        # Streamlit Cloud ke secrets -> environment variables
        os.environ.setdefault(key, str(value))
except Exception:
    pass                                          # laptop par secrets file nahi hoti, .env kaam karti hai

with st.spinner("AI model load ho raha hai (pehli dafa 1-2 minute)..."):
    if not Path("index/vectors.npy").exists():   # server par index nahi hota (.gitignore), pehli dafa banao
        import build_index  # noqa: F401
    from rag import answer

if "messages" not in st.session_state:          # chat history, har rerun par bachi rehti hai
    st.session_state.messages = []

for m in st.session_state.messages:              # purani chat dobara dikhao
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Sawal poochiye..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Notes dhoond raha hoon..."):
            start = time.time()
            reply, results = answer(question)
            latency = time.time() - start
        st.markdown(reply)
        with st.expander(f"Sources ({latency:.1f}s)"):
            for _, c in results:
                st.markdown(f"**Page {c['page']}:** {c['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": reply})