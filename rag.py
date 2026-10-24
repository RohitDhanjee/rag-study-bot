import os

from dotenv import load_dotenv
from groq import Groq

from search import search

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a study assistant for computer architecture notes.
Answer ONLY using the context below. Every context block starts with its page number.
Rules:
- If the answer is not in the context, say exactly: "Ye notes mein nahi hai."
- After each fact, cite the page like [page 4].
- Keep the answer short and clear. Reply in the same language as the question."""


def build_context(results):
    """Top chunks ko ek text mein jodo, har ek ke aage page number."""
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for _, c in results)


def answer(question, k=3):
    results = search(question, k=k)
    context = build_context(results)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content, results


if __name__ == "__main__":
    while True:
        question = input("\nSawal poochiye (khali Enter = exit): ").strip()
        if not question:
            break
        reply, results = answer(question)
        print("\n" + reply)
        pages = sorted({c["page"] for _, c in results})
        print(f"\n(Search ne ye pages diye: {pages})")