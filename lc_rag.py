import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import load_pdf
from rag import SYSTEM_PROMPT
from search import tokenize

load_dotenv()

# 1. Load: our own load_pdf (with header/footer cleaning) -> LangChain Documents
pdf = next(Path("data").glob("*.pdf"))
pages = [Document(page_content=p["text"], metadata={"page": p["page"]}) for p in load_pdf(pdf)]

# 2. Split: tries to break on paragraphs, then sentences, then words
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)

# 3. Embed + store: E5's "query: " / "passage: " prefixes must be set explicitly
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
    query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
)
if Path("faiss_index").exists():
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")

# 4. Retrieve: semantic (FAISS) + keyword (BM25), merged with RRF
bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize, k=20)
faiss = vectorstore.as_retriever(search_kwargs={"k": 20})
retriever = EnsembleRetriever(retrievers=[faiss, bm25], weights=[0.5, 0.5])

# 5. Generate: prompt -> LLM -> plain text
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Context:\n{context}\n\nQuestion: {question}"),
])
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, api_key=os.environ["GROQ_API_KEY"])
chain = prompt | llm | StrOutputParser()


def retrieve(question, k=3):
    return retriever.invoke(question)[:k]


def answer(question, k=3):
    found = retrieve(question, k)
    context = "\n\n".join(f"[page {d.metadata['page']}]\n{d.page_content}" for d in found)
    reply = chain.invoke({"context": context, "question": question})
    results = [(0.0, {"page": d.metadata["page"], "text": d.page_content}) for d in found]
    return reply, results            # same format as rag.answer(), so evaluate.py works with both


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (empty Enter = exit): ").strip()
        if not question:
            break
        reply, results = answer(question)
        print("\n" + reply)
        print(f"\n(Retrieved pages: {sorted({c['page'] for _, c in results})})")
