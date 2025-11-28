import streamlit as st
from PyPDF2 import PdfReader  # Correct import
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    else:
        return file.read().decode("utf-8", errors="ignore").strip()

st.title("📘 Knowledge Base AI Agent (Online Demo – Free)")

docs = st.file_uploader("Upload documents (PDF/TXT)", type=["pdf","txt"], accept_multiple_files=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.df = None

if st.button("Process Documents"):
    if not docs:
        st.error("Upload at least one document!")
    else:
        all_text = []
        for d in docs:
            all_text.append(extract_text(d))

        lines = []
        for doc_text in all_text:
            for line in doc_text.split("\n"):
                if line.strip():
                    lines.append(line.strip())

        embeddings = model.encode(lines)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))

        st.session_state.index = index
        st.session_state.df = pd.DataFrame({"text": lines})

        st.success("✅ FAISS index built! Ask questions now.")

question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if st.session_state.index is None:
        st.error("Process docs first!")
    else:
        q_embedding = model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embedding, dtype="float32"), 1)
        idx = I[0][0]
        st.write("🤖 Most relevant answer from docs:")
        st.write(st.session_state.df.iloc[idx]["text"])
