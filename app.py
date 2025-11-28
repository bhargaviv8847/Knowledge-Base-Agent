import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text helper
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

# UI
st.title("📘 Knowledge Base AI Agent (Free & Online Demo)")

uploaded_files = st.file_uploader("Upload documents (PDF/TXT)", type=["pdf","txt"], accept_multiple_files=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if st.button("Process Documents"):
    if not uploaded_files:
        st.error("Please upload at least one document!")
    else:
        texts = []
        for file in uploaded_files:
            texts.append(extract_text(file))

        combined_text = "\n\n".join(texts)
        chunks = [c.strip() for c in combined_text.split("\n") if c.strip()]

        st.session_state.chunks = chunks

        # Generate embeddings
        embeddings = model.encode(chunks)

        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))

        st.session_state.index = index
        st.success("✅ Documents processed and indexed successfully!")

st.markdown("### ❓ Ask Questions From Your Docs")

question = st.text_input("Enter your question here")

if st.button("Get Answer"):
    if st.session_state.index is None:
        st.error("Process documents first before asking questions!")
    elif not question.strip():
        st.error("Please enter a question!")
    else:
        q_embedding = model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embedding, dtype="float32"), 3)

        st.markdown("### 🤖 Answer")
        
        # Combine 3 most relevant lines for answer generation
        answer_lines = " ".join([st.session_state.chunks[i] for i in I[0]])
        
        st.write(answer_lines)
