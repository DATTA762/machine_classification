import os
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

# ----------------------------
# Load ENV
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Resume Assistant API")

# ----------------------------
# CONFIG
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
RESUME_FOLDER = "resumes"

JOB_SKILLS = {
    "Python Developer": ["python", "django", "sql", "html", "css", "javascript", "react", "fullstack"],
    "Data Science Developer": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "ml", "statistics"],
    "ML Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "numpy", "pandas"],
    "AI Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "nlp", "cv"],
    "Fullstack Developer": ["html", "css", "javascript", "react", "python", "django", "sql"]
}

# ----------------------------
# Request Model
# ----------------------------
class QueryRequest(BaseModel):
    job_role: str
    min_score: int = 60

# ----------------------------
# PDF Loader
# ----------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ----------------------------
# Chunk Text
# ----------------------------
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ----------------------------
# Build Index
# ----------------------------
def build_index(folder):
    all_chunks = []
    metadata = []

    for file in os.listdir(folder):
        if not file.endswith(".pdf"):
            continue
        text = load_pdf(os.path.join(folder, file))
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"text": chunk, "source": file})

    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "resume.index")

    with open("resume_docs.pkl", "wb") as f:
        pickle.dump(metadata, f)

# ----------------------------
# Load Index
# ----------------------------
def load_index():
    if not os.path.exists("resume.index"):
        build_index(RESUME_FOLDER)

    index = faiss.read_index("resume.index")

    with open("resume_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    return index, docs

# Load once (startup optimization)
index, docs = load_index()

# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [docs[i] for i in indices[0]]

# ----------------------------
# ATS Score
# ----------------------------
def calculate_ats_score(text, job_role):
    skills = JOB_SKILLS.get(job_role, [])
    if not skills:
        return 0

    matches = sum(1 for skill in skills if skill in text.lower())
    return int((matches / len(skills)) * 100)

# ----------------------------
# Core Logic
# ----------------------------
def process_resumes(job_role, min_score):
    retrieved_docs = retrieve(job_role)
    eligible_docs = []

    for doc in retrieved_docs:
        score = calculate_ats_score(doc["text"], job_role)
        if score >= min_score:
            eligible_docs.append({**doc, "ats_score": score})

    if not eligible_docs:
        return {"message": "No eligible resumes found", "results": []}

    context_text = "\n\n".join(
        [f"[{i+1}] {doc['text']}" for i, doc in enumerate(eligible_docs)]
    )

    prompt = f"""
You are an AI recruitment assistant.

Analyze the resume content and return structured output.

Provide:
1. Candidate Skills
2. Extracted Projects
3. ATS Score
4. Skill Match with Job Role
5. Interview Questions

Context:
{context_text}

Job Role:
{job_role}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an AI recruitment assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    ats_scores = {doc["source"]: doc["ats_score"] for doc in eligible_docs}

    return {
        "analysis": response.choices[0].message.content,
        "ats_scores": ats_scores
    }

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/analyze")
def analyze_resumes(request: QueryRequest):
    return process_resumes(request.job_role, request.min_score)