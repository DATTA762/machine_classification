# import os
# import pickle
# import faiss
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer

# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.prompts import ChatPromptTemplate

# import google.generativeai as genai

# # ----------------------------
# # CONFIG
# # ----------------------------
# genai.configure(api_key="AIzaSyDrZml9GJYlRBgIXyZjxBUmsyF2T5Z60jY")
# model = genai.GenerativeModel("gemini-2.5-flash")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# RESUME_FOLDER = "resumes"

# # ----------------------------
# # Job role required skills mapping
# # ----------------------------
# JOB_SKILLS = {
#     "Python Developer": ["python", "django", "sql", "html", "css", "javascript", "react", "fullstack"],
#     "Data Science Developer": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "ml", "statistics"],
#     "ML Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "numpy", "pandas"],
#     "AI Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "nlp", "cv"],
#     "Fullstack Developer": ["html", "css", "javascript", "react", "python", "django", "sql"]
# }

# # ----------------------------
# # PDF Loader
# # ----------------------------
# def load_pdf(file_path):
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#     return text

# # ----------------------------
# # Chunk Text
# # ----------------------------
# def chunk_text(text, chunk_size=200, overlap=40):
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks

# # ----------------------------
# # Build FAISS Index
# # ----------------------------
# def build_index(folder):
#     all_chunks = []
#     metadata = []

#     for file in os.listdir(folder):
#         if not file.endswith(".pdf"):
#             continue
#         text = load_pdf(os.path.join(folder, file))
#         chunks = chunk_text(text)
#         for chunk in chunks:
#             all_chunks.append(chunk)
#             metadata.append({"text": chunk, "source": os.path.basename(file)})

#     embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)

#     faiss.write_index(index, "resume.index")
#     with open("resume_docs.pkl", "wb") as f:
#         pickle.dump(metadata, f)

# # ----------------------------
# # Load FAISS Index
# # ----------------------------
# def load_index():
#     index = faiss.read_index("resume.index")
#     with open("resume_docs.pkl", "rb") as f:
#         docs = pickle.load(f)
#     return index, docs

# # ----------------------------
# # Retrieve Relevant Chunks
# # ----------------------------
# def retrieve(query, index, docs, top_k=5):
#     query_vec = embedder.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(query_vec, top_k)
#     return [docs[i] for i in indices[0]]

# # ----------------------------
# # Calculate ATS score based on required skills
# # ----------------------------
# def calculate_ats_score(text, job_role):
#     skills = JOB_SKILLS.get(job_role, [])
#     if not skills:
#         return 0
#     matches = sum(1 for skill in skills if skill.lower() in text.lower())
#     return int((matches / len(skills)) * 100)

# # ----------------------------
# # Resume Assistant using GenAI with ATS filter
# # ----------------------------
# def resume_assistant(query, index, docs, top_k=5, min_score=60):
#     retrieved_docs = retrieve(query, index, docs, top_k)
#     eligible_docs = []

#     # Filter resumes by ATS score
#     for doc in retrieved_docs:
#         score = calculate_ats_score(doc["text"], query)
#         if score >= min_score:
#             eligible_docs.append({**doc, "ats_score": score})

#     if not eligible_docs:
#         return "No eligible resumes found with ATS score >= 60%", []

#     context_text = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(eligible_docs)])

#     prompt = ChatPromptTemplate.from_template("""
# You are an AI recruitment assistant.

# Analyze the resume content and return structured output.

# Provide:
# 1. Candidate Skills
# 2. Extracted Projects
# 3. ATS Score
# 4. Skill Match with Job Role
# 5. Interview Questions

# Rules:
# - Extract projects only from the resume context.

# Context:
# {context}

# Job Role:
# {job_role}
# """)

#     formatted_prompt = prompt.format(context=context_text, job_role=query)
#     response = model.generate_content(formatted_prompt)

#     # Return both GenAI output and individual ATS scores
#     ats_scores = {doc["source"]: doc["ats_score"] for doc in eligible_docs}

#     return response.text, ats_scores

# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     # build_index(RESUME_FOLDER)  # Run once to create the index
#     index, docs = load_index()
#     job_query = "Python Developer"

#     output, ats_scores = resume_assistant(job_query, index, docs)
#     print("\n=== Resume Assistant Output ===\n")
#     print(output)
#     print("\n=== ATS Scores for Eligible Resumes ===\n", ats_scores)
import os
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from groq import Groq   # ✅ NEW

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("API key not found! Please set GROQ_API_KEY in your .env file.")
#     st.stop()

client = Groq(api_key=GROQ_API_KEY)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
RESUME_FOLDER = "resumes"

# ----------------------------
# Job role required skills mapping
# ----------------------------
JOB_SKILLS = {
    "Python Developer": ["python", "django", "sql", "html", "css", "javascript", "react", "fullstack"],
    "Data Science Developer": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "ml", "statistics"],
    "ML Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "numpy", "pandas"],
    "AI Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "nlp", "cv"],
    "Fullstack Developer": ["html", "css", "javascript", "react", "python", "django", "sql"]
}

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
# Build FAISS Index
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
            metadata.append({"text": chunk, "source": os.path.basename(file)})

    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "resume.index")
    with open("resume_docs.pkl", "wb") as f:
        pickle.dump(metadata, f)

# ----------------------------
# Load FAISS Index
# ----------------------------
def load_index():
    index = faiss.read_index("resume.index")
    with open("resume_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

# ----------------------------
# Retrieve Relevant Chunks
# ----------------------------
def retrieve(query, index, docs, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [docs[i] for i in indices[0]]

# ----------------------------
# Calculate ATS score
# ----------------------------
def calculate_ats_score(text, job_role):
    skills = JOB_SKILLS.get(job_role, [])
    if not skills:
        return 0
    matches = sum(1 for skill in skills if skill.lower() in text.lower())
    return int((matches / len(skills)) * 100)

# ----------------------------
# Resume Assistant (Groq)
# ----------------------------
def resume_assistant(query, index, docs, top_k=5, min_score=60):
    retrieved_docs = retrieve(query, index, docs, top_k)
    eligible_docs = []

    for doc in retrieved_docs:
        score = calculate_ats_score(doc["text"], query)
        if score >= min_score:
            eligible_docs.append({**doc, "ats_score": score})

    if not eligible_docs:
        return "No eligible resumes found with ATS score >= 60%", []

    context_text = "\n\n".join(
        [f"[{i+1}] {doc['text']}" for i, doc in enumerate(eligible_docs)]
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI recruitment assistant.

Analyze the resume content and return structured output.

Provide:
1. Candidate Skills
2. Extracted Projects
3. ATS Score
4. Skill Match with Job Role
5. Interview Questions

Rules:
- Extract projects only from the resume context.

Context:
{context}

Job Role:
{job_role}
""")

    formatted_prompt = prompt.format(context=context_text, job_role=query)

    # ✅ GROQ API CALL
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an AI recruitment assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.3
    )

    ats_scores = {doc["source"]: doc["ats_score"] for doc in eligible_docs}

    return response.choices[0].message.content, ats_scores

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # build_index(RESUME_FOLDER)  # Run once
    index, docs = load_index()

    job_query = "Python Developer"

    output, ats_scores = resume_assistant(job_query, index, docs)

    print("\n=== Resume Assistant Output ===\n")
    print(output)

    print("\n=== ATS Scores for Eligible Resumes ===\n")
    print(ats_scores)