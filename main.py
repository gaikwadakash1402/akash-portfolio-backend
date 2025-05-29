import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # For request body validation
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import asyncio  # For running synchronous code in an async context

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Akash Gaikwad's AI Portfolio Chatbot Backend",
    description="A RAG-based AI assistant to answer questions about Akash's professional background and AI/Data Science topics.",
    version="1.0.0",
)

# --- CORS Configuration ---
# Allows requests from your frontend domain.
# IMPORTANT: Replace 'https://YOUR_GITHUB_PAGES_URL' with your actual deployed GitHub Pages URL!
# Example: "https://gaikwadakash1402.github.io"
origins = [
    "http://localhost:5000",  # Local frontend dev server
    "http://127.0.0.1:5000",  # Local frontend dev server
    "http://localhost:8000",  # Local backend dev server
    "http://127.0.0.1:8000",  # Local backend dev server
    "https://gaikwadakash1402.github.io",  # <--- REPLACE WITH YOUR ACTUAL GITHUB PAGES URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- Configuration Constants ---
DATA_DIR = 'data'  # Relative path within the backend folder
RESUME_FILE = os.path.join(DATA_DIR, 'resume.txt')
PROJECTS_FILE = os.path.join(DATA_DIR, 'projects.txt')
AI_DS_INFO_FILE = os.path.join(DATA_DIR, 'ai_ds_info.txt')

# --- Google Gemini API Configuration ---
# IMPORTANT: Your API Key must be set as an environment variable on Render!
# For local testing, you can set it in a .env file, but Render handles it securely.
GOOGLE_API_KEY = os.getenv("AIzaSyCkgpkEEaHOv0445yOKU5RZbNM2HnWLuRE")

# Configure the Google Generative AI client
if not GOOGLE_API_KEY:  # Check if API key is loaded
    print("WARNING: GOOGLE_API_KEY environment variable is not set. Chatbot may not function correctly.")
    # For local testing, you might want to hardcode it here temporarily for debugging,
    # but remove before committing to Git for security if not using .env
    # genai.configure(api_key="YOUR_HARDCODED_API_KEY_FOR_LOCAL_DEBUG_ONLY")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Use 'gemini-pro' for text generation.
GEMINI_MODEL_NAME = 'gemini-pro'
# Initialize model only if API key is present
gemini_llm = genai.GenerativeModel(GEMINI_MODEL_NAME) if GOOGLE_API_KEY else None

# --- Global Data Structures for RAG ---
corpus_chunks = []
embedding_model = None
corpus_embeddings = None


# --- Data Loading and Processing ---
def load_text_data(filepath: str, label_prefix: str) -> list[str]:
    """Loads text from a file, splits into chunks, and adds to corpus_chunks."""
    # Ensure file exists before trying to read it
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found: {filepath}. Skipping {label_prefix} data loading.")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

    # Optionally, for resume, split by single newline for finer retrieval
    if label_prefix == 'resume':
        finer_chunks = []
        for chunk in chunks:
            finer_chunks.extend([line.strip() for line in chunk.split('\n') if line.strip()])
        return finer_chunks

    return chunks


async def initialize_chatbot_data():
    """Loads all data, initializes embedding model, and generates embeddings.
    This runs once on application startup.
    """
    global embedding_model, corpus_embeddings, corpus_chunks

    print("--- Initializing Chatbot Data ---")

    # Ensure the data directory exists before trying to load from it
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and populate with .txt files.")
        # Attempt to create it, but warn the user it needs populating
        os.makedirs(DATA_DIR, exist_ok=True)
        # return # Might not want to return if we can create empty files

    # Load data from the text files
    corpus_chunks.extend(load_text_data(RESUME_FILE, "resume"))
    corpus_chunks.extend(load_text_data(PROJECTS_FILE, "project"))
    corpus_chunks.extend(load_text_data(AI_DS_INFO_FILE, "ai_ds"))

    if not corpus_chunks:
        print("Error: No data chunks loaded. Chatbot will have limited information.")
        return

    print(f"Loaded {len(corpus_chunks)} text chunks for embedding.")

    print("Loading Sentence Transformer model (all-MiniLM-L6-v2)... This might take a moment.")
    # Run synchronous model loading in a thread pool to not block FastAPI's event loop
    embedding_model = await asyncio.to_thread(SentenceTransformer, 'all-MiniLM-L6-v2', cache_folder='./.cache_model')
    print("Embedding model loaded.")

    print("Generating corpus embeddings...")
    # Run synchronous embedding generation in a thread pool
    corpus_embeddings = await asyncio.to_thread(embedding_model.encode, corpus_chunks, convert_to_tensor=True,
                                                show_progress_bar=False)
    corpus_embeddings = corpus_embeddings.cpu()  # Ensure it's on CPU if not already
    print("Corpus embeddings generated.")
    print("--- Chatbot Initialization Complete ---")


# --- Generative LLM Interaction Function (for Google Gemini) ---
# This function is synchronous and will be run in a thread pool by FastAPI.
def get_generative_llm_response(query: str, retrieved_context: str) -> str:
    """
    Sends a request to the Google Gemini API to generate a response.
    """
    if gemini_llm is None:  # Check if model was initialized (i.e., API key was present)
        raise HTTPException(status_code=500, detail="LLM not initialized. GOOGLE_API_KEY may be missing.")

    if not retrieved_context:
        # If no context, still try to answer general AI/DS questions
        if "ai" in query.lower() or "data science" in query.lower():
            prompt = f"""
            You are an AI assistant. You don't have personal information about Akash Gaikwad.
            Answer the following question about AI/Data Science generally.
            Question: {query}
            Answer:
            """
        else:
            return "I couldn't find enough specific information in my knowledge base related to Akash to answer that. Could you please rephrase or ask about general AI/Data Science topics?"
    else:  # If context is available, use RAG prompt
        prompt = f"""
        You are Akash Gaikwad's AI assistant. Your primary goal is to answer questions about Akash's professional background,
        projects, skills, and general AI/Data Science concepts based *only* on the provided context.

        If the context does not contain enough specific information to answer the question,
        state that you don't have enough specific information, but try to provide a general
        answer if it's an AI/Data Science concept that you know generally.

        Context:
        {retrieved_context}

        Question: {query}

        Answer:
        """

    try:
        response = gemini_llm.generate_content(
            prompt.strip(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Re-raise as HTTPException for FastAPI to handle
        raise HTTPException(status_code=500,
                            detail=f"LLM API error: {e}. Please check API key, internet connection, or try again later.")


# --- Request Body Model for Chat Endpoint ---
# Defines the expected structure of the JSON payload for the /chat endpoint
class ChatRequest(BaseModel):
    message: str


# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the application starts, to load data and models.
    """
    # Ensure data directory exists. Render will handle file presence if committed.
    if not os.path.exists(DATA_DIR):
        print(
            f"Warning: Data directory '{DATA_DIR}' not found. It might be created during deployment or already exists.")
        # On Render, the 'data' folder will be present if committed to Git.
        # This check is primarily for local dev.

    await initialize_chatbot_data()
    print("\n--- Using Google Gemini API for LLM. ---")
    print("--- Backend ready for requests! ---")
    print("--------------------------------------------------------------------------\n")


# --- FastAPI Routes ---
# Add this new OPTIONS route handler for /chat (Crucial for CORS preflight)
@app.options("/chat")
async def chat_options():
    """
    Handles CORS preflight requests for the /chat endpoint.
    This ensures proper CORS handshake before the actual POST request.
    """
    return {}  # An empty dictionary or 200 OK is sufficient for preflight success


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    API endpoint for chatbot messages.
    Performs RAG (Retrieval-Augmented Generation) to answer user queries.
    """
    user_message = request.message
    if not user_message:
        raise HTTPException(status_code=400, detail="No message received.")

    query_lower = user_message.lower()

    # Basic Greetings (handled directly for quick, fixed responses)
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
        return {
            "response": "Hello! How can I assist you today regarding Akash Gaikwad's expertise, projects, or general AI and Data Science concepts?"}
    elif "your name" in query_lower or "who are you" in query_lower:
        return {"response": "I am Akash's AI Assistant, designed to answer questions about him, AI, and Data Science."}
    elif any(thanks in query_lower for thanks in ["thank you", "thanks", "appreciate it"]):
        return {"response": "You're welcome! Feel free to ask anything else."}

    # Ensure chatbot is initialized before proceeding with RAG
    if embedding_model is None or corpus_embeddings is None:
        raise HTTPException(status_code=503,
                            detail="Chatbot is still initializing or encountered an error during startup. Please try again in a moment.")

    # Ensure gemini_llm is initialized (API key present)
    if gemini_llm is None:
        raise HTTPException(status_code=500,
                            detail="LLM not configured. GOOGLE_API_KEY environment variable is likely missing on the server.")

    try:
        user_query_embedding = await asyncio.to_thread(embedding_model.encode, user_message, convert_to_tensor=True)
        user_query_embedding = user_query_embedding.cpu()

        hits = await asyncio.to_thread(util.semantic_search, user_query_embedding, corpus_embeddings, top_k=3)
        hits = hits[0]

        retrieved_chunks = []
        for hit in hits:
            if hit['score'] > 0.4:  # Only consider relevant hits
                retrieved_chunks.append(corpus_chunks[hit['corpus_id']])

        context = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""

        # LLM response generation is synchronous, run in thread pool
        llm_response = await asyncio.to_thread(get_generative_llm_response, user_message, context)
        return {"response": llm_response}

    except HTTPException as e:  # Re-raise HTTPExceptions from get_generative_llm_response
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Basic Root Endpoint (Optional: for checking if backend is alive) ---
@app.get("/")
async def read_root():
    """Returns a simple message to indicate the backend is running."""
    return {"message": "Akash's AI Portfolio Chatbot Backend is running!"}

# How to run this file locally for development:
# 1. Ensure you have the 'data' folder and your .txt files.
# 2. Set your GOOGLE_API_KEY as an environment variable in your terminal
#    (e.g., set GOOGLE_API_KEY=YOUR_KEY on Windows, or export GOOGLE_API_KEY=YOUR_KEY on Linux/Mac)
#    OR use a .env file and 'python-dotenv'.
# 3. Open your terminal in the 'backend' directory.
# 4. Run: uvicorn main:app --reload --port 8000
# The '--reload' flag automatically reloads the server on code changes.
# The '--port 8000' sets the port to 8000, matching your frontend's default.