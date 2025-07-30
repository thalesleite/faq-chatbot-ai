# chatbot.py

import os
from dotenv import load_dotenv
from textblob import TextBlob
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()  # loads OPENAI_API_KEY from .env

def build_chain(faq_path: str = "faqs.txt") -> RetrievalQA:
    """Load FAQs, build embeddings & FAISS index, and return a RetrievalQA chain."""
    # 1. Load FAQ document
    loader = TextLoader(faq_path)
    docs = loader.load()

    # 2. Split into manageable chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Embed and index with FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Create a RetrievalQA chain using a Chat model
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=vectorstore.as_retriever()
    )
    return qa

# Build the chain once at import time
qa_chain = build_chain()

def correct_spelling(text: str) -> str:
    """Lightweight spell‐correction via TextBlob."""
    blob = TextBlob(text)
    return str(blob.correct())

def get_response(question: str) -> str:
    """
    1) Correct common typos in the user’s question.
    2) Run the RetrievalQA chain to find the most relevant FAQ answer.
    """
    clean_q = correct_spelling(question)
    return qa_chain.run(clean_q)
