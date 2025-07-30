import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from textblob import TextBlob

# 1. Load env
load_dotenv()

# 2. Build the QA chain once
def build_chain(faq_path="faqs.txt"):
    # a) Load text file
    loader   = TextLoader(faq_path)
    docs     = loader.load()

    # b) Split into chunks (helps with long FAQ files)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    # c) Embed & index
    embeddings  = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # d) Wrap into a QA chain
    return RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=vectorstore.as_retriever()
    )

qa_chain = build_chain()

def correct_spelling(text: str) -> str:
    blob = TextBlob(text)
    # .correct() can be a bit slow on very long text, but it's fine for FAQs
    return str(blob.correct())

def get_response(question: str) -> str:
    # 1. Correct common typos
    clean_q = correct_spelling(question)
    # 2. Run the QA chain
    return qa_chain.run(clean_q)
