import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"  # ✅ stable

def build_qa_chain():

    # LLM (HF)
    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)
    prompt = ChatPromptTemplate.from_template("""
You are a supportive smoking cessation counseling assistant.

Use the provided context to answer the user's question.

Guidelines:
- Provide supportive and empathetic responses
- Offer practical coping strategies
- Encourage positive progress
- Do not provide medical diagnosis
- If answer is not in context, say you don't know

Context:
{context}

Question:
{question}
""")
    

    # Embeddings (HF – SAME as FAISS creation)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = (
        {
            "context": retriever,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain