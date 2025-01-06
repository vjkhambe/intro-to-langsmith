import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langsmith import traceable
from typing import List
import nest_asyncio
#from common_genai_utils.geminihelper import 
from common_genai_utils.llmhelper import get_llm, get_retriever, retrieve_docs


APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

#llm_client = get_llm()
#print("llm model " , llm_client)
#nest_asyncio.apply()
#web_url = "https://docs.smith.langchain.com/sitemap.xml"
#retriever = retrieve_docs(web_url, is_sitemap_url=True)
#print("llm model retriever " , retriever)

MODEL_NAME = "llama3.2"
MODEL_PROVIDER = "Gool"


"""
retrieve_documents
- Returns documents fetched from a vectorstore based on the user's question
"""
@traceable(run_type="chain")
def retrieve_documents(retriever, question: str):
    return retriever.invoke(question)

"""
generate_response
- Calls `call_openai` to generate a model response after formatting inputs
"""
@traceable(run_type="chain")
def generate_response(llm_client, question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_model(llm_client, messages)

"""
call_openai
- Returns the chat completion output from OpenAI
"""
@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_model(llm_client, messages: List[dict]) -> str:
    return llm_client.invoke(messages)

"""
langsmith_rag
- Calls `retrieve_documents` to fetch documents
- Calls `generate_response` to generate a response based on the fetched documents
- Returns the model response
"""
@traceable(run_type="chain")
def langsmith_rag(question: str, llm_client, retriever):
    
    documents = retrieve_documents(retriever, question)
    response = generate_response(llm_client, question, documents)
    return response
