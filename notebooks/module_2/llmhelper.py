
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from dotenv import load_dotenv
import os
import anthropic
from huggingface_hub import HfApi


from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM, ChatOllama


load_dotenv()

hf_repo_id_llama70b = "meta-llama/Meta-Llama-3.1-70B-Instruct"

def get_model_id():
        
    # Load and run the model:
    hf_repo_id = "HuggingFaceH4/mistral-7b-anthropic"
    
    hf_repo_id = "mistralai/Ministral-8B-Instruct-2410"

    hf_repo_id="meta-llama/Meta-Llama-3-8B-Instruct" 
    
    return hf_repo_id
    
def get_llm(model_name=""): 
    
    #llm = get_llm_anthropic()
    #llm = get_llm_from_huggingface()
    if model_name == "gemini": 
        return get_google_gemini_llm()
    return get_llm_from_ollama()

def get_llm_from_huggingface():
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    #api = HfApi()
    #api.set_access_token(hf_api_key, add_to_git_credential=True)
    login(token=hf_api_key, add_to_git_credential=True) # login to HuggingFace Hub   
    
    hf_repo_id = get_model_id()

    llm_endpoint = HuggingFaceEndpoint(repo_id=hf_repo_id)    
    llm =ChatHuggingFace(llm=llm_endpoint)
    return llm

def get_llm_from_ollama():
    llm = OllamaLLM(model="llama3.2:3b")
    #llm = ChatOllama(model="llama3.2:3b")
    return llm

def get_chatmodel_from_ollama():
    #llm = OllamaLLM(model="llama3.2:3b")
    llm = ChatOllama(model="llama3.2:3b")
    return llm

from langchain_google_genai import ChatGoogleGenerativeAI

def get_google_gemini_llm(): 
    model_name = "models/gemini-1.5-pro"
    print("model name used - ", model_name)
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm


import chromadb
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import tempfile
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langsmith import traceable
from typing import List
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_vector_db_retriever(llm_model):
    persist_path = os.path.join("./temp/", "union.parquet")
    print(" get_vector_db_retriever for model ", llm_model)
    if llm_model.model == "models/gemini-1.5-pro":
        embd = GoogleGenerativeAIEmbeddings(model=llm_model.model)
    else:
        embd = OllamaEmbeddings(model=llm_model.model)
    

    # If vector store exists, then load it
    if os.path.exists(persist_path):
        print("loading existing vectorstore ")
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(lambda_mult=0)
    print("index LangSmith documents and create new vector store : started")
    # Otherwise, index LangSmith documents and create new vector store
    ls_docs_sitemap_loader = SitemapLoader(web_path="https://docs.smith.langchain.com/sitemap.xml")
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    print("index LangSmith documents and create new vector store : completed ")
    return vectorstore.as_retriever(lambda_mult=0)

def get_retriever(llm_model): 
    ollama_embeddings = OllamaEmbeddings(model=llm_model.model)
    embed_fun = create_langchain_embedding(ollama_embeddings)
    vectordb_client = chromadb.PersistentClient()
    vectordb = Chroma(embedding_function=embed_fun, client=vectordb_client)
    retriever = vectordb.as_retriever(search_kwargs={"filter":{"id":"1"}})
    return retriever
    
def get_llm_anthropic():
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )
    """
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022", 
        anthropic_api_key=api_key)
    print("anth model ", model)
    return model


    