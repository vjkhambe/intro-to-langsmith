
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

    hf_repo_id = "openai-community/openai-gpt"
    
    return hf_repo_id
    
def get_llm(): 
    
    #llm = get_llm_anthropic()
    #llm = get_llm_from_huggingface()
    llm = get_llm_from_ollama()
    return llm





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


import chromadb
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

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


    