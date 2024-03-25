import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma, Pinecone, Qdrant, FAISS
import pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, PyPDFDirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import time 
# from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#Extract Information from text file 
def get_text(txt_doc):
    loader = TextLoader(txt_doc)
    text = loader.load()
    return text

#Extract Information from text file 
def get_word_text(word_doc):
    loader = Docx2txtLoader(word_doc)#mode="elements",single
    text = loader.load()
    return text

#Extract Information from PDF file
def get_pdf_text_v2(pdf_doc):
    loader = PyPDFLoader(pdf_doc)
    text = loader.load()
    return text

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#This function will split the documents into chunks using RecursiveCharacterTextSplitter
def split_docs(documents, chunk_size=300, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)#CharacterTextSplitter
    docs = text_splitter.split_documents(documents)
    return docs

# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id, chunk_size=700, cond_split_docs='Yes'):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"unique_id":unique_id},
            # metadata={"name": filename.name,"id":filename.id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))
    
    if cond_split_docs=='Yes':
        docs = split_docs(docs, chunk_size=int(chunk_size), chunk_overlap=20)
    return docs

#Create embeddings instance
def create_embeddings_model(embedding_model_choice):
    #embeddings = OpenAIEmbeddings()
    # embedding models all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2, 
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_choice)
    embedding_dimension = embeddings.client[1].word_embedding_dimension
    # embeddings = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cpu')
    # embedding_dimension = embeddings[1].word_embedding_dimension
    return embeddings, embedding_dimension

#----------------------------------------------------------------------------------------------------------------

def llm_models(openai_key,repo_id,huggingface_token):
    # OpenAI.api_key = openai_key
    # llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":1e-10},#Content-Length,, "max_length": 512
                         huggingfacehub_api_token=huggingface_token)
    return llm

class VectorDatabase:
    def __init__(self, database):
        self.database = database

    @classmethod
    def create(cls, docs, embeddings, database_type, new_or_last_resumes_options, folder_path, index_name):
        if database_type == 'chroma':
            if new_or_last_resumes_options == 'Yes':
                db = Chroma.from_documents(docs, embeddings, persist_directory=folder_path,collection_name=index_name)
                return cls(db)
            elif new_or_last_resumes_options == 'No':
                db = Chroma(persist_directory=folder_path, embedding_function=embeddings,collection_name=index_name)
                return cls(db)
            else:
                raise ValueError("Invalid option for new_or_last_resumes_options")
        elif database_type == 'faiss':
            if new_or_last_resumes_options == 'Yes':
                db = FAISS.from_documents(docs, embeddings)
                db.save_local(folder_path=folder_path, index_name=index_name)
                return cls(db)
            elif new_or_last_resumes_options == 'No':
                db = FAISS.load_local(folder_path=folder_path, embeddings=embeddings, index_name=index_name)#, allow_dangerous_deserialization=True
                return cls(db)
            else:
                raise ValueError("Invalid option for new_or_last_resumes_options")
        else:
            raise ValueError("Invalid database type")

    def get_similarity(self, query_embedding, k=2, unique_id=None,new_or_last_resumes_choice='Yes'):
        if (new_or_last_resumes_choice=='Yes') and (unique_id is not None):
            return self.database.similarity_search_with_score(query_embedding, int(k), {"unique_id": unique_id})
        elif (new_or_last_resumes_choice=='Yes') and (unique_id is None):
            return self.database.similarity_search_with_score(query_embedding, int(k))
        elif (new_or_last_resumes_choice=='No'):
            return self.database.similarity_search_with_score(query_embedding, int(k))

# Helps us get the summary of a document
def get_summary(current_doc,llm):
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary
