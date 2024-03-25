import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import os

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def vector_database(final_docs_list, embeddings, selected_option, new_or_last_resumes_choice, folder_path, index_name):
    if selected_option=='chroma':
        # Create vector database
        vector_db = VectorDatabase.create(final_docs_list, embeddings, selected_option, new_or_last_resumes_choice, folder_path, index_name)
    
    elif selected_option=='faiss':
        # Create vector database
        vector_db = VectorDatabase.create(final_docs_list, embeddings, selected_option, new_or_last_resumes_choice, folder_path, index_name)
    return vector_db
def main():
    ## Load the dotenv file
    _ = load_dotenv(override=True)
    openai_key = os.getenv('OPENAI_API_KEY')
    huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    repo_id="CohereForAI/c4ai-command-r-v01"#CohereForAI/c4ai-command-r-v01,bigscience/bloom
    
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁 ")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'QUESTIONS' here...",key="1")
    document_count = st.text_input("No.of 'RESUMES' to return", value="1", key="2")
    
    chunk_size_options  = ('No','Yes')
    chunk_size_choice   = st.selectbox('Select whether you need to split resumes into chunks choice Yes, or not choice No: ',chunk_size_options)
    st.write('You selected:', chunk_size_choice )
    chunk_size = st.text_input("No.of 'chunk_size' to return", value="700", key="3")
    
    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    
    embedding_model_options = ('all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'nomic-ai/nomic-embed-text-v1',
                               'jinaai/jina-embeddings-v2-base-en', 'jinaai/jina-embeddings-v2-small-en', 'cointegrated/rubert-tiny2')
    embedding_model_choice  = st.selectbox('Select embedding model:',embedding_model_options)
    st.write('You selected:', embedding_model_choice )

    vector_database_options = ('chroma', 'faiss')
    vector_database_selected_option  = st.selectbox('Select the type of vector database:',vector_database_options)
    st.write('You selected:', vector_database_selected_option )
    
    folder_path = st.text_input("Please paste the folder path to save vector database", value="faiss_index", key="4")
    index_name = st.text_input("Please paste the index name of vector database", value="FaissIndex", key="5")

    new_or_last_resumes_options  = ('No','Yes')
    new_or_last_resumes_choice   = st.selectbox('Select whether you need an answer from new resumes choice Yes, or just from last using resume choice No: ',new_or_last_resumes_options)
    st.write('You selected:', new_or_last_resumes_choice )
    
    summary_options  = ('No','Yes')
    summary_choice   = st.selectbox('Select if you need to create summary to most similar resume choice Yes, or Not choice No: ',summary_options)
    st.write('You selected:', summary_choice )
    

    submit=st.button("Help me with the analysis",key="6")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            final_docs_list = []
            if new_or_last_resumes_choice == 'Yes':
                #Create a documents list out of all the user uploaded pdf files
                final_docs_list=create_docs(pdf,st.session_state['unique_id'], chunk_size,chunk_size_choice)
    
                #Displaying the count of resumes that have been uploaded
                # st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            #Create embeddings instance
            embeddings, embedding_dimension=create_embeddings_model(embedding_model_choice)

            vector_db = vector_database(final_docs_list, embeddings, vector_database_selected_option, new_or_last_resumes_choice, folder_path, index_name)

            similarities_docs = vector_db.get_similarity(job_description, document_count,st.session_state['unique_id'],new_or_last_resumes_choice)
            #t.write(relavant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(similarities_docs)):
                
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+similarities_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show me 👀'): 
                    st.info("**Match Score** : "+str(similarities_docs[item][1]))
                    #st.write("***"+similarities_docs[item][0].page_content)
                    
                    if summary_choice =='Yes':
                        #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                        llm = llm_models(openai_key,repo_id,huggingface_token)
                        summary = get_summary(similarities_docs[item][0],llm)
                        st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time❤️")


#Invoking main function
if __name__ == '__main__':
    main()
