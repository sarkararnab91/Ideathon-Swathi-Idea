# Create a form using st.form
import os
import sys
import getpass
import json
import PyPDF2
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import streamlit as st
import traceback
from utils import get_table_data,parse_pdf,parse_file,save_temp_file,parse_long_pdf
import pandas as pd
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def prompt_format():
    # Quiz Prompt Template
    quiz_template = """
    Context: {context}
    You are an expert MCQ maker on the {topic}. Given the {context}, it is your job to\
    create a quiz of {number} multiple choice questions for professionals in {difficulty} level.\
    Make sure that questions are not repeated and check all the questions to be conforming to the {context} only as well.
    Make sure to format your response like the RESPONSE_JSON below and use it as a guide.\
    Ensure to make the {number} MCQs.
    ### RESPONSE_JSON
    {response_json}
    """
    quiz_generation_prompt = PromptTemplate(
        input_variables=["context", "topic", "difficulty", "number", "response_json"],
        template=quiz_template,
    )

    RESPONSE_JSON = {
        "1": {
            "no": "1",
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "2": {
            "no": "2",
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "3": {
            "no": "3",
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
    }
    return RESPONSE_JSON,quiz_generation_prompt
with st.form("user_inputs"):
        load_dotenv()
        # File upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
   
        # Input fields
        mcq_count = st.number_input("No of MCQs", min_value=3, max_value=20,placeholder=3)
        topic = st.text_input("Provide a topic", max_chars=100,placeholder="sampling")
        difficulty = st.text_input("Provide Quiz difficulty", max_chars=100, placeholder="simple or complex")
        # Every form must have a submit button.
        #submitted = st.form_submit_button("Submit")
        button = st.form_submit_button("Create quiz")
        
        if uploaded_file is not None: 
            #docs = parse_file(uploaded_file)
            # Save the uploaded file temporarily
            temp_file_path = save_temp_file(uploaded_file)

            # Display file details
            st.write("File details:")
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File type: {uploaded_file.type}")
            st.write(f"File size: {uploaded_file.size} bytes")

            # Load and display PDF content using langchain document loader
            #pdf_loader = PyPDFLoader(temp_file_path)
            retriever = parse_long_pdf(temp_file_path)
            retrieved_document = retriever.get_relevant_documents(topic)
            retrieved_docs = retrieved_document[0].page_content
            print(retrieved_docs)
            # Retrieval: Retrieve
            #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})    
            
        if button and uploaded_file is not None and mcq_count and topic and difficulty: 
        # Check if the button is clicked and all fields have inputs
            with st.spinner("MCQ Generating..."):
                try:
                   #section - I 
                    

                    #section - II
                    # This is an LLMChain to create 10-20 multiple choice questions from a given piece of text.
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                    #section - III
                    embeddings = OpenAIEmbeddings()
                    # section IV
                    RESPONSE_JSON,quiz_generation_prompt = prompt_format()
                    
                    # section V: Testing RAG Output

                    quiz_chain = LLMChain(
                        llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True
                    )
                    # This is the overall chain where we run these two chains in sequence.
                    print("*** This is the topic   " , topic)
                    generate_evaluate_chain = SequentialChain(
                        chains=[quiz_chain],
                        input_variables=["context", "topic", "difficulty", "number", "response_json"],
                        # Here we return multiple variables
                        output_variables=["quiz"],
                        verbose=True,
                    )
                    with get_openai_callback() as cb:
                        response = generate_evaluate_chain(
                        {
                            "context": retrieved_docs,
                            "topic" : topic,
                            "number": mcq_count,
                            "difficulty": difficulty,
                            "response_json": json.dumps(RESPONSE_JSON),
                                    })
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    st.error("Error")               

                else:
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost (USD): ${cb.total_cost}")
            if isinstance(response, dict):
                # Extract quiz data from the response
                quiz = response.get("quiz", None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.to_csv('out.csv')
                        st.write("MCQ is successfully generated") 
                        #df.index = df.index + 1
                        #st.table(df)
                    else:
                        st.error("Error in table data")
            else:
                st.error("Error in table data")
                st.write(response)   

                 