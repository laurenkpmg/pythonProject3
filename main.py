# https://youtu.be/Dh0sWMQzNH4

import os
from fastapi import FastAPI
from PyPDF2 import PdfReader
import docx  # pip install python-docx
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import openai
import pandas as pd
import pickle
from starlette.middleware.cors import CORSMiddleware


def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

############ TEXT LOADERS ############
# Functions to read different file types
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_txt(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    return text


def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text



def save_strings_as_files(string_list, directory):
    """
    Saves each string in a list as a separate text file in the specified directory.

    :param string_list: List of strings to be saved.
    :param directory: Directory where the text files will be saved.
    """
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save each string in a separate file
    for idx, string in enumerate(string_list):
        file_path = os.path.join(directory, f'string_{idx}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(string)

    print(f"Saved {len(string_list)} files in '{directory}'.")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_vars = {}


@app.on_event('startup')
async def init():

    # Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
    openai.api_base = "https://testopenai4.openai.azure.com/"
    openai.api_key = "19f61bb2a418484dbe809cb720c2527b"
    openai.api_version = "2023-05-15"
    openai.api_type = "azure"

    df = pd.read_excel('data/maccabi_table.xlsx').dropna().drop_duplicates()
    # Removing the number and the last hyphen from the 'SG_TREAT_NAME' column
    treat_names = df['SG_TREAT_NAME'].astype(str)

    # Removing the number and the last hyphen from each string
    processed_names = treat_names.str.rsplit('-', n=1).str[0].str.strip()

    # Converting the processed names to a list
    processed_names_list = processed_names.tolist()

    processed_names_list = list(set(processed_names_list))

    # save_strings_as_files(processed_names_list, 'C:\\Users\\ltobaly\\PycharmProjects\\pythonProject3\\app\\vector_db')

    text_chunks = processed_names_list
    # char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
    #                                            chunk_overlap=200, length_function=len)
    # # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_base = "https://testopenai4.openai.azure.com/",
                                  openai_api_key = "19f61bb2a418484dbe809cb720c2527b",
                                  openai_api_version = "2023-05-15",
                                  openai_api_type = "azure",
                                  deployment="text-embedding-ada-002")

    llm_vars['pdfDocSearch'] = FAISS.from_texts(text_chunks, embeddings)
    #
    question = ''
    documents =[]
    # docs = llm_vars['pdfDocSearch'].similarity_search(inquiry, k=50)

    llm = AzureChatOpenAI(temperature=0, max_tokens=800, openai_api_base=openai.api_base,
                        openai_api_key=openai.api_key,
                        openai_api_version=openai.api_version, deployment_name="gpt-4-32k-testing")

    prompt_template = 'for each of the following services, decide whether the user query is relevant to it in terms of content.' \
                      'output the service name + yes/no if relevant or not' \
                      'user query: {question}' \
                      'categories: {documents}'
    #
    prompt_template = PromptTemplate(input_variables=["documents", 'question'], template= prompt_template)
    llm_vars['chain'] = load_qa_chain(llm=llm, chain_type="stuff", verbose=True, prompt=prompt_template, document_variable_name='documents')

@app.post("/question")
def llm_question(req: dict):
    inquiry = req["prompt"]
    docs = llm_vars['pdfDocSearch'].similarity_search(inquiry, k=50)
    res = llm_vars['chain'].run(input_documents=docs, question=inquiry)
    categories = res.split('\n')
    categories = [{'cpt':categories[i][:-6]} for i in range(len(categories)) if 'Yes' in categories[i]]
    return categories


