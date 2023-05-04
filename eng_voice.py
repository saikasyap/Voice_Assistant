
#Libraries used 
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchTool
import torch
from transformers import pipeline
from langchain.embeddings import (
    LlamaCppEmbeddings, 
    HuggingFaceEmbeddings, 
    SentenceTransformerEmbeddings
)
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (
    PyPDFLoader,
    DataFrameLoader,
    GitLoader
  )
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.chains.question_answering import load_qa_chain
import requests
import base64
import wave
import json
import gradio as gr

# Set up the API endpoint URL
url = "https://tts-api.ai4bharat.org/"

print(os.getcwd())

# For English Embeddings  
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wikipedia Wrapper
wikipedia = WikipediaAPIWrapper()

def wiki_run(question):
    context = wikipedia.run(question).replace('\n', '')
    return context


# DuckduckGO
search = DuckDuckGoSearchTool()
def duck_run(question):
    context = search.run(question)
    return context

#Transcribe ASR Output
def transcribe(audio, pipe):
    text = pipe(audio)["text"]
    return text

#Book based Search 
def book_search(question,book,chain):
    if book == "Ramayana":
        index = "pdf_book1"
        #query = "What are the qualtities of Hanuman?"
 
    else:
        index = "pdf_book2"
        #query = "how did abhimanyu die?"

    docsearch = FAISS.load_local(index,embeddings)
    docs = docsearch.similarity_search(question)
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
    return result




# Loading Language models based on language 
def load_local_LangModel(question,option):
    model_id = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=200
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(llm=local_llm, prompt=PROMPT)
    if option == "Ramayan Book":
        index = "pdf_book1"
        #query = "What are the qualtities of Hanuman?"
 
    else:
        index = "pdf_book2"
        #query = "how did abhimanyu die?"
    docsearch = FAISS.load_local(index,embeddings)
    docs = docsearch.similarity_search(question)
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
    return result
#ASR - English 
pipe = pipeline("automatic-speech-recognition",model="openai/whisper-tiny")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

def API_TTS (input_text):
    params = {
    "input": [
        {
        "source": input_text
    }
    ],
    "config": {
        "gender": "female",
        "language": {
        "sourceLanguage": "en"
        }
    }
    }

    # Send the API request
    response = requests.post(url, json=params)
    # Decode the Base64-encoded audio content
    resp = json.loads(response.content)
    wav_file = open('tts1.wav', "wb")
    audio_content = base64.b64decode(resp["audio"][0]["audioContent"])
    # Write the raw binary data to a WAV file
    wav_file.write(audio_content)
  


    return 'tts1.wav'


def my_function(input):
    return "You selected: " + input

# main voice chat app 
def app(audio, option):
    #ASR Output
    question = transcribe(audio)
    #NLU output

    if option =="Ramayan Book":
        answer = load_local_LangModel(question,option)
    elif option =="Mahabharat Book" :
        answer = load_local_LangModel(question,option)
    elif  option =="Wikipedia" :
        answer = wiki_run(question)
    else:
        answer = duck_run(question)
    
    params = {
    "input": [
        {
        "source": answer
    }
    ],
    "config": {
        "gender": "female",
        "language": {
        "sourceLanguage": "en"
        }
    }
    }

    # Send the API request
    response = requests.post(url, json=params)
    # Decode the Base64-encoded audio content
    resp = json.loads(response.content)
    wav_file = open('tts1.wav', "wb")
    audio_content = base64.b64decode(resp["audio"][0]["audioContent"])
    # Write the raw binary data to a WAV file
    wav_file.write(audio_content)

    return [question,answer,'tts1.wav']

#Gradio Interface

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="LLM Output")
output_3 = gr.Audio('tts1.wav',  label="Text to speech audio")
dropdown = gr.inputs.Dropdown(choices=["Ramayan Book", "Mahabharat Book", "Wikipedia", "DuckDuckGo"], label="Select an option:")

gr.Interface(
    title = 'Voice Assistant For Indian Knowledge Systems', 
    fn=app, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath"),
        dropdown
    ],

    outputs=[
        output_1,  output_2, output_3
    ],
    live=True).launch(share=True)











