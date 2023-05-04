
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
import streamlit as st
from audiorecorder import audiorecorder
import magic
import os
import re

# Set up the API endpoint URL
url = "https://tts-api.ai4bharat.org/"



# Loading Language models based on language 
def load_local_LangModel(language):
    if language == "hindi":
        model_id = 'doc2query/msmarco-hindi-mt5-base-v1'
    else:
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
    return chain

#ASR Models used 
pipe_en = pipeline("automatic-speech-recognition",model="openai/whisper-tiny")
pipe_hi = pipeline(model="sanchit-gandhi/whisper-small-hi") 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# For Hindi 
embeddings2 = HuggingFaceEmbeddings(model_name="google/muril-base-cased")

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
# Book based search 
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

def hindi_csv(question, chain):
    docsearch = FAISS.load_local("csv_index1",embeddings2)
    docs = docsearch.similarity_search(question)
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
    return result


def API_TTS(language, input_text):
    if lang == 'hindi':
        params = {
        "input": [
            {
            "source": input_text
        }
        ],
        "config": {
            "gender": "male",
            "language": {
            "sourceLanguage": "hi"
            }
        }
        }
    else:
        params = {
        "input": [
            {
            "source": input_text
        }
        ],
        "config": {
            "gender": "male",
            "language": {
            "sourceLanguage": "en"
            }
        }
        }

    # Send the API request
    response = requests.post(url, json=params)
    # Decode the Base64-encoded audio content
    resp = json.loads(response.content)

    audio_content = base64.b64decode(resp["audio"][0]["audioContent"])
    # Write the raw binary data to a WAV file
    with wave.open('tts_output.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)   # Set the number of channels (1 for mono, 2 for stereo)
        wav_file.setsampwidth(2)   # Set the sample width in bytes (2 for 16-bit audio)
        wav_file.setframerate(22050)  # Set the sample rate (22050 Hz is a common value)
        wav_file.writeframes(audio_content)
    return 'tts_output.wav'

# main voice chat app 
def app():
    # Put expensive initialize computation here
    st.title("Voice Assistant for Indian Knowledge Systems")
    st.subheader("For English & Hindi language!")

    st.sidebar.header("Voice Assistant Settings")
    language = st.sidebar.selectbox("Language", ("English", "Hindi"))
    if language == "Hindi":
        search_type_disabled = True
    else:
        search_type_disabled = False
    search_type = st.sidebar.selectbox("Search Type", ("Wikipedia", "DuckDuckgo", "Ramayana_book","Mahabharat_book"),key=search_type_disabled)
  

    # recorder 
    audio = audiorecorder("Push to Talk", "Recording... (push again to stop)")

    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.tobytes())
        # To save audio to a file:
        wav_file = open("record.wav", "wb")
        wav_file.write(audio.tobytes())

        with st.spinner("Recognizing your voice command ..."):
            if language == "English":
                question = transcribe("record.wav",pipe_en)
            else:
                question = transcribe('record.wav',pipe_hi)
                    
            st.markdown("<b>You:</b> " + question, unsafe_allow_html=True)
            print('ASR result is:' + question)


        st.write(question)









