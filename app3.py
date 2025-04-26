import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "ChatBot with LLM"
os.environ['GOOGLE_API_KEY'] = "AIzaSyD3jjlk9rl6FBUASv21T1aBAFo_h_R6rTk"

# Defining prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant, kindly responnse to the user query."),
        ("user", "Question: {question}")
    ]
)

def generate_response_ollama(question, llm, temperature):
    try:        
        llm = Ollama(model = llm, temperature = temperature)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser 
        answer = chain.invoke({"question" : question})
        return answer 
    except Exception as e:
        return f"Error: {str(e)}"
    
def generate_response_gemini(question, llm, temperature, max_tokens):
    try:        
        llm = ChatGoogleGenerativeAI(model = llm, temperature = temperature, max_output_tokens = max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser 
        answer = chain.invoke({"question" : question})
        return answer 
    except Exception as e:
        return f"Error: {str(e)}"
    
# Streamlit Interface
st.title("Q&A ChatBot (Ollama + Gemini)")
st.sidebar.title("Settings")

# Select Provider
provider = st.sidebar.selectbox("Select Provider", ["Ollama", "Gemini"])

# Model and settings based on provider
if provider == "Ollama":
    model = st.sidebar.selectbox("Select Ollama Model", ["tinyllama", "gemma:2b", "phi"])
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5)
elif provider == "Gemini":
    model = st.sidebar.selectbox("Select Gemini Model", ["gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"])
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.sidebar.slider("Max Tokens:", min_value=50, max_value=300, value=150)
    
# Main Interface
st.write("Ask anything...")
user_input = st.text_input("You:")

submit = st.button("Submit")  # 👈 This is important!

if submit and user_input:  
    with st.spinner('Generating response...'):
        if provider == "Ollama":
            response = generate_response_ollama(user_input, model, temperature)
        else:
            response = generate_response_gemini(user_input, model, temperature, max_tokens)
    st.success("Response Generated!")
    st.write(response)

elif submit and not user_input:
    st.warning("Please enter your question first.")
    