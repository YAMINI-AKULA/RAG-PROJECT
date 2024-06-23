# Import necessary libraries
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper




__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Initialize the Wikipedia API wrapper with configuration
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Import additional required libraries for document processing and embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# Set your OpenAI API key

#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load documents from a specified web page
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# Split documents into manageable chunks
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Create a vector store for the documents using OpenAI embeddings
vectordb = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()

# Optional code for embedding using Huggingface, commented out
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# huggingface_embeddings = HuggingFaceBgeEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-l6-v2",
#     model_kwargs={'device':'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# vectorstore = Chroma.from_documents(documents[:500], huggingface_embeddings)

# Create a retriever tool for LangSmith search
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever, "langsmith_search", 
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
)

# Optional code to test the retriever tool, commented out
# response = retriever_tool.invoke({"query":"what langsmith?"})
# response

# List of tools available for the agent
tools = [wiki, retriever_tool]

# Import the ChatOpenAI class and create an instance of it
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Pull the prompt from the LangChain hub
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt.messages

# Import necessary classes for agent creation and execution
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Create an OpenAI tools agent with the specified LLM and tools
agent = create_openai_tools_agent(llm, tools, prompt)

# Initialize the agent executor with the created agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Optional code to test the agent executor, commented out
# agent_executor.invoke({"input":"what is Langsmith"})
# agent_executor.invoke({"input":"expain what is machine learning for a 5 year old kid"})

# Streamlit framework for deploying the app
import streamlit as st

# Set the title of the Streamlit app
st.title('Implementing RAG application using Langchain, OPENAI API, Hugging Face')

# Create an input text box for user queries
input_text = st.text_input("Hi, I am your assistant trained on Langsmith and I also have Wikipedia search enabled, try me!")

# If user input is provided, execute the agent and display the response
if input_text:
    st.write(agent_executor.invoke({'input':input_text}))

# To run the Streamlit app, use the following command:
# streamlit run agents_RAW.ipynb
