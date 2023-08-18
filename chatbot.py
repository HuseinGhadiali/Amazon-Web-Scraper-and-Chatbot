import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
def generate_response(openai_api_key, query_text):
    # Load conversation context
    context = memory.load_memory_variables({})
    # Load document from file
    with open('your_filename.csv', 'r', encoding='UTF-8') as f:
        documents = [f.read()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    response = qa.run(query=query_text, context=context)
    # Save conversation context
    memory.save_context({"input": query_text}, {"output": response})
    return response
# Page title
st.set_page_config(page_title='🤖🔗 Ask ScrapBot')
st.title('🤖🔗 Ask ScrapBot')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please show top 5 products based on the reviews.')
# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not query_text)
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(openai_api_key, query_text)
            result.append(response)
            del openai_api_key
if len(result):
    st.info(response)
