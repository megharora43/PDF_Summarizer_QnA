import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
load_dotenv()

# Function to read PDF and extract text
def read_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=3000, chunk_overlap=200):  # Increased chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create embeddings using HuggingFace
def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(chunks, embeddings)
    return docsearch

# Function to create a QA chain
def create_qa_chain(docsearch):
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",temperature=0.8, max_new_tokens=1024)
    # Increased max_length
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa_chain

# Streamlit interface
def app():
    st.title("PDF QnA Chatbot")
    st.write("Upload your PDF and ask questions based on its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Initialize chat history in session state if not already initialized
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file is not None:
        try:
            text = read_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return

        if not text:
            st.error("No text extracted from the PDF.")
            return

        # Chunk the text to avoid exceeding the token limit
        chunks = chunk_text(text)

        # Create embeddings and retrieval system for the chunks
        docsearch = create_embeddings(chunks)

        # Create a QA chain
        qa_chain = create_qa_chain(docsearch)

        st.write("Ask a question:")
        user_question = st.text_input("Your question:")

        if user_question:
            with st.spinner('Thinking...'):
                # Invoke the QA chain
                result = qa_chain({"query": user_question})

                # Extract the response and source documents
                response = result["result"]
                source_documents = result["source_documents"]

                # Add the current question and response to the chat history
                st.session_state.chat_history.append((user_question, response))

                # Limit the chat history to the last 5 interactions
                if len(st.session_state.chat_history) > 3:
                    st.session_state.chat_history = st.session_state.chat_history[-3:]

                st.write(f"Answer: {response}")

                # Display source documents (optional)
                # st.write("Source Documents:")
                # for doc in source_documents:
                #     st.write(doc.page_content)

        # Display chat history
        st.markdown("\n")
        st.markdown("**Chat History:**")
        for q, a in st.session_state.chat_history:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")

# Run the Streamlit app
if __name__ == "__main__":
    app()