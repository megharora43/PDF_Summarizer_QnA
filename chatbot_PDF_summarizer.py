import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever # New import
from langchain.chains.retrieval import create_retrieval_chain # New import
from langchain.chains.combine_documents import create_stuff_documents_chain # New import for combining docs
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # New imports for prompts
from langchain.memory import ConversationBufferMemory # Still used for managing chat history
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file
# Ensure you have your GOOGLE_API_KEY="your_google_api_key" in your .env file.
load_dotenv()

# Retrieve Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Validation of necessary environment variables ---
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. "
             "Please set it in a .env file or as an environment variable.")
    st.stop() # Stop the app if API key is missing

# Function to read PDF and extract text
def read_pdf(pdf_file):
    """
    Reads a PDF file using PyPDF2 and extracts all text content.

    Args:
        pdf_file: A file-like object representing the PDF.

    Returns:
        A string containing all extracted text from the PDF.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=3000, chunk_overlap=200):
    """
    Splits a given text into smaller, overlapping chunks.
    This is important for processing large documents with LLMs that have token limits.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters that overlap between consecutive chunks.

    Returns:
        list: A list of text chunks (strings).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Specifies the length calculation method
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create embeddings using Google Generative AI
@st.cache_resource # Cache the embeddings to avoid re-creating them on every rerun
def create_embeddings(chunks):
    """
    Generates embeddings for a list of text chunks using Google Generative AI's embedding model
    and stores them in a FAISS vector store.

    Args:
        chunks (list): A list of text chunks (strings).

    Returns:
        FAISS: A FAISS vector store containing the embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = FAISS.from_texts(chunks, embeddings)
    return docsearch

# Function to create a QA chain using LCEL components
def create_qa_chain(docsearch_obj, google_api_key):
    """
    Creates a RAG chain using create_history_aware_retriever and create_retrieval_chain.

    Args:
        docsearch_obj (FAISS): The FAISS vector store to be used as a retriever.
        google_api_key (str): Your Google Generative AI API key.

    Returns:
        Runnable: A LangChain Runnable (chain) for Q&A.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.8,
        max_output_tokens=1024,
        google_api_key=google_api_key
    )

    retriever = docsearch_obj.as_retriever(search_kwargs={"k": 4})

    ### 1. Contextualize question for history-aware retrieval
    # This prompt helps the LLM rephrase the user's question into a standalone query,
    # considering the ongoing chat history.
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation and the current question. Do not include any intro or outro, just the query itself.")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### 2. Answer generation chain
    # This prompt provides context and chat history to the LLM to generate the final answer.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder("chat_history"), # This will contain the actual chat history
        ("user", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    ### 3. Combine to form the full retrieval chain
    # This chain first uses the history-aware retriever, then passes the retrieved
    # documents and the original input/history to the answer generation chain.
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return retrieval_chain

# Streamlit interface
def app():
    """
    The main Streamlit application function for the PDF QnA Chatbot.
    """
    st.set_page_config(page_title="PDF QnA Chatbot", layout="centered")

    st.title("📄 PDF QnA Chatbot")
    st.write("Upload your PDF and ask questions based on its content using Google Generative AI (Free Tier).")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Initialize session state variables
    if 'chat_history_display' not in st.session_state:
        st.session_state.chat_history_display = []
    if 'doc_processed' not in st.session_state:
        st.session_state.doc_processed = False
    if 'docsearch_obj' not in st.session_state:
        st.session_state.docsearch_obj = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'memory_obj' not in st.session_state: # New: Store ConversationBufferMemory instance
        st.session_state.memory_obj = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    if uploaded_file is not None and not st.session_state.doc_processed:
        with st.spinner('Processing PDF and setting up chatbot... This might take a moment.'):
            try:
                # Clear previous state when a new PDF is uploaded
                st.session_state.chat_history_display = []
                st.session_state.doc_processed = False
                st.session_state.docsearch_obj = None
                st.session_state.qa_chain = None
                # Re-initialize memory for a fresh start with a new PDF
                st.session_state.memory_obj = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )

                text = read_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}. Please ensure it's a valid PDF.")
                return

            if not text:
                st.error("No text extracted from the PDF. The PDF might be empty or scanned.")
                return

            chunks = chunk_text(text)

            # Store docsearch object in session state
            st.session_state.docsearch_obj = create_embeddings(chunks)

            # Pass the docsearch object and google_api_key to create_qa_chain
            st.session_state.qa_chain = create_qa_chain(st.session_state.docsearch_obj, google_api_key)
            st.session_state.doc_processed = True
            st.success("PDF processed successfully! You can now ask questions.")
            st.rerun()

    # Only show question input if a document has been processed and chain is ready
    if st.session_state.doc_processed and st.session_state.qa_chain and st.session_state.docsearch_obj:
        st.markdown("---")
        st.write("Ask a question about the PDF content:")
        user_question = st.text_input("Your question:", key="user_question_input")

        if user_question:
            with st.spinner('Thinking...'):
                try:
                    # Invoke the new LCEL chain
                    # Pass the current input and the chat history from the memory object
                    result = st.session_state.qa_chain.invoke({
                        "input": user_question,
                        "chat_history": st.session_state.memory_obj.load_memory_variables(inputs={})["chat_history"] # Fix: Added inputs={}
                    })

                    response = result["answer"] # The output is now in 'answer' key
                    # 'source_documents' are nested under 'context' in the new structure's output
                    # The retrieval_chain returns {'input': ..., 'chat_history': ..., 'context': [...], 'answer': ...}
                    source_documents = result["context"]

                    # Explicitly save context to memory AFTER getting the response
                    st.session_state.memory_obj.save_context(
                        {"input": user_question},
                        {"output": response}
                    )

                    st.session_state.chat_history_display.append({"role": "user", "content": user_question})
                    st.session_state.chat_history_display.append({"role": "assistant", "content": response})

                    if len(st.session_state.chat_history_display) > 10:
                        st.session_state.chat_history_display = st.session_state.chat_history_display[-10:]

                    if source_documents:
                        with st.expander("Show Source Documents"):
                            for i, doc in enumerate(source_documents):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content)
                                if doc.metadata:
                                    st.json(doc.metadata)

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    import traceback
                    st.exception(traceback.format_exc()) # Show detailed traceback for debugging

        st.markdown("---")
        st.markdown("### Chat History:")
        for message in st.session_state.chat_history_display:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Bot:** {message['content']}")

# Run the Streamlit app
if __name__ == "__main__":
    app()
