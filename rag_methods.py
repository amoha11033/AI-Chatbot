import streamlit as st
import os
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from time import time
import pandas as pd
from langchain.schema import Document
import logging
from PyPDF2 import PdfReader
from docx import Document as DocxDocument



os.environ["USER_AGENT"] = "YourAppName/1.0"
# Load SpaCy model for semantic splitting

dotenv.load_dotenv()

DB_DOCS_LIMIT = 20  # Increase the document upload limit

def extract_text_by_page(file_path):
    """
    Extract text from a PDF file, page by page.
    """
    try:
        with open(file_path, 'rb') as pdf:
            reader = PdfReader(pdf)
            return [page.extract_text().strip() for page in reader.pages]
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return []

def extract_text_from_docx(file_path):
    """
    Extract text from a Word document, including headers, paragraphs, and tables.
    """
    try:
        doc = DocxDocument(file_path)
        content = []

        # Extract paragraphs
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                content.append(text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_data:
                    content.append(" | ".join(row_data))  # Represent rows as pipe-separated

        return "\n\n".join(content)

    except Exception as e:
        logging.error(f"Error processing Word document {file_path}: {e}")
        return ""

def clean_data(dataframe):
    """
    Clean the data in a Pandas DataFrame by:
    - Dropping rows/columns with excessive NaN values.
    - Filling NaN values with a placeholder.
    - Removing duplicate rows.
    - Resetting the index.
    """
    # Drop rows/columns with more than 50% NaN values
    dataframe = dataframe.dropna(thresh=dataframe.shape[1] * 0.5, axis=0)
    
    # Fill remaining NaN values with placeholders
    dataframe = dataframe.fillna("MISSING_VALUE")
    
    # Drop duplicate rows
    dataframe = dataframe.drop_duplicates()
    
    # Reset the index
    dataframe.reset_index(drop=True, inplace=True)
    
    return dataframe

import chardet

def load_doc_to_db():
    """
    Handle document uploads. Use semantic chunking to split documents and store them in the vector database.
    """
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []  # List to store processed documents
        unique_sources = set(st.session_state.rag_sources)  # Track uploaded files to avoid duplicates

        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in unique_sources and len(unique_sources) < DB_DOCS_LIMIT:
                os.makedirs("source_files", exist_ok=True)
                file_path = f"./source_files/{doc_file.name}"

                with open(file_path, "wb") as file:
                    file.write(doc_file.read())

                try:
                    if doc_file.type == "application/pdf":
                        raw_pages = extract_text_by_page(file_path)
                        raw_text = "\n\n".join(raw_pages)
                        docs.append(Document(page_content=raw_text))

                    elif doc_file.name.endswith(".docx"):
                        raw_text = extract_text_from_docx(file_path)
                        if raw_text.strip():
                            docs.append(Document(page_content=raw_text))
                        else:
                            st.warning(f"No content extracted from {doc_file.name}.")

                    elif doc_file.type in ["text/plain", "text/markdown"]:
                        with open(file_path, "r", encoding="utf-8") as file:
                            raw_text = file.read()
                        docs.append(Document(page_content=raw_text))

                    elif doc_file.name.endswith((".xls", ".xlsx")):
                        excel_data = pd.read_excel(file_path, engine="openpyxl")
                        cleaned_data = clean_data(excel_data)

                        # Chunk the data into smaller pieces
                        chunk_size = 5  # Define number of rows per chunk
                        for i in range(0, len(cleaned_data), chunk_size):
                            chunk = cleaned_data.iloc[i:i + chunk_size]
                            json_chunk = chunk.to_json(orient="records")
                            docs.append(Document(page_content=json_chunk))

                    elif doc_file.name.endswith(".csv"):
                        try:
                            csv_data = pd.read_csv(file_path, encoding="utf-8")
                        except UnicodeDecodeError:
                            with open(file_path, 'rb') as f:
                                result = chardet.detect(f.read(10000))
                            csv_data = pd.read_csv(file_path, encoding=result['encoding'])

                        cleaned_data = clean_data(csv_data)

                        # Chunk the data into smaller pieces
                        chunk_size = 5  # Define number of rows per chunk
                        for i in range(0, len(cleaned_data), chunk_size):
                            chunk = cleaned_data.iloc[i:i + chunk_size]
                            json_chunk = chunk.to_json(orient="records")
                            docs.append(Document(page_content=json_chunk))

                    else:
                        st.warning(f"Document type {doc_file.type} not supported.")
                        continue

                    unique_sources.add(doc_file.name)
                    st.session_state.rag_sources = list(unique_sources)

                except Exception as e:
                    st.error(f"Error processing {doc_file.name}: {e}")

        if docs:
            _split_and_load_docs(docs)
            st.toast("Documents loaded successfully.", icon="✅")
        else:
            st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")






def _split_and_load_docs(docs, chunk_size=400, overlap_size=200):
    """
    Split documents into sequential chunks based on fixed size with optional overlap and load into the vector database.
    """
    chunks = []

    for doc in docs:
        # Tokenize content into fixed-size chunks
        token_multiplier = 4  # Approximation: 4 characters ≈ 1 token
        max_char_count = chunk_size * token_multiplier
        overlap_char_count = overlap_size * token_multiplier

        text = doc.page_content
        start = 0

        while start < len(text):
            # End index for current chunk
            end = start + max_char_count
            chunks.append(text[start:end])

            # Move the start point with overlap
            start += max_char_count - overlap_char_count

    # Filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Store chunks in session state for display
    if "chunked_knowledge" not in st.session_state:
        st.session_state.chunked_knowledge = []
    st.session_state.chunked_knowledge.extend(chunks)

    # Create Document objects for each chunk
    document_chunks = [Document(page_content=chunk) for chunk in chunks]

    # Check if vector_db is initialized
    if "vector_db" not in st.session_state or st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)
        st.session_state.vector_db.persist()  # Persist changes to database





def initialize_vector_db(docs):
    """
    Initialize a vector database and store documents with embeddings and metadata.
    """
    # Add metadata to each document
    for i, doc in enumerate(docs):
        doc.metadata = {"index": i}

    # Ensure database folder exists
    os.makedirs("./chroma_db", exist_ok=True)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key,
                                  model="text-embedding-3-small"
                                )
    document_texts = [doc.page_content for doc in docs]
    try:
        embeddings_data = embeddings.embed_documents(document_texts)
        if not embeddings_data:
            raise ValueError("Generated embeddings are empty.")
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

    # Initialize Chroma vector database
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",  # Persist directory for Chroma
        collection_name=f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}",
    )
    vector_db.persist()  # Ensure changes are saved

    return vector_db





def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve fewer documents for relevance, 
                                # Relevance: Increase k if you want to broaden the scope of retrieved documents.
                                # Efficiency: Decrease k if performance or relevance sufficiency is a concern.
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("system", "Use the retrieved knowledge to craft a relevant response."),
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain



def get_conversational_rag_chain(llm):
    if "vector_db" not in st.session_state or not st.session_state.vector_db:
        raise ValueError("No vector database found. Please upload a knowledge base or use the default LLM.")

    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         
#Role
You are an expert AI assistant specializing in providing accurate and concise information about Axent's products, services, and internal resources to employees. Your deep knowledge of the company allows you to efficiently answer queries and offer relevant guidance.

Task
Respond to employee questions using the following step-by-step process:

Analyze the user's query to identify key information they are seeking.
Search your knowledge base for relevant data related to the query.
If the query is within your knowledge domain, provide a concise and accurate response.
If the query is outside your knowledge domain, politely inform the user that you are unable to assist with their specific request.
Ask the user if they need any additional information or clarification on the topic.
{user_query}

#Specifics
Your responses should be brief and to the point, focusing on the most essential information to address the user's query.
If you are unable to find relevant information in your knowledge base, do not attempt to provide an answer. Instead, inform the user that you cannot assist with their specific request.
Your role in supporting Axent employees is crucial to their productivity and the smooth operation of the company, so please provide accurate and helpful responses whenever possible.

#Context
Axent is an engineering company that designs, manufactures, and maintains proprietary electronic signage systems. As an AI assistant, your purpose is to help employees quickly access information about Axent's products, services, and internal resources. This includes details about sign dimensions, manufacturing capabilities, fault troubleshooting, and employee-related queries such as leave balances and payroll.

Your knowledge base consists of Axent-specific data sourced from internal databases and documentation. By providing accurate and timely responses to employee queries, you contribute to the efficiency and effectiveness of the organization.

Examples
##Example 1
Question: What manufacturing capabilities does Axent have? Answer: Axent provides design, prototyping, testing, and manufacturing of proprietary electrical systems and metal sheet components using state-of-the-art SMT lines, industrial routers, printers, brake presses, and laser cutters.

##Example 2
Question: What are the dimensions of SAMS lighting matrix? Answer: The LED lighting matrix for WAMS is 512mm (W) x 512mm (H).

##Example 3
Question: What are the dimensions of Type-A ESZS Sign? Answer: 600mm (W) x 1750mm (H) on one 65NB pole.

##Example 4
Question: How do fire signs work? Answer: AFDRS display location specific fire risks using real-time data from the Bureau of Meteorology.

##Example 5
Question: What is the amber or yellow on an AFDRS sign? Answer: The amber level represents a high fire risk, which advises people to be ready to act.

##Example 6
Question: Tell me about the future of AI? Answer: My apologies, but I'm unable to tell you the future of AI. Is there anything else you would like to know?

##Example 7
Question: Can I get a raise? Answer: My apologies, but I cannot help you get a raise. Is there anything else you would like to know?

##Example 8
Question: Are you going to take our jobs? Answer: Fortunately, I will not take your job. I'm simply here to assist you with any questions you may have regarding Axent. Is there anything else you would like to know?

#Notes
Focus on providing information directly related to Axent's products, services, and internal resources.
If a query falls outside your knowledge domain, politely inform the user that you are unable to assist them and ask if there is anything else you can help with.
Maintain a professional and helpful tone throughout your interactions with employees.
Remember, your role is to support Axent employees by providing accurate and relevant information to enhance their productivity and decision-making.
    """),  # Custom prompt, can modify for better compactness and token efficiency if needed.
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    
    # Try retrieving relevant results
    try:
        # Use the RAG retriever chain
        for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
            response_message += chunk
            yield chunk
    except ValueError:
        # Fallback to default Claude response
        for chunk in llm_stream.stream(messages):
            response_message += chunk.content
            yield chunk

    # Return the response message
    return response_message



