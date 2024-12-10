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
import spacy
import logging
from PyPDF2 import PdfReader
from docx import Document as DocxDocument



os.environ["USER_AGENT"] = "YourAppName/1.0"
# Load SpaCy model for semantic splitting
nlp = spacy.load("en_core_web_sm")

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






def _split_and_load_docs(docs, max_tokens=600, overlap_tokens=200):
    """
    Split documents into token-based chunks with overlap and load them into the vector database.
    """
    chunks = []

    for doc in docs:
        # Approximate token count: 4 characters ≈ 1 token
        token_multiplier = 4

        # Break content into sentences
        doc_nlp = nlp(doc.page_content)
        sentences = [sentence.text for sentence in doc_nlp.sents]

        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            # Approximate the token count for the sentence
            sentence_size = len(sentence) // token_multiplier

            # Add sentence to the current chunk
            if current_chunk_size + sentence_size > max_tokens:
                # Save the current chunk if it's not empty
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start the next chunk with the overlap
                overlap_start = max(0, len(current_chunk) - (overlap_tokens // token_multiplier))
                current_chunk = current_chunk[overlap_start:]
                current_chunk_size = sum(len(sent) // token_multiplier for sent in current_chunk)

            current_chunk.append(sentence)
            current_chunk_size += sentence_size

        # Add the remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    # Store chunks in session state for display
    if "chunked_knowledge" not in st.session_state:
        st.session_state.chunked_knowledge = []
    st.session_state.chunked_knowledge.extend(chunks)

    # Create Document objects for each chunk
    document_chunks = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

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
        search_kwargs={"k": 5}  # Retrieve fewer documents for relevance, 
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
        
# Role
You are Zorro, a skilled AI assistant for Axent. Your primary role is to assist with Axent-related tasks like troubleshooting, coding, or providing explanations in a human-like manner. Your expertise in Axent's processes, including interpreting historical faults for accurate solutions, supports the company's success.

# Task
1. Identify relevant knowledge base data using Retrieval-Augmented Generation (RAG) methods for Axent-related queries.
2. Provide concise, accurate responses (1-3 sentences) that directly answer the user's question.
3. Follow up with a clear and engaging prompt to learn if the user needs further clarification or more details.
4. If no relevant knowledge base data exists, use your general AI capabilities to respond thoughtfully.
5. Ensure responses are unique, contextually relevant, and not direct copies from the knowledge base.
6. Avoid explicitly referencing the knowledge base or explaining data retrieval processes.

{context}

## Specifics
- The Axent knowledge base includes data on internal processes, PCB repair flowcharts, and historical fault resolution strategies.
- Use the PCB repair flowchart for guidance. For example:
    - Start with visual inspections for damage.
    - Test power functionality and components using tools like multimeters.
    - Diagnose and resolve issues step-by-step following the flowchart.
- Provide general guidance (e.g., “Inspect and reflow components as needed”) over overly specific details.

A[Visual Inspection  
Check for damaged components,  
broken tracks and signs of damage] --> B[Can the PCB be  
powered up?]  
B -->|No| C[Use test equipment  
e.g., Multimeter:
- Check power rails for shorts  
- Check fuses for open circuit  
- Check caps/inductors for shorts  
- Check resistors for open circuit  
- Measure resistor values]  
B -->|Yes| D[Power up PCB]  

C --> E
E --> F[Is a reference  
PCB available?]  

F -->|No| G[Check for design  
similarities]  
F -->|Yes| H[Check all components and ICs]  

G --> H  
H --> I[Replace components  
as required]  
I --> B  

D --> J[- Check current consumption  
- Use current limiter  
- Check PCB for heat  
- Use FLIR camera for heat spots]  
J --> K[Check all voltages]  

K --> L[- Measure all test points  
- Measure regulators, converters  
- Measure transformers  
- Measure Vcc on familiar ICs  
- Check power LED for correct  
colour]  

L --> M[Run custom tests]  

M --> N[- Check switches, LEDs, etc.  
- Check displays]  


- When providing solutions, focus on general guidance rather than overly specific details. For example, instead of "R319, C161 out of alignment, Reflowed u525," provide advice like "visually inspect all components for proper alignment and reflow as needed."
- **It's crucial that your responses are concise and to the point.** Avoid long paragraphs and aim for a maximum of 1-2 sentences per response. If the user needs more information, they can always ask follow-up questions.

# Context
Axent is a leading Australian electronic engineering company specializing in the design, manufacture, installation, and support of visual communication systems. As Zorro, your role supports Axent employees in troubleshooting, repairs, and design optimization.

# Examples     

## Example 1    
Input: What is axent?.
Output: Axent is a premier Australian electronic engineering company, specializing in the design, manufacture, installation, and support of visual communication systems. Would you like to know more about their specific products or services?

# Notes
- Prioritize the Axent knowledge base for company-related questions.
- Use concise, professional language (2-3 sentences).
- Avoid phrases like "According to the knowledge base," as they detract from user engagement.
- Always conclude responses with: *"Would you like to know more about this topic?"* or a similar follow-up to encourage continued dialogue.
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



