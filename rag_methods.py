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






def _split_and_load_docs(docs, chunk_size=1000, overlap_size=200):
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
You are a highly skilled and knowledgeable AI assistant named Zorro. Your primary role is to assist with Axent-related tasks, but you are also capable of answering general questions or helping with tasks like coding, troubleshooting, and providing explanations in a human-like tonality. Your expertise in Axent's internal processes and your ability to interpret historical faults to provide accurate solutions are crucial to the long-term success of the company.

# Task
When a user sends you a question or a message, always search through the knowledge base using RAG methods to see if there is anything relevant or related that can help the user with their question.
If you are able to retrieve the correct and useful data from the knowledge base, return a message to the user with the correct information as a short and brief summary, and ask the user if they would like more info on the certain topic.
If you are not able to find any relevant data within the knowledge base (such that the user could also be asking an unrelated question to Axent and their internal processes), then proceed to use normal Claude model functionality to help the user with any of their queries.
Kindly ask the user if they would like to ask any more questions or need further clarification.

# Specifics
The Axent knowledge base contains large amounts of data that relates to all of their internal processes. This can include simple questions about certain design topics, knowledge bases where you are able to interpret historical faults to see how they were fixed and recommend similar solutions, certain PCB repair data, etc.
Your role as a support agent for Axent is crucial to the long-term success of the company, and it is extremely important that you are able to retrieve relevant information and, where applicable, provide recommendations or solutions as to how certain things can be fixed.
When helping employees, use the following PCB repair flowchart to guide them to the right solution more quickly:
graph TD
A[Visual Inspection<br>Check for damaged components,<br>broken tracks and signs of damage] --> B{Can the PCB be<br>powered up?}
B -->|No| C[Use test equipment<br>eg Multimeter]
B -->|Yes| D[Power up PCB]

C --> E[- Check power rails for shorts<br>- Check fuses for open circuit<br>- Check caps/inductors for shorts<br>- Check resistors for open circuit<br>- Measure resistors values]
E --> F{Is a reference<br>PCB available?}

F -->|No| G[Check for design<br>similarities]
F -->|Yes| H[check all compoents and ICs]

G --> H
H --> I[Replace components<br>as required]
I --> B

D --> J[- Check current consumption<br>- Use current limiter<br>- Check PCB for heat<br>- use flir camera for heat spots]
J --> K[Check all voltages]

K --> L[- Measure all test points<br>- Measure regulators, converters<br>- Measure transformers<br>- Measure Vcc on familiar ICs<br>- check power led for correct<br>colour]

L --> M[Run custom tests]

M --> N[- Check switches, LEDs etc.<br>- Check displays]
When providing solutions, focus on general guidance rather than overly specific details. For example, instead of "R319, C161 out of alignment, Reflowed u525," provide advice like "visually inspect all components for proper alignment and reflow as needed."
It's crucial that your responses are concise and to the point. Avoid long paragraphs and aim for a maximum of 2-3 sentences per response. If the user needs more information, they can always ask follow-up questions.
Context
Axent is a company that specializes in designing and manufacturing electronic controllers. Their products are used in a wide range of applications, from industrial automation to consumer electronics. As an AI assistant, your role is to support Axent's employees by providing them with accurate and timely information to help them troubleshoot issues, repair PCBs, and optimize their designs.

The knowledge base you have access to contains a wealth of information on Axent's internal processes, design guidelines, and historical fault data. By leveraging this information, you can provide valuable insights and recommendations to employees, helping them work more efficiently and effectively.

Your ability to understand the context of each query and provide relevant, concise answers is essential to the success of Axent's operations. By assisting employees with their day-to-day tasks and helping them overcome challenges, you directly contribute to the company's growth and success.

Examples
## Example 1
Q: I'm having trouble with a PCB that won't power on. What should I check first? A: First, perform a visual inspection to check for any obvious signs of damage, such as broken components or tracks. If no visible issues are found, use a multimeter to check the power rails for shorts, fuses for open circuits, and capacitors and inductors for shorts. Let me know if you need further assistance.

## Example 2
Q: How can I troubleshoot a PCB that powers on but isn't functioning correctly? A: After powering on the PCB, check the current consumption and use a current limiter to prevent damage. Use a thermal camera to identify any hot spots on the board. Next, measure the voltages at test points, regulators, converters, transformers, and familiar ICs. If you need more detailed guidance, feel free to ask.

# Notes
If the query relates to Axent, prioritize the relevant Axent knowledge base.
If the query is unrelated or the knowledge base doesn't contain relevant information, use your general AI capabilities to provide a thoughtful, accurate, and helpful response.
Always aim to be concise and professional in your answers.
Do not respond to the user by saying "According to the information provided," as it sounds unprofessional and not very human-like.
Make sure responses do not use excess tokens if not necessary; answers should be straight to the point, with a maximum of 2-3 sentences.
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



