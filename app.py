import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI

# Importing necessary functions and classes from rag_methods
from rag_methods import (
    load_doc_to_db,  # To handle document uploads and processing
    stream_llm_rag_response,  # To handle RAG-based streaming LLM responses
)

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

dotenv.load_dotenv()

MODELS = [
    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-3-5-sonnet-20241022",
]

# Set page configuration
st.set_page_config(
    page_title="Axent AI Chatbot Testing Server",
    page_icon="https://media.licdn.com/dms/image/v2/D560BAQEBF-3HnrxuVA/company-logo_200_200/company-logo_200_200/0/1693549647080/axent_global_logo?e=2147483647&v=beta&t=dwGW7Fn_ieSVjui35tQ626-Isd9WMHVcWUpOzC0JjME",  # Custom logo URL
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header with custom logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D560BAQEBF-3HnrxuVA/company-logo_200_200/company-logo_200_200/0/1693549647080/axent_global_logo?e=2147483647&v=beta&t=dwGW7Fn_ieSVjui35tQ626-Isd9WMHVcWUpOzC0JjME" alt="Chatbot Icon" width="50">
        <h2>Axent Zorro AI Chatbot</h2>
    </div>
    """,
    unsafe_allow_html=True,
)



# Initial Setup

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assitant", "content": "Hi there by name's Zorro! How can I assist you?"}
]

# Initialise vector_db in session state to avoid AttributeError
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None  # Set vector_db to None by default

# Sidebar with logo
with st.sidebar:
    st.image("https://media.licdn.com/dms/image/v2/D560BAQEBF-3HnrxuVA/company-logo_200_200/company-logo_200_200/0/1693549647080/axent_global_logo?e=2147483647&v=beta&t=dwGW7Fn_ieSVjui35tQ626-Isd9WMHVcWUpOzC0JjME", width=100)  # Custom sidebar logo
    st.write("**Claude Model**")
    
    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""

    with st.popover("OpenAI"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)",
            value=default_openai_api_key,
            type="password",
            key = "openai_api_key",
        )

    default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""

    with st.popover("Anthropic"):
        anthropic_api_key = st.text_input(
            "Introduce your Anthropic API Key (https://console.anthropic.com/)",
            value=default_anthropic_api_key,
            type="password",
            key = "anthropic_api_key"
        )

# Main Content
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None

if missing_anthropic:
    st.write("#")
    st.warning("Please introduce an API Key to continue...")

else:
    # Sidebar with model selection
    with st.sidebar:
        st.divider()
        if "chunked_knowledge" in st.session_state and st.session_state.chunked_knowledge:
            with st.expander("Chunked Knowledge Data"):
                for i, chunk in enumerate(st.session_state.chunked_knowledge, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)
        else:
            st.write("No chunked knowledge data available.")

        st.selectbox(
            "Claude Model",
            [model for model in MODELS if ("anthropic" in model and not missing_anthropic)],
            key="model",
        )
        # Clear Chat Button
        cols0 = st.columns(2)

        with cols0[0]:
            st.session_state.use_rag = True  # Always enable RAG


        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")


        st.header("RAG Sources:")



        # Check if the vector database is loaded
        is_vector_db_loaded = "vector_db" in st.session_state and st.session_state.vector_db is not None

        # File upload input for RAG with documents
        uploaded_files = st.file_uploader(
            "📄 Upload documents (PDF, Word, Text, CSV, Excel):",
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,  # Uses the updated load_doc_to_db function
            key="rag_docs",
        )

        # Display formatted results for AI-processed files
        if "formatted_results" in st.session_state:
            with st.expander("Formatted Results (PDF and Word Only)"):
                st.write("Below is the formatted output from Claude AI:")
                for batch_id, content in st.session_state.formatted_results.items():
                    st.markdown(f"### {batch_id}")
                    st.text(content)

        # Display documents loaded into the vector DB
        with st.expander(f"📂 Documents in DB ({len(st.session_state.rag_sources)})"):
            if not st.session_state.get("vector_db"):
                st.write("No documents loaded into the vector database.")
            else:
                st.write([source for source in st.session_state.rag_sources])





    # Main chat app
    model_provider = st.session_state.model.split("/")[0]


    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
            max_tokens_to_sample=500,  # Set max tokens for OpenAI
        )

    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key = anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
            max_tokens_to_sample=500,  # Set max tokens for Anthropic
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if the input contains a code snippet
        if "```" in prompt or any(line.strip().startswith("from") or line.strip().startswith("import") for line in prompt.splitlines()):
            # Automatically assume the user is asking for code assistance
            response_prompt = f"Please comment on this code:\n{prompt}"
        else:
            # Use the input as-is for general conversation
            response_prompt = prompt

        # Generate the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Create messages for query
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            if st.session_state.vector_db:
                # Use RAG if the vector database is available
                for chunk in stream_llm_rag_response(llm_stream, messages):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

            # Save the response in session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})