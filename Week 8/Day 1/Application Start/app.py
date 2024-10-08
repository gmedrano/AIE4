### Import Section ###
"""
IMPORTS HERE
"""
import os
import uuid
import openai  # Add this import
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
import chainlit as cl
import tempfile

### Global Section ###
"""
GLOBAL CODE HERE
"""
# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up LangSmith
os.environ["LANGCHAIN_PROJECT"] = f"AIM Week 8 Assignment 1 - {uuid.uuid4().hex[0:8]}"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Set up embeddings with cache
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings, store, namespace=core_embeddings.model
)

# Set up QDrant vector store
collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Set up chat model and cache
chat_model = ChatOpenAI(model="gpt-4o-mini")
set_llm_cache(InMemoryCache())

# Set up RAG prompt
rag_system_prompt_template = """
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existence of context.
"""

rag_user_prompt_template = """
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    """ SESSION SPECIFIC CODE HERE """
    # Upload and process PDF
    files = await cl.AskFileMessage(content="Please upload a PDF file to begin.", accept=["application/pdf"]).send()
    if not files:
        await cl.Message(content="No file was uploaded. Please try again.").send()
        return

    file = files[0]
    
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.content)
            tmp_file_path = tmp_file.name

        # Load and split the PDF
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        # Set up vector store
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=cached_embedder
        )
        vectorstore.add_documents(docs)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        # Set up RAG chain
        global retrieval_augmented_qa_chain
        retrieval_augmented_qa_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | chat_prompt | chat_model
        )

        msg.content = f"`{file.name}` processed. You can now ask questions about it!"
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"An error occurred while processing the file: {str(e)}").send()
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """
    return "RAG Assistant"

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """
    MESSAGE CODE HERE
    """
    response = retrieval_augmented_qa_chain.invoke({"question": message.content})
    await cl.Message(content=response.content).send()