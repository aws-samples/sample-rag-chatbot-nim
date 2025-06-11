import gradio as gr
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
import boto3
import os

LLM_URL=os.environ.get("LLM_URL", "http://localhost:8000/v1")
LLM_MODEL=os.environ.get("LLM_MODEL", "meta/llama-3.2-1b-instruct")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://localhost:8001/v1")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
AWS_DEFAULT_REGION=os.environ.get("AWS_DEFAULT_REGION")
OPENSEARCH_COLLECTION_ID=os.environ.get("OPENSEARCH_COLLECTION_ID")
OPENSEARCH_INDEX=os.environ.get("OPENSEARCH_INDEX")

host="https://" + str(OPENSEARCH_COLLECTION_ID) + "." + str(AWS_DEFAULT_REGION) + ".aoss.amazonaws.com"

embedder = NVIDIAEmbeddings(base_url=EMBEDDINGS_URL, model= EMBEDDINGS_MODEL)

text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=100,
)

def create_vectorstore():
    credentials = boto3.Session().get_credentials()
    awsauth = AWSV4SignerAuth(credentials, AWS_DEFAULT_REGION, "aoss")

    vectorstore = OpenSearchVectorSearch(
        host,
        OPENSEARCH_INDEX,
        embedder,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        vector_field="vector_field"
    )

    index_mapping = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 2048,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    }
                }
            }
        }
    }

    if vectorstore.index_exists(OPENSEARCH_INDEX):
        vectorstore.delete_index(OPENSEARCH_INDEX)
    vectorstore.client.indices.create(index=OPENSEARCH_INDEX, body=index_mapping)
    
    return vectorstore

def predict(message, _):
    return chain.invoke(message)

def upload_file(files):
    for file_path in files:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        split_documents = text_splitter.split_documents(documents)
        vectorstore.add_documents(split_documents)
    return files

vectorstore=create_vectorstore()

llm = ChatNVIDIA(
    base_url=LLM_URL,
    model=LLM_MODEL,
)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer based on the following context:\n<Documents>\n{context}\n</Documents>",
        ),
        ("user", "{question}"),
    ]
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=predict, 
            type="messages",
        )
    with gr.Tab("Document upload"):
        with gr.Row():
            file_output = gr.File()
        with gr.Row():
            upload_button = gr.UploadButton("Click to upload documents", file_types=[".pdf"], file_count="multiple")
            upload_button.upload(
                fn=upload_file,
                inputs=upload_button,
                outputs=file_output
            )
    
demo.launch()
