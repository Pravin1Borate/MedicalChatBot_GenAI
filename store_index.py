import os 
from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import  PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()



extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data=extracted_data)
embeddings = download_huggingface_embeddings()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "medicalchatbot"

pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)