from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore , Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()
import os

class QdrantInsertRetrievalAll:
    def __init__(self,api_key,url):
        self.url = url 
        self.api_key = api_key

    # Method to insert documents into Qdrant vector store
    def insertion(self,text,embeddings,collection_name):
        qdrant = QdrantVectorStore.from_documents(
        text,
        embeddings,
        url=self.url,
        prefer_grpc=True,
        api_key=self.api_key,
        collection_name=collection_name,
        )
        print("insertion successfull")
        return qdrant

    # Method to retrieve documents from Qdrant vector store
    def retrieval(self,collection_name,embeddings):
        qdrant_client = QdrantClient(
        url=self.url,
        api_key=self.api_key,
        )
        qdrant_store = Qdrant(qdrant_client,collection_name=collection_name ,embeddings=embeddings)
        return qdrant_store
    
    # Method to delete a collection from Qdrant
    def delete_collection(self,collection_name):
        qdrant_client = QdrantClient(
        url=self.url,
        api_key=self.api_key,
        )
        qdrant_client.delete_collection(collection_name)
        return collection_name
    
    # Method to create a new collection in Qdrant with cosine similarity
    def create_collection(self,collection_name):
        qdrant_client = QdrantClient(
        url=self.url,
        api_key=self.api_key,
        )
        qdrant_client.create_collection(collection_name,vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE))
        print(f"Your collection {collection_name} created successfully")
        return collection_name

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("URL")
my_qdrant = QdrantInsertRetrievalAll(api_key=qdrant_api_key,url=qdrant_url)

