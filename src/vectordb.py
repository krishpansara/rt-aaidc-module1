import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Implement text chunking logic
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap = 20
        )
        chunks = splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # document ingestion logic
        print(f"Processing {len(documents)} documents...")
        # Your implementation here
        all_metadata = []
        all_ids = []
        all_chunks = []

        for doc_index, doc in enumerate(documents):
            content = doc.page_content
            metadata = doc.metadata

            if not content.strip():
                continue

            chunks = self.chunk_text(content)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_index}_chunk_{chunk_idx}"

                all_chunks.append(chunk)
                all_metadata.append({
                    **metadata,
                    "chunk_id": chunk_id,
                    "doc_index": doc_index,
                    "chunk_index": chunk_idx
                })

                all_ids.append(chunk_id)
    
        print(f"Created {len(all_chunks)} chunks from documents")

        if not all_chunks:
            print("Warning: No chunks created from documents")
            return
        
        print("Generating embeddings for all chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        print("Storing vectors and embeddings into vector database...")
        self.collection.add(
            ids = all_ids,
            documents = all_chunks,
            metadatas = all_metadata,
            embeddings = embeddings.tolist()
        )

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        #  Implement similarity search logic
        query_embedding = self.embedding_model.encode([query])

        result = self.collection.query(
            query_embeddings= query_embedding.tolist(),
            n_results=n_results
        )

        if not result or not result.get("documents"):
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        return {
            "documents": result.get("documents", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "distances": result.get("distances", [[]])[0],
            "ids": result.get("ids", [[]])[0],
            }