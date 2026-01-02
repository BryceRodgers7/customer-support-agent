# rag_search.py
"""
RAG-based knowledge base search using Qdrant vector database.
This replaces the simple dictionary-based search in sample_data.py
"""
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import logging

logger = logging.getLogger("support_bot")

# Initialize client and embedder
# These will be initialized once when the module is imported
_client: Optional[QdrantClient] = None
_embedder: Optional[TextEmbedding] = None
COLLECTION = "support_kb"
QDRANT_URL = "http://localhost:6333"


def _initialize_rag():
    """Lazy initialization of Qdrant client and embedder."""
    global _client, _embedder
    
    if _client is None:
        try:
            _client = QdrantClient(url=QDRANT_URL)
            logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(
                f"Cannot connect to Qdrant at {QDRANT_URL}. "
                "Make sure Qdrant is running (e.g., via docker-compose up -d)."
            ) from e
    
    if _embedder is None:
        try:
            _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            logger.info("Initialized text embedder")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise RuntimeError("Failed to initialize text embedder") from e


def search_kb_rag(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search the knowledge base using RAG (Retrieval Augmented Generation).
    
    This function:
    1. Embeds the query using the same model used to create the vector database
    2. Performs semantic search in Qdrant
    3. Returns the most relevant chunks from the knowledge base
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        Dictionary containing:
        - results: List of matching documents with id, title, source, text snippet, and score
        - query: The original query
        - count: Number of results returned
    """
    try:
        # Initialize on first use
        _initialize_rag()
        
        if not query or not query.strip():
            return {"error": "Query cannot be empty", "results": [], "count": 0}
        
        # Embed the query
        query_vector = next(_embedder.embed([query.strip()]))
        query_vector_list = [float(x) for x in query_vector]
        
        # Search for similar points in Qdrant
        search_results = _client.query_points(
            collection_name=COLLECTION,
            query=query_vector_list,
            limit=max(1, min(max_results, 10))  # Clamp between 1 and 10
        ).points
        
        # Format results
        results = []
        for result in search_results:
            payload = result.payload or {}
            text = payload.get("text", "")
            
            # Create a snippet (first 300 chars)
            snippet = text[:300] + "..." if len(text) > 300 else text
            
            results.append({
                "id": result.id,
                "title": payload.get("title", "Untitled"),
                "source": payload.get("source", "Unknown"),
                "snippet": snippet,
                "relevance_score": round(result.score, 4),
                "full_text": text  # Include full text for LLM to use
            })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            "error": f"Knowledge base search failed: {str(e)}",
            "query": query,
            "results": [],
            "count": 0
        }


def get_kb_document(doc_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific knowledge base document by ID.
    
    Args:
        doc_id: The document ID
    
    Returns:
        Dictionary containing the full document details
    """
    try:
        _initialize_rag()
        
        retrieved_points = _client.retrieve(
            collection_name=COLLECTION,
            ids=[doc_id],
            with_vectors=False
        )
        
        if not retrieved_points:
            return {"error": f"Document with ID {doc_id} not found"}
        
        point = retrieved_points[0]
        payload = point.payload or {}
        
        return {
            "id": point.id,
            "title": payload.get("title", "Untitled"),
            "source": payload.get("source", "Unknown"),
            "text": payload.get("text", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve document {doc_id}: {e}")
        return {"error": f"Failed to retrieve document: {str(e)}"}


def check_rag_connection() -> Dict[str, Any]:
    """
    Check if RAG system is properly connected and operational.
    
    Returns:
        Dictionary with connection status and collection info
    """
    try:
        _initialize_rag()
        
        collection_info = _client.get_collection(collection_name=COLLECTION)
        
        return {
            "connected": True,
            "collection": COLLECTION,
            "total_documents": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "status": "operational"
        }
        
    except Exception as e:
        logger.error(f"RAG connection check failed: {e}")
        return {
            "connected": False,
            "error": str(e),
            "status": "error"
        }

