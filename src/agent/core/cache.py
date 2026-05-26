"""
Persistent semantic cache engine using ChromaDB.
Checks for semantically similar historical queries to bypass expensive web search calls.
"""
import os
import chromadb
from src.core.config import (
    SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD
)
from src.core.logging import setup_logger

logger = setup_logger("nexus_ai.cache")

# Persistent directory inside the workspace
CACHE_DIR = os.path.join(os.getcwd(), ".chroma_cache")

# Global variables for lazy initialization
_client = None
_collection = None

def _get_collection():
    """Lazily initializes and returns the persistent Chroma collection."""
    global _client, _collection
    if not SEMANTIC_CACHE_ENABLED:
        return None
    
    if _collection is None:
        try:
            # Ensure path exists
            os.makedirs(CACHE_DIR, exist_ok=True)
            
            # Initialize persistent client
            _client = chromadb.PersistentClient(path=CACHE_DIR)
            # Create or get collection with cosine space
            _collection = _client.get_or_create_collection(
                name="nexus_ai_semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Semantic cache initialized successfully at .chroma_cache/")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma semantic cache: {str(e)}")
            return None
    return _collection

def get_semantic_cache(query: str) -> str | None:
    """
    Checks if a semantically similar query exists in the cache.
    Returns cached research data if a match is found above the similarity threshold.
    """
    collection = _get_collection()
    if not collection:
        return None
        
    try:
        query = query.strip()
        if not query:
            return None
            
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        
        # Results structure: { 'ids': [['id1']], 'distances': [[0.12]], 'documents': [['content']], 'metadatas': [[{...}]] }
        # Note: With 'cosine' distance, a distance of 0 means identical, and distance of 1 means orthogonal.
        # Cosine similarity = 1 - Cosine distance.
        if results and results.get("ids") and results["ids"][0]:
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            logger.info(f"Semantic cache search for '{query}': nearest similarity is {similarity:.4f} (distance: {distance:.4f})")
            
            if similarity >= SEMANTIC_CACHE_THRESHOLD:
                # Extract research data from metadata
                metadata = results["metadatas"][0][0]
                if metadata and "research_data" in metadata:
                    cached_data = metadata["research_data"]
                    logger.info(f"🚀 [SEMANTIC CACHE HIT] Found cached research (Similarity: {similarity:.4f})")
                    return cached_data
                
    except Exception as e:
        logger.error(f"Error querying semantic cache: {str(e)}")
        
    return None

def add_to_semantic_cache(query: str, research_data: str):
    """
    Saves a high-confidence research result into the semantic cache.
    Stores the query itself as the document (for query-to-query similarity matching)
    and the research findings in the metadata.
    """
    collection = _get_collection()
    if not collection:
        return
        
    try:
        query = query.strip()
        research_data = research_data.strip()
        if not query or not research_data:
            return
            
        # Check if already exists exactly to avoid duplication
        existing = collection.get(ids=[query])
        if existing and existing.get("ids"):
            # Update existing
            collection.update(
                ids=[query],
                documents=[query],
                metadatas=[{"research_data": research_data}]
            )
            logger.info(f"Updated semantic cache entry for: '{query}'")
        else:
            # Add new
            collection.add(
                ids=[query],
                documents=[query],
                metadatas=[{"research_data": research_data}]
            )
            logger.info(f"Successfully cached research for query: '{query}'")
            
    except Exception as e:
        logger.error(f"Error writing to semantic cache: {str(e)}")
