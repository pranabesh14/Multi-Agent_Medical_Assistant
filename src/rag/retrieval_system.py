# src/rag/retrieval_system.py
import numpy as np
import faiss
import pickle
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
import uuid
import os
import json
from datetime import datetime

class MedicalRAGSystem:
    """RAG system for medical document retrieval and question answering"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = embedding_model
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_text_splitter()
        self._initialize_storage()
        
        # Document storage
        self.documents = {}
        self.document_metadata = {}
        
        # Vector store
        self.vector_store = None
        self.bm25_retriever = None
        
        # Medical knowledge base (can be pre-loaded)
        self.medical_kb_path = "data/medical_kb"
        self._load_medical_knowledge_base()
    
    def _initialize_embeddings(self):
        """Initialize embedding models"""
        try:
            # Use BioBERT for medical document embeddings
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Also initialize HuggingFace embeddings for LangChain compatibility
            self.hf_embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            self.embeddings_ready = True
            self.logger.info("Embedding models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings_ready = False
    
    def _initialize_text_splitter(self):
        """Initialize text splitter for chunking documents"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _initialize_storage(self):
        """Initialize storage directories"""
        os.makedirs("data/documents", exist_ok=True)
        os.makedirs("data/embeddings", exist_ok=True)
        os.makedirs("data/indices", exist_ok=True)
    
    def _load_medical_knowledge_base(self):
        """Load pre-existing medical knowledge base"""
        try:
            if os.path.exists(f"{self.medical_kb_path}/documents.json"):
                with open(f"{self.medical_kb_path}/documents.json", 'r') as f:
                    kb_docs = json.load(f)
                
                for doc_id, doc_data in kb_docs.items():
                    self.documents[doc_id] = doc_data['content']
                    self.document_metadata[doc_id] = doc_data['metadata']
                
                self.logger.info(f"Loaded {len(kb_docs)} documents from knowledge base")
                
                # Rebuild vector store if embeddings exist
                if os.path.exists(f"{self.medical_kb_path}/vector_store"):
                    self._load_vector_store()
                    
        except Exception as e:
            self.logger.error(f"Error loading medical knowledge base: {e}")
    
    def check_status(self) -> bool:
        """Check if RAG system is ready"""
        return self.embeddings_ready
    
    def add_document(self, content: str, source: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the RAG system"""
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Store document
            self.documents[doc_id] = content
            
            # Store metadata
            doc_metadata = {
                'source': source,
                'added_date': datetime.now().isoformat(),
                'content_length': len(content),
                'doc_type': 'medical_document'
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            self.document_metadata[doc_id] = doc_metadata
            
            # Update vector store
            self._update_vector_store()
            
            self.logger.info(f"Added document {doc_id} from source: {source}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return None
    
    def _update_vector_store(self):
        """Update the vector store with new documents"""
        try:
            if not self.embeddings_ready:
                return
            
            # Create documents for LangChain
            langchain_docs = []
            
            for doc_id, content in self.documents.items():
                # Split document into chunks
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'doc_id': doc_id,
                            'chunk_id': i,
                            'source': self.document_metadata[doc_id]['source'],
                            'content_length': len(chunk)
                        }
                    )
                    langchain_docs.append(doc)
            
            # Create or update FAISS vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    langchain_docs,
                    self.hf_embeddings
                )
            else:
                # Add new documents to existing vector store
                new_docs = [doc for doc in langchain_docs 
                           if doc.metadata['doc_id'] not in [existing_doc.metadata['doc_id'] 
                                                           for existing_doc in self.vector_store.docstore._dict.values()]]
                if new_docs:
                    self.vector_store.add_documents(new_docs)
            
            # Update BM25 retriever
            self._update_bm25_retriever(langchain_docs)
            
            self.logger.info(f"Updated vector store with {len(langchain_docs)} document chunks")
            
        except Exception as e:
            self.logger.error(f"Error updating vector store: {e}")
    
    def _update_bm25_retriever(self, documents: List[Document]):
        """Update BM25 retriever for keyword-based search"""
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 10  # Number of documents to retrieve
            
        except Exception as e:
            self.logger.error(f"Error updating BM25 retriever: {e}")
    
    def query(self, query: str, max_results: int = 5, threshold: float = 0.7, 
              use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """Query the medical knowledge base"""
        try:
            if not self.vector_store:
                return []
            
            results = []
            
            if use_hybrid:
                # Hybrid search: combine vector similarity and BM25
                results = self._hybrid_search(query, max_results, threshold)
            else:
                # Vector similarity search only
                results = self._vector_search(query, max_results, threshold)
            
            # Post-process and rank results
            processed_results = self._post_process_results(results, query)
            
            return processed_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {e}")
            return []
    
    def _vector_search(self, query: str, max_results: int, threshold: float) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            # Search using FAISS vector store
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=max_results * 2  # Get more results for filtering
            )
            
            results = []
            for doc, score in docs_and_scores:
                # Convert distance to similarity (FAISS returns L2 distance)
                similarity = 1 / (1 + score)
                
                if similarity >= threshold:
                    results.append({
                        'content': doc.page_content,
                        'score': similarity,
                        'source': doc.metadata.get('source', 'unknown'),
                        'doc_id': doc.metadata.get('doc_id', 'unknown'),
                        'chunk_id': doc.metadata.get('chunk_id', 0),
                        'search_type': 'vector'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    def _hybrid_search(self, query: str, max_results: int, threshold: float) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and BM25"""
        try:
            vector_results = self._vector_search(query, max_results, threshold)
            
            # BM25 search
            bm25_results = []
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                
                for doc in bm25_docs[:max_results]:
                    bm25_results.append({
                        'content': doc.page_content,
                        'score': 0.8,  # Default BM25 score
                        'source': doc.metadata.get('source', 'unknown'),
                        'doc_id': doc.metadata.get('doc_id', 'unknown'),
                        'chunk_id': doc.metadata.get('chunk_id', 0),
                        'search_type': 'bm25'
                    })
            
            # Combine and deduplicate results
            combined_results = self._combine_search_results(vector_results, bm25_results)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return self._vector_search(query, max_results, threshold)
    
    def _combine_search_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate search results from different methods"""
        try:
            # Create a dictionary to track unique results
            unique_results = {}
            
            # Add vector results
            for result in vector_results:
                key = f"{result['doc_id']}_{result['chunk_id']}"
                unique_results[key] = result
                unique_results[key]['vector_score'] = result['score']
            
            # Add BM25 results
            for result in bm25_results:
                key = f"{result['doc_id']}_{result['chunk_id']}"
                if key in unique_results:
                    # Combine scores
                    unique_results[key]['bm25_score'] = result['score']
                    unique_results[key]['score'] = (
                        unique_results[key]['vector_score'] * 0.7 + 
                        result['score'] * 0.3
                    )
                    unique_results[key]['search_type'] = 'hybrid'
                else:
                    unique_results[key] = result
                    unique_results[key]['bm25_score'] = result['score']
            
            # Convert back to list and sort by score
            combined_results = list(unique_results.values())
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error combining search results: {e}")
            return vector_results + bm25_results
    
    def _post_process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Post-process search results for better relevance"""
        try:
            # Add query-specific scoring
            query_words = set(query.lower().split())
            
            for result in results:
                content_words = set(result['content'].lower().split())
                
                # Calculate word overlap score
                overlap = len(query_words.intersection(content_words))
                overlap_score = overlap / len(query_words) if query_words else 0
                
                # Adjust final score
                result['overlap_score'] = overlap_score
                result['final_score'] = (
                    result['score'] * 0.8 + 
                    overlap_score * 0.2
                )
                
                # Add relevance category
                if result['final_score'] > 0.8:
                    result['relevance'] = 'high'
                elif result['final_score'] > 0.6:
                    result['relevance'] = 'medium'
                else:
                    result['relevance'] = 'low'
            
            # Sort by final score
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error post-processing results: {e}")
            return results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the system"""
        return len(self.documents)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID"""
        if doc_id in self.documents:
            return {
                'content': self.documents[doc_id],
                'metadata': self.document_metadata[doc_id]
            }
        return None
    
    def clear_documents(self):
        """Clear all documents from the system"""
        try:
            self.documents.clear()
            self.document_metadata.clear()
            self.vector_store = None
            self.bm25_retriever = None
            
            self.logger.info("Cleared all documents from RAG system")
            
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
    
    def save_knowledge_base(self, path: str = None):
        """Save the current knowledge base to disk"""
        try:
            if path is None:
                path = self.medical_kb_path
            
            os.makedirs(path, exist_ok=True)
            
            # Save documents and metadata
            kb_data = {}
            for doc_id in self.documents:
                kb_data[doc_id] = {
                    'content': self.documents[doc_id],
                    'metadata': self.document_metadata[doc_id]
                }
            
            with open(f"{path}/documents.json", 'w') as f:
                json.dump(kb_data, f, indent=2)
            
            # Save vector store
            if self.vector_store:
                self.vector_store.save_local(f"{path}/vector_store")
            
            self.logger.info(f"Saved knowledge base to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
    
    def _load_vector_store(self):
        """Load vector store from disk"""
        try:
            vector_store_path = f"{self.medical_kb_path}/vector_store"
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.hf_embeddings
                )
                self.logger.info("Loaded vector store from disk")
                
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
    
    def search_by_category(self, query: str, category: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents by medical category"""
        try:
            # Filter documents by category if specified in metadata
            filtered_results = []
            
            all_results = self.query(query, max_results * 2)
            
            for result in all_results:
                doc_id = result['doc_id']
                if doc_id in self.document_metadata:
                    doc_category = self.document_metadata[doc_id].get('category', '').lower()
                    if category.lower() in doc_category:
                        filtered_results.append(result)
            
            return filtered_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error searching by category: {e}")
            return []
    
    def get_similar_documents(self, doc_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            if doc_id not in self.documents:
                return []
            
            # Use the document content as query
            doc_content = self.documents[doc_id]
            
            # Get a summary of the document (first 500 chars) for similarity search
            query_text = doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
            
            results = self.query(query_text, max_results + 1)  # +1 to exclude the original document
            
            # Filter out the original document
            similar_docs = [r for r in results if r['doc_id'] != doc_id]
            
            return similar_docs[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error finding similar documents: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        try:
            stats = {
                'total_documents': len(self.documents),
                'total_chunks': 0,
                'average_document_length': 0,
                'document_types': {},
                'sources': {}
            }
            
            if self.documents:
                # Calculate total chunks
                total_length = 0
                for content in self.documents.values():
                    chunks = self.text_splitter.split_text(content)
                    stats['total_chunks'] += len(chunks)
                    total_length += len(content)
                
                stats['average_document_length'] = total_length / len(self.documents)
                
                # Count document types and sources
                for metadata in self.document_metadata.values():
                    doc_type = metadata.get('doc_type', 'unknown')
                    source = metadata.get('source', 'unknown')
                    
                    stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
                    stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}