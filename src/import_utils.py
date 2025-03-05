import os
import hashlib
import json
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import PyPDF2
import docx
import hashlib
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any
import uuid

# Load environment variables
load_dotenv()

class ElasticsearchConnector:
    """Class to handle Elasticsearch connection and operations"""
    
    def __init__(self, hosts: Optional[List[str]] = None, index_prefix: str = "rag"):
        """Initialize Elasticsearch connection
        
        Args:
            hosts: List of Elasticsearch hosts. If None, uses ES_HOSTS from environment
            index_prefix: Prefix for Elasticsearch indices
        """
        if hosts is None:
            es_hosts = os.getenv("ES_HOSTS", "http://localhost:9200")
            hosts = [host.strip() for host in es_hosts.split(",")]
        
        # Create Elasticsearch client
        self.client = Elasticsearch(hosts)
        
        # Set up indices
        self.index_prefix = index_prefix
        self.doc_index = f"{index_prefix}_documents"
        self.chunk_index = f"{index_prefix}_chunks"
        
        # Create indices if they don't exist
        self._create_indices()
        
    def _create_indices(self):
        """Create the necessary Elasticsearch indices if they don't exist"""
        # Documents index
        if not self.client.indices.exists(index=self.doc_index):
            self.client.indices.create(
                index=self.doc_index,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "content_length": {"type": "integer"},
                            "metadata": {"type": "object", "dynamic": True},
                            "filename": {"type": "keyword"}
                        }
                    }
                }
            )
            print(f"Created index: {self.doc_index}")
        
        # Chunks index
        if not self.client.indices.exists(index=self.chunk_index):
            self.client.indices.create(
                index=self.chunk_index,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "content": {
                                "type": "text", 
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "chunk_index": {"type": "integer"},
                            "metadata": {"type": "object", "dynamic": True},
                            "vector": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            )
            print(f"Created index: {self.chunk_index}")
    
    def close(self):
        """Close Elasticsearch connection"""
        pass  # Elasticsearch client handles connection pooling automatically
        
    def insert_document(self, document: Dict[str, Any]) -> str:
        """Insert a document into the documents index
        
        Args:
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        doc_id = document.get("doc_id")
        if not doc_id:
            raise ValueError("Document must have a doc_id field")
            
        self.client.index(
            index=self.doc_index,
            id=doc_id,
            document=document,
            refresh=True
        )
        return doc_id
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Insert document chunks into the chunks index
        
        Args:
            chunks: List of chunks to insert
            
        Returns:
            List of inserted chunk IDs
        """
        if not chunks:
            return []
        
        chunk_ids = []
        actions = []
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError("Each chunk must have a chunk_id field")
                
            chunk_ids.append(chunk_id)
            action = {
                "_index": self.chunk_index,
                "_id": chunk_id,
                "_source": chunk
            }
            actions.append(action)
            
        helpers.bulk(self.client, actions, refresh=True)
        return chunk_ids
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dict if found, None otherwise
        """
        try:
            result = self.client.get(index=self.doc_index, id=doc_id)
            return result["_source"]
        except Exception:
            return None
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dicts
        """
        query = {
            "query": {
                "term": {
                    "doc_id": doc_id
                }
            },
            "sort": [
                {"chunk_index": {"order": "asc"}}
            ],
            "size": 10000  # Adjust if you have more chunks per document
        }
        
        result = self.client.search(index=self.chunk_index, body=query)
        return [hit["_source"] for hit in result["hits"]["hits"]]
    
    def search_chunks(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks using text query
        
        Args:
            query_text: The text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunks
        """
        query = {
            "query": {
                "match": {
                    "content": query_text
                }
            },
            "size": limit
        }
        
        result = self.client.search(index=self.chunk_index, body=query)
        return [hit["_source"] for hit in result["hits"]["hits"]]


class DocumentProcessor:
    """Class to process documents for import into Elasticsearch"""
    
    def __init__(self, es_connector: ElasticsearchConnector, chunk_size: int = 1000):
        """Initialize document processor
        
        Args:
            es_connector: Elasticsearch connector instance
            chunk_size: Size of text chunks in characters
        """
        self.es = es_connector
        self.chunk_size = chunk_size
        
    def _generate_doc_id(self, filepath: str, content: str) -> str:
        """Generate a unique document ID based on filename and content hash
        
        Args:
            filepath: Path to the document file
            content: Document content
            
        Returns:
            Unique document ID
        """
        filename = os.path.basename(filepath)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{filename}_{content_hash[:10]}"
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID
        
        Args:
            doc_id: Document ID
            chunk_index: Chunk index
            
        Returns:
            Unique chunk ID
        """
        return f"{doc_id}_chunk_{chunk_index}"
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size characters
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the end of the chunk
            chunk_end = min(current_pos + self.chunk_size, len(text))
            
            # If we're not at the end of the text, try to break at a sentence or paragraph
            if chunk_end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind('\n\n', current_pos, chunk_end)
                if paragraph_break != -1 and paragraph_break > current_pos + self.chunk_size // 2:
                    chunk_end = paragraph_break + 2
                else:
                    # Try to find sentence break (period followed by space)
                    sentence_break = text.rfind('. ', current_pos, chunk_end)
                    if sentence_break != -1 and sentence_break > current_pos + self.chunk_size // 2:
                        chunk_end = sentence_break + 2
            
            chunks.append(text[current_pos:chunk_end].strip())
            current_pos = chunk_end
            
        return chunks
    
    def process_text_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a plain text file
        
        Args:
            filepath: Path to the text file
            metadata: Additional metadata to store
            
        Returns:
            Document ID
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return self.process_text(content, filepath, metadata)
    
    def process_pdf_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a PDF file
        
        Args:
            filepath: Path to the PDF file
            metadata: Additional metadata to store
            
        Returns:
            Document ID
        """
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        return self.process_text(text, filepath, metadata)
    
    def process_docx_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a DOCX file
        
        Args:
            filepath: Path to the DOCX file
            metadata: Additional metadata to store
            
        Returns:
            Document ID
        """
        doc = docx.Document(filepath)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        return self.process_text(text, filepath, metadata)
    
    def process_text(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process text content
        
        Args:
            text: Text content
            source: Source identifier (e.g., filepath)
            metadata: Additional metadata to store
            
        Returns:
            Document ID
        """
        if not text.strip():
            raise ValueError(f"Empty content from source: {source}")
            
        # Create document record
        doc_id = self._generate_doc_id(source, text)
        doc_metadata = metadata or {}
        
        document = {
            "doc_id": doc_id,
            "source": source,
            "content_length": len(text),
            "metadata": doc_metadata,
            "filename": os.path.basename(source) if os.path.exists(source) else None
        }
        
        # Insert document
        self.es.insert_document(document)
        
        # Process chunks
        text_chunks = self._chunk_text(text)
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "chunk_id": self._generate_chunk_id(doc_id, i),
                "doc_id": doc_id,
                "content": chunk_text,
                "chunk_index": i,
                "metadata": doc_metadata.copy() if doc_metadata else {}
                # Note: Vector embeddings should be added here, but we leave as None for now
                # as they would typically be added by a separate embedding process
            }
            chunks.append(chunk)
            
        # Insert chunks
        self.es.insert_chunks(chunks)
        
        return doc_id
        
    def process_file(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a file based on its extension
        
        Args:
            filepath: Path to the file
            metadata: Additional metadata to store
            
        Returns:
            Document ID
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.txt':
            return self.process_text_file(filepath, metadata)
        elif ext == '.pdf':
            return self.process_pdf_file(filepath, metadata)
        elif ext == '.docx':
            return self.process_docx_file(filepath, metadata)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
    def process_directory(self, directory: str, recursive: bool = True, 
                         extensions: List[str] = ['.txt', '.pdf', '.docx'],
                         metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process all files in a directory
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            extensions: List of file extensions to process
            metadata: Additional metadata to store for all files
            
        Returns:
            List of processed document IDs
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
            
        processed_ids = []
        
        for root, dirs, files in os.walk(directory):
            if not recursive and root != directory:
                continue
                
            for file in tqdm(files, desc=f"Processing files in {root}"):
                filepath = os.path.join(root, file)
                ext = os.path.splitext(filepath)[1].lower()
                
                if ext in extensions:
                    try:
                        file_metadata = metadata.copy() if metadata else {}
                        file_metadata['directory'] = os.path.relpath(root, directory)
                        
                        doc_id = self.process_file(filepath, file_metadata)
                        processed_ids.append(doc_id)
                    except Exception as e:
                        print(f"Error processing {filepath}: {str(e)}")
        
        return processed_ids


def main():
    """Simple CLI for document import"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import documents to Elasticsearch for RAG system")
    parser.add_argument("source", help="File or directory to process")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--index-prefix", default="rag", help="Elasticsearch index prefix")
    parser.add_argument("--es-hosts", help="Elasticsearch hosts (comma separated)")
    
    args = parser.parse_args()
    
    # Setup Elasticsearch connection
    hosts = args.es_hosts.split(",") if args.es_hosts else None
    es = ElasticsearchConnector(hosts=hosts, index_prefix=args.index_prefix)
    
    # Initialize document processor
    processor = DocumentProcessor(es, args.chunk_size)
    
    try:
        # Process source
        if os.path.isfile(args.source):
            doc_id = processor.process_file(args.source)
            print(f"Processed file: {args.source}")
            print(f"Document ID: {doc_id}")
        elif os.path.isdir(args.source):
            doc_ids = processor.process_directory(args.source, recursive=args.recursive)
            print(f"Processed {len(doc_ids)} documents from directory: {args.source}")
        else:
            print(f"Source not found: {args.source}")
    finally:
        es.close()


if __name__ == "__main__":
    main()