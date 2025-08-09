import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document


class DocumentChunker:
    """
    A comprehensive document chunker for RAG that loads documents from a folder
    and chunks them using LangChain's text splitters.
    """
    
    def __init__(
        self,
        documents_folder: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        output_file: str = "documents_chunks.json"
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            documents_folder (str): Path to the folder containing documents
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            separators (List[str], optional): Custom separators for text splitting
            output_file (str): File to save chunked documents
        """
        self.documents_folder = Path(documents_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_file = output_file
        
        # Default separators optimized for Vietnamese text
        self.separators = separators or [
            "\n\n",  # Double newline (paragraphs)
            "\n",    # Single newline
            ". ",    # Sentence end
            ".",     # Period
            " ",     # Space
            ""       # Character level
        ]
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        self.chunks = []
        self.metadata = {}
    
    def load_documents(self, file_extensions: List[str] = None) -> List[Document]:
        """
        Load all documents from the documents folder.
        
        Args:
            file_extensions (List[str]): List of file extensions to process
            
        Returns:
            List[Document]: List of loaded documents
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.text']
        
        documents = []
        
        if not self.documents_folder.exists():
            raise FileNotFoundError(f"Documents folder '{self.documents_folder}' not found!")
        
        # Find all files with specified extensions
        for extension in file_extensions:
            pattern = f"*{extension}"
            files = list(self.documents_folder.glob(pattern))
            
            for file_path in files:
                print(f"Loading document: {file_path.name}")
                try:
                    # Load document using TextLoader
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    doc = loader.load()[0]
                    
                    # Add metadata
                    doc.metadata.update({
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_size': file_path.stat().st_size,
                        'file_extension': file_path.suffix
                    })
                    
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Error loading {file_path.name}: {str(e)}")
                    continue
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Chunk the loaded documents into smaller pieces.
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[Dict[str, Any]]: List of chunked documents with metadata
        """
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            print(f"Chunking document: {document.metadata.get('filename', f'Document {doc_idx}')}")
            
            # Split document into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Process each chunk and add enhanced metadata
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': f"doc_{doc_idx}_chunk_{chunk_idx}",
                    'document_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'content': chunk.page_content,
                    'content_length': len(chunk.page_content),
                    'word_count': len(chunk.page_content.split()),
                    'metadata': {
                        **chunk.metadata,
                        'chunk_size': self.chunk_size,
                        'chunk_overlap': self.chunk_overlap,
                        'total_chunks_in_document': len(chunks)
                    }
                }
                
                all_chunks.append(chunk_data)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str = None) -> None:
        """
        Save chunks to a JSON file.
        
        Args:
            chunks (List[Dict[str, Any]]): List of chunked documents
            output_file (str, optional): Output file path
        """
        output_path = output_file or self.output_file
        
        # Create summary metadata
        summary = {
            'total_chunks': len(chunks),
            'total_documents': len(set(chunk['document_index'] for chunk in chunks)),
            'chunk_settings': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'separators': self.separators
            },
            'documents_folder': str(self.documents_folder),
            'created_at': str(Path().cwd())
        }
        
        # Prepare output data
        output_data = {
            'summary': summary,
            'chunks': chunks
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Chunks saved to: {output_path}")
        print(f"Summary: {summary['total_chunks']} chunks from {summary['total_documents']} documents")
    
    def process_documents(
        self, 
        file_extensions: List[str] = None,
        save_to_file: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: load documents, chunk them, and optionally save to file.
        
        Args:
            file_extensions (List[str]): File extensions to process
            save_to_file (bool): Whether to save chunks to file
            
        Returns:
            List[Dict[str, Any]]: List of processed chunks
        """
        print("Starting document processing pipeline...")
        print(f"Documents folder: {self.documents_folder}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        # Load documents
        documents = self.load_documents(file_extensions)
        
        if not documents:
            print("No documents found to process!")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Save chunks
        if save_to_file:
            self.save_chunks(chunks)
        
        # Store in instance for later access
        self.chunks = chunks
        
        return chunks
    
    def get_chunks_by_document(self, document_index: int) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_index (int): Index of the document
            
        Returns:
            List[Dict[str, Any]]: Chunks for the specified document
        """
        return [chunk for chunk in self.chunks if chunk['document_index'] == document_index]
    
    def search_chunks(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Simple text search within chunks.
        
        Args:
            query (str): Search query
            case_sensitive (bool): Whether search should be case sensitive
            
        Returns:
            List[Dict[str, Any]]: Matching chunks
        """
        if not case_sensitive:
            query = query.lower()
        
        matching_chunks = []
        for chunk in self.chunks:
            content = chunk['content'] if case_sensitive else chunk['content'].lower()
            if query in content:
                matching_chunks.append(chunk)
        
        return matching_chunks
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks.
        
        Returns:
            Dict[str, Any]: Statistics about chunks
        """
        if not self.chunks:
            return {"error": "No chunks available. Run process_documents() first."}
        
        chunk_lengths = [chunk['content_length'] for chunk in self.chunks]
        word_counts = [chunk['word_count'] for chunk in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(set(chunk['document_index'] for chunk in self.chunks)),
            'average_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'average_word_count': sum(word_counts) / len(word_counts),
            'total_characters': sum(chunk_lengths),
            'total_words': sum(word_counts)
        }


def main():
    """
    Example usage of the DocumentChunker.
    """
    # Initialize the chunker
    chunker = DocumentChunker(
        documents_folder="documents",
        chunk_size=1000,
        chunk_overlap=200,
        output_file="documents_chunks.json"
    )
    
    # Process documents
    chunks = chunker.process_documents()
    
    # Get statistics
    stats = chunker.get_chunk_statistics()
    print("\nChunking Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Example: Search for specific content
    print("\nSearching for 'Hồ Chí Minh':")
    matching_chunks = chunker.search_chunks("Hồ Chí Minh")
    print(f"Found {len(matching_chunks)} chunks containing 'Hồ Chí Minh'")
    
    if matching_chunks:
        print(f"First match preview: {matching_chunks[0]['content'][:200]}...")


if __name__ == "__main__":
    main()