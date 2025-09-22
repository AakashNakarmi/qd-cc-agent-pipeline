import json
import logging
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import pandas as pd
from threading import Lock
import time
from datetime import datetime
import math


# Configure enhanced logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better formatting"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add timestamp and color
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format: [HH:MM:SS.mmm] LEVEL | message
        formatted = f"[{timestamp}] {color}{record.levelname:8}{reset} | {record.getMessage()}"
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted

# Setup enhanced logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler('ai_search_processing.log')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

class ProgressTracker:
    """Track and display progress with detailed metrics"""
    
    def __init__(self, total_items: int, operation_name: str):
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self._lock = Lock()
    
    def update(self, increment: int = 1):
        with self._lock:
            self.processed_items += increment
            current_time = time.time()
            
            # Update every 5 seconds or every 1000 items
            if (current_time - self.last_update > 5.0) or (self.processed_items % 1000 == 0):
                self._log_progress()
                self.last_update = current_time
    
    def _log_progress(self):
        elapsed = time.time() - self.start_time
        progress_pct = (self.processed_items / self.total_items) * 100
        items_per_sec = self.processed_items / elapsed if elapsed > 0 else 0
        
        if items_per_sec > 0:
            eta_seconds = (self.total_items - self.processed_items) / items_per_sec
            eta_str = f" | ETA: {eta_seconds/60:.1f}m" if eta_seconds > 60 else f" | ETA: {eta_seconds:.0f}s"
        else:
            eta_str = ""
        
        logger.info(
            f"{self.operation_name}: {self.processed_items:,}/{self.total_items:,} "
            f"({progress_pct:.1f}%) | {items_per_sec:.1f} items/sec{eta_str}"
        )
    
    def complete(self):
        elapsed = time.time() - self.start_time
        items_per_sec = self.processed_items / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.operation_name} COMPLETED: {self.processed_items:,} items "
            f"in {elapsed/60:.1f}m ({items_per_sec:.1f} items/sec)"
        )


class AISearchVectorService:
    def __init__(self, 
                 search_service_name: Optional[str] = None,
                 search_api_key: Optional[str] = None,
                 azure_openai_api_key: Optional[str] = None,
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_embedding_deployment: Optional[str] = None,
                 index_name: Optional[str] = None,
                 max_workers: int = 8,  # Reduced for upload stability
                 embedding_batch_size: int = 100,  # Keep high for embeddings
                 upload_batch_size: int = 500,  # Reduced upload batch size
                 upload_workers: int = 3):  # Separate upload worker count
        
        # Azure Search config
        self.search_service_name = search_service_name or os.getenv('AZURE_SEARCH_SERVICE_NAME', 'azs-qd-test-eus-01')
        self.search_api_key = search_api_key or os.getenv('AZURE_SEARCH_API_KEY')
        self.index_name = index_name or os.getenv('AZURE_SEARCH_INDEX_NAME', 'qd-cc-boqitems-index')

        # Azure OpenAI config
        azure_openai_key = azure_openai_api_key or os.getenv('AZURE_OPENAI_API_KEY')
        azure_openai_endpoint = azure_openai_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT', 'https://oai-cgb-jobmatching-sc-01.openai.azure.com/')
        self.azure_openai_embedding_deployment = azure_openai_embedding_deployment or os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT','text-embedding-3-large')

        # Performance config
        self.max_workers = max_workers
        self.embedding_batch_size = embedding_batch_size
        self.upload_batch_size = upload_batch_size
        self.upload_workers = upload_workers  # Separate worker count for uploads
        self._embedding_lock = Lock()
        self._upload_lock = Lock()  # Additional lock for upload coordination

        if not self.search_service_name:
            raise ValueError("Azure Search service name is required.")
        if not self.search_api_key:
            raise ValueError("Azure Search API key is required.")
        if not azure_openai_key or not azure_openai_endpoint or not self.azure_openai_embedding_deployment:
            raise ValueError("Azure OpenAI API key, endpoint, and embedding deployment name are required.")

        # Azure Search clients
        self.search_endpoint = f"https://{self.search_service_name}.search.windows.net"
        credential = AzureKeyCredential(self.search_api_key)

        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=credential
        )
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=credential
        )

        # Azure OpenAI client
        self.azure_openai_client = AzureOpenAI(
            api_key=azure_openai_key,
            api_version="2024-02-01",
            azure_endpoint=azure_openai_endpoint
        )

        self.vector_dimensions = int(os.getenv('VECTOR_DIMENSIONS', '3072'))
        
        logger.info(f"Initialized AISearchVectorService:")
        logger.info(f"  - Embedding workers: {self.max_workers}")
        logger.info(f"  - Upload workers: {self.upload_workers}")
        logger.info(f"  - Embedding batch size: {self.embedding_batch_size}")
        logger.info(f"  - Upload batch size: {self.upload_batch_size}")
        logger.info(f"  - Vector dimensions: {self.vector_dimensions}")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.azure_openai_client.embeddings.create(
                    model=self.azure_openai_embedding_deployment,
                    input=texts
                )
                return [data.embedding for data in response.data]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                    return [[0.0] * self.vector_dimensions] * len(texts)

    def create_vector_index(self) -> bool:
        logger.info("Checking/creating vector index...")
        try:
            try:
                existing_index = self.index_client.get_index(self.index_name)
                logger.info(f" Index '{self.index_name}' already exists")
                return True
            except Exception:
                logger.info(f"Index '{self.index_name}' doesn't exist, creating...")

            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="item_id", type=SearchFieldDataType.String),
                SimpleField(name="project_id", type=SearchFieldDataType.String),
                SimpleField(name="section_id", type=SearchFieldDataType.String),
                SimpleField(name="sheet_name", type=SearchFieldDataType.String),
                SimpleField(name="source_row", type=SearchFieldDataType.Int32),
                SimpleField(name="item_number", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SimpleField(name="quantity", type=SearchFieldDataType.Double),
                SimpleField(name="unit", type=SearchFieldDataType.String),
                SimpleField(name="unit_rate", type=SearchFieldDataType.Double),
                SimpleField(name="total_cost", type=SearchFieldDataType.Double),
                SearchableField(name="vector_content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.vector_dimensions,
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 16,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )

            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            result = self.index_client.create_index(index)
            logger.info(f" Created index: {result.name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating index: {str(e)}")
            return False

    def load_csv_data(self, csv_path: str = "boqitems.csv") -> List[Dict[str, Any]]:
        """Load BOQ items from CSV file with optimized pandas reading"""
        logger.info(f"Loading CSV data from {csv_path}...")
        start_time = time.time()
        
        try:
            # Optimize pandas reading
            df = pd.read_csv(
                csv_path,
                dtype={
                    'ItemID': 'string',
                    'ProjectID': 'string',
                    'SectionID': 'string',
                    'SheetName': 'string',
                    'ItemNumber': 'string',
                    'Description': 'string',
                    'Unit': 'string'
                },
                engine='c'  # Use C engine for better performance
            )
            
            load_time = time.time() - start_time
            logger.info(f" Loaded {len(df):,} records in {load_time:.2f}s")
            
            # Convert to dict efficiently
            items = df.to_dict('records')
            logger.info(f" Converted to dictionary format")
            
            return items
            
        except Exception as e:
            logger.error(f"✗ Error loading CSV: {str(e)}")
            return []

    def process_embedding_batch(self, items_batch: List[Dict[str, Any]], batch_id: int, 
                              progress_tracker: ProgressTracker) -> List[Dict[str, Any]]:
        """Process a batch of items to generate embeddings"""
        try:
            # Extract valid texts
            texts = []
            valid_items = []
            
            for item in items_batch:
                vector_content = str(item.get('Description', '')).strip()
                if vector_content and vector_content.lower() != 'nan':
                    texts.append(vector_content)
                    valid_items.append(item)
            
            if not texts:
                progress_tracker.update(len(items_batch))
                return []
            
            # Process texts in sub-batches for embedding API
            documents = []
            for i in range(0, len(texts), self.embedding_batch_size):
                sub_batch_texts = texts[i:i + self.embedding_batch_size]
                sub_batch_items = valid_items[i:i + self.embedding_batch_size]
                
                # Generate embeddings
                embeddings = self.generate_embeddings_batch(sub_batch_texts)
                
                # Create documents
                for item, embedding in zip(sub_batch_items, embeddings):
                    doc = self.create_document(item, embedding)
                    if doc:
                        documents.append(doc)
                
                progress_tracker.update(len(sub_batch_items))
                
                # Small delay to respect rate limits
                time.sleep(0.01)
            
            return documents
            
        except Exception as e:
            logger.error(f"✗ Error processing batch {batch_id}: {str(e)}")
            progress_tracker.update(len(items_batch))
            return []

    def create_document(self, item: Dict[str, Any], embedding: List[float]) -> Optional[Dict[str, Any]]:
        """Create a search document from item and embedding"""
        try:
            unique_id = sanitize_key(str(item.get('ItemID', '')))
            
            # Handle potential None/NaN values efficiently
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value is not None and str(value).lower() != 'nan' else default
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                try:
                    return int(value) if value is not None and str(value).lower() != 'nan' else default
                except (ValueError, TypeError):
                    return default
            
            def safe_str(value, default=''):
                return str(value) if value is not None and str(value).lower() != 'nan' else default
            
            return {
                'id': unique_id,
                'item_id': safe_str(item.get('ItemID')),
                'project_id': safe_str(item.get('ProjectID')),
                'section_id': safe_str(item.get('SectionID')),
                'sheet_name': safe_str(item.get('SheetName')),
                'source_row': safe_int(item.get('SourceRow')),
                'item_number': safe_str(item.get('ItemNumber')),
                'description': safe_str(item.get('Description')),
                'quantity': safe_float(item.get('Quantity')),
                'unit': safe_str(item.get('Unit')),
                'unit_rate': safe_float(item.get('UnitRate')),
                'total_cost': safe_float(item.get('TotalCost')),
                'vector_content': safe_str(item.get('Description')).strip(),
                'content_vector': embedding
            }
            
        except Exception as e:
            logger.error(f"✗ Error creating document for item {item.get('ItemID', 'unknown')}: {str(e)}")
            return None

    def prepare_documents_optimized(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare documents with optimized parallel processing"""
        if not items:
            return []
        
        logger.info(f"Starting optimized document preparation for {len(items):,} items")
        start_time = time.time()
        
        # Progress tracking
        progress_tracker = ProgressTracker(len(items), "Document Preparation")
        
        # Calculate optimal chunk size based on embedding batch size
        chunk_size = self.embedding_batch_size * 5  # Process multiple embedding batches per chunk
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        logger.info(f"Processing in {len(chunks)} chunks of ~{chunk_size} items each")
        logger.info(f"Using {self.max_workers} parallel workers")
        
        all_documents = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_batch = {
                executor.submit(self.process_embedding_batch, chunk, i, progress_tracker): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            completed_chunks = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    completed_chunks += 1
                    
                    if completed_chunks % 10 == 0:
                        logger.info(f"Completed {completed_chunks}/{len(chunks)} chunks")
                        
                except Exception as e:
                    logger.error(f"✗ Chunk {batch_id} failed: {str(e)}")
        
        progress_tracker.complete()
        
        preparation_time = time.time() - start_time
        logger.info(f" Document preparation completed: {len(all_documents):,} documents in {preparation_time/60:.1f}m")
        
        return all_documents

    def upload_batch_to_search(self, batch: List[Dict[str, Any]], batch_num: int, 
                              total_batches: int, progress_tracker: ProgressTracker) -> bool:
        """Upload a single batch to Azure Search with enhanced error handling and connection management"""
        max_retries = 3
        base_retry_delay = 2.0  # Increased base delay
        
        for attempt in range(max_retries):
            try:
                # Add delay between uploads to prevent connection overload
                with self._upload_lock:
                    time.sleep(0.5)  # Stagger uploads
                
                # Create a fresh client for each upload to avoid connection pooling issues
                credential = AzureKeyCredential(self.search_api_key)
                fresh_search_client = SearchClient(
                    endpoint=self.search_endpoint,
                    index_name=self.index_name,
                    credential=credential
                )
                
                result = fresh_search_client.upload_documents(documents=batch)
                
                # Check for failures
                failed = [r for r in result if not r.succeeded]
                if failed:
                    logger.warning(f"Batch {batch_num}: {len(failed)} docs failed: {[f.key[:20] for f in failed[:3]]}")
                    if len(failed) > len(batch) * 0.2:  # If >20% failed, consider it a failure
                        raise Exception(f"Too many failures: {len(failed)}/{len(batch)}")
                
                progress_tracker.update(len(batch))
                logger.info(f" Batch {batch_num}/{total_batches} uploaded successfully")
                return True
                
            except Exception as e:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                
                if attempt < max_retries - 1:
                    logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"✗ Batch {batch_num} failed after {max_retries} attempts: {str(e)}")
                    progress_tracker.update(len(batch))
                    return False

    def upload_documents_optimized(self, documents: List[Dict[str, Any]]) -> bool:
        """Upload documents with optimized parallel processing"""
        if not documents:
            logger.error("No documents to upload")
            return False
        
        logger.info(f"Starting optimized document upload for {len(documents):,} documents")
        start_time = time.time()
        
        # Progress tracking
        progress_tracker = ProgressTracker(len(documents), "Document Upload")
        
        # Create batches
        batches = [documents[i:i + self.upload_batch_size] 
                  for i in range(0, len(documents), self.upload_batch_size)]
        total_batches = len(batches)
        
        logger.info(f"Uploading in {total_batches} batches of {self.upload_batch_size} documents each")
        
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, total_batches)) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.upload_batch_to_search, batch, i + 1, total_batches, progress_tracker): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future] + 1
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"✗ Batch {batch_num} processing error: {str(e)}")
        
        progress_tracker.complete()
        
        upload_time = time.time() - start_time
        success_rate = (success_count / total_batches) * 100
        
        logger.info(f"Upload completed: {success_count}/{total_batches} batches successful ({success_rate:.1f}%)")
        logger.info(f"Total upload time: {upload_time/60:.1f}m")
        
        return success_count == total_batches

    def push_items(self, csv_path: str = "boqitems.csv") -> bool:
        """Main method with comprehensive optimization and logging"""
        logger.info("=" * 60)
        logger.info("STARTING OPTIMIZED AI SEARCH INDEXING PIPELINE")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        try:
            # 1. Index creation/verification
            logger.info("STEP 1: Index Setup")
            if not self.create_vector_index():
                logger.error("✗ Index setup failed")
                return False
            
            # 2. Load CSV data
            logger.info("STEP 2: Data Loading")
            items = self.load_csv_data(csv_path)
            if not items:
                logger.error("✗ No data loaded")
                return False
            
            # 3. Document preparation (with embeddings)
            logger.info("STEP 3: Document Preparation & Embedding Generation")
            documents = self.prepare_documents_optimized(items)
            if not documents:
                logger.error("✗ No documents prepared")
                return False
            
            # 4. Document upload
            logger.info("STEP 4: Document Upload to Azure Search")
            success = self.upload_documents_optimized(documents)
            
            # 5. Final summary
            total_time = time.time() - total_start_time
            logger.info("=" * 60)
            if success:
                logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
                logger.info(f"Processed {len(items):,} items → {len(documents):,} documents")
                logger.info(f"Total time: {total_time/60:.1f} minutes ({len(items)/(total_time/60):.0f} items/min)")
            else:
                logger.error("PIPELINE COMPLETED WITH ERRORS")
            logger.info("=" * 60)
            
            return success
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"CRITICAL ERROR after {total_time/60:.1f}m: {str(e)}")
            return False


def sanitize_key(key: str) -> str:
    """Sanitize keys for Azure Search"""
    return re.sub(r'[^A-Za-z0-9_\-=]', '_', str(key))


def main():
    # Initialize with conservative settings to avoid SSL issues
    search_service = AISearchVectorService(
        max_workers=12,          # Good for embeddings
        embedding_batch_size=100,  # Keep high for efficiency
        upload_batch_size=500,   # Reduced to avoid overloading
        upload_workers=3         # Very conservative for uploads
    )
    
    # Run the optimized pipeline
    success = search_service.push_items("boqitems.csv")
    
    if success:
        print("\nSuccess! All BOQ items have been indexed!")
    else:
        print("\n⚠️  Process completed with some errors. Check logs for details.")
        print("💡 Tip: You can re-run to retry failed batches - duplicates will be updated automatically.")


if __name__ == "__main__":
    main()