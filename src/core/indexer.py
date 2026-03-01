import json
import os
import re
import time
import logging
from typing import Dict, Any, Generator
from pypdf import PdfReader
from pydantic import ValidationError
from src.config import Config
from src.models.schemas import PageNode

logger = logging.getLogger(__name__)

class RegulatoryIndexer:
    def __init__(self) -> None:
        self.generation_config: Any = Config.get_generation_config()

    @staticmethod
    def _clean_markdown_json(text: str) -> str:
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'```json|```', '', text)
        text = text.strip()
        return text

    def extract_text_stream(self, pdf_path: str) -> Generator[tuple[int, str], None, None]:
        """Extract text page by page as generator to minimize memory usage."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"PDF loading: {os.path.basename(pdf_path)} ({total_pages} pages)")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                yield (i + 1, text)

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text at once (backward compatibility).
        
        WARNING: For large PDFs (1000+ pages), prefer extract_text_stream()
        or create_index_from_stream() to avoid memory issues.
        """
        text_parts = []
        
        for page_num, text in self.extract_text_stream(pdf_path):
            text_parts.append(f"\n--- [Page {page_num}] ---\n{text}")
        
        if not text_parts:
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        return "".join(text_parts)
    
    def create_index_from_stream(self, doc_title: str, pdf_path: str, max_pages_per_chunk: int = 100) -> Dict[str, Any]:
        """
        Create index directly from PDF stream without loading entire document.
        Processes PDF in chunks to minimize memory footprint.
        
        Args:
            doc_title: Document title
            pdf_path: Path to PDF file
            max_pages_per_chunk: Maximum pages to process in one chunk
        
        Returns:
            Structured JSON tree index
        """
        if not doc_title:
            raise ValueError("doc_title cannot be empty")
        
        logger.info(f"Streaming Indexing started for: {doc_title} (Model: {Config.MODEL_NAME})")
        
        chunk_parts = []
        page_count = 0
        
        for page_num, text in self.extract_text_stream(pdf_path):
            chunk_parts.append(f"\n--- [Page {page_num}] ---\n{text}")
            page_count += 1
            
            if page_count >= max_pages_per_chunk:
                break
        
        if not chunk_parts:
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        chunk_text = "".join(chunk_parts)
        
        logger.info(f"Processing {page_count} pages (chunk size: {len(chunk_text)} chars)")
        
        return self.create_index(doc_title, chunk_text)

    def create_index(self, doc_title: str, full_text: str) -> Dict[str, Any]:
        if not doc_title or not full_text:
            raise ValueError("doc_title and full_text cannot be empty")
        
        logger.info(f"Indexing started for: {doc_title} (Model: {Config.MODEL_NAME})")
        
        prompt = f"""
        You are a Legal & Regulatory Expert AI.
        Your task is to convert the provided regulatory document text into a structured JSON tree.
        
        ### Structure Requirements:
        - Root: Document Title
        - Children: Chapters -> Sections -> Articles
        - Each node MUST have: "id", "title", "summary", "page_ref" (e.g., "12-15").
        
        ### Text to Analyze:
        Title: {doc_title}
        Content:
        {full_text}
        """

        max_retries = 3
        retry_delay = 15
        
        for attempt in range(max_retries):
            try:
                response = Config.CLIENT.models.generate_content(
                    model=Config.MODEL_NAME,
                    contents=prompt,
                    config=self.generation_config
                )
                cleaned_text = self._clean_markdown_json(response.text or "")
                logger.debug(f"API Response (first 500 chars): {cleaned_text[:500]}")
                result = json.loads(cleaned_text)
                logger.debug(f"Parsed JSON type: {type(result).__name__}")
                if isinstance(result, list):
                    logger.debug(f"List length: {len(result)}")
                    if len(result) > 0:
                        logger.debug(f"First item type: {type(result[0]).__name__}")
                        if isinstance(result[0], dict):
                            logger.debug("Converting list to dict (using first element)")
                            result = result[0]
                        else:
                            raise ValueError("List does not contain a dictionary")
                    else:
                        raise ValueError("Empty list returned")
                
                if not isinstance(result, dict):
                    raise ValueError("Root JSON must be a dictionary")
                
                try:
                    PageNode.model_validate(result)
                    logger.info(f"Indexing completed and validated for: {doc_title}")
                except ValidationError as ve:
                    logger.warning(f"Pydantic validation failed: {ve}")
                    raise ValueError(f"Schema validation failed: {ve}")
                
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return {}
            except ValueError as e:
                logger.error(f"Invalid JSON structure: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return {}
            except Exception as e:
                error_str = str(e)
                logger.error(f"Indexing Failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.info(f"Server overloaded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if "retry" in error_str.lower():
                        import re
                        match = re.search(r'retry in ([\d.]+)s', error_str)
                        if match and attempt < max_retries - 1:
                            wait_time = float(match.group(1)) + 1
                            logger.info(f"Quota exceeded. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
        
        logger.error(f"All retry attempts failed for: {doc_title}")
        return {}

    def save_index(self, data: Dict, filename: str) -> None:
        if not data:
            logger.warning("Warning: No data to save")
            return
        
        os.makedirs(Config.INDEX_DIR, exist_ok=True)
        path = os.path.join(Config.INDEX_DIR, filename)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved index to: {path}")
        except IOError as e:
            print(f"❌ Failed to save index: {e}")