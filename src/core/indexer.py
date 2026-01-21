import json
import os
import re
import time
from typing import Dict, Any
from pypdf import PdfReader
from src.config import Config

class RegulatoryIndexer:
    def __init__(self) -> None:
        self.generation_config: Dict[str, Any] = Config.GENERATION_CONFIG

    @staticmethod
    def _clean_markdown_json(text: str) -> str:
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'```json|```', '', text)
        text = text.strip()
        return text

    def extract_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        full_text = ""
        print(f"📄 PDF 로딩 중: {os.path.basename(pdf_path)} ({len(reader.pages)} pages)")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- [Page {i+1}] ---\n{text}"
        
        if not full_text:
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        return full_text

    def create_index(self, doc_title: str, full_text: str) -> Dict[str, Any]:
        if not doc_title or not full_text:
            raise ValueError("doc_title and full_text cannot be empty")
        
        print(f"🏗️ Indexing started for: {doc_title} (Model: {Config.MODEL_NAME})")
        
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
                cleaned_text = self._clean_markdown_json(response.text)
                result = json.loads(cleaned_text)
                print(f"✅ Indexing completed for: {doc_title}")
                return result
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                try:
                    print(f"Response text preview: {response.text[:500]}...")
                except:
                    pass
                return {}
            except Exception as e:
                error_str = str(e)
                print(f"❌ Indexing Failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⏳ Server overloaded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if "retry" in error_str.lower():
                        import re
                        match = re.search(r'retry in ([\d.]+)s', error_str)
                        if match and attempt < max_retries - 1:
                            wait_time = float(match.group(1)) + 1
                            print(f"⏳ Quota exceeded. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                
                return {}
        
        print(f"❌ All retry attempts failed for: {doc_title}")
        return {}

    def save_index(self, data: Dict, filename: str) -> None:
        if not data:
            print("⚠️ Warning: No data to save")
            return
        
        os.makedirs(Config.INDEX_DIR, exist_ok=True)
        path = os.path.join(Config.INDEX_DIR, filename)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved index to: {path}")
        except IOError as e:
            print(f"❌ Failed to save index: {e}")