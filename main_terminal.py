import os
from typing import Optional, List
from src.core.indexer import RegulatoryIndexer
from src.core.reasoner import RegulatoryReasoner
from src.config import Config

def _ensure_data_directories() -> None:
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(Config.INDEX_DIR, exist_ok=True)

def _find_pdf_files() -> List[str]:
    pdf_files = []
    for filename in os.listdir(Config.RAW_DATA_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(filename)
    return sorted(pdf_files)

def _select_pdf(pdf_files: List[str]) -> Optional[str]:
    if not pdf_files:
        print(f"⚠️ No PDF files found in: {Config.RAW_DATA_DIR}/")
        print("📁 Please place a PDF file in the directory.")
        return None
    
    if len(pdf_files) == 1:
        print(f"📄 Found PDF: {pdf_files[0]}")
        return pdf_files[0]
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    
    while True:
        try:
            choice = input(f"\nSelect PDF (1-{len(pdf_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(pdf_files):
                return pdf_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(pdf_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return None

def _process_document(pdf_path: str, pdf_filename: str, index_filename: str) -> bool:
    print("\n--- 🏗️ Phase 1: Creating Knowledge Tree ---")
    try:
        indexer = RegulatoryIndexer()
        text = indexer.extract_text(pdf_path)
        tree = indexer.create_index("Medical Device Guideline", text)
        indexer.save_index(tree, index_filename)
        return True
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        return False

def _run_consultation(index_filename: str) -> None:
    print("\n--- 🧠 Phase 2: AI Consultant Ready ---")
    try:
        reasoner = RegulatoryReasoner(index_filename)
        
        while True:
            q = input("\nQ (종료: q): ").strip()
            if q.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not q:
                print("Please enter a question.")
                continue
            
            print("\nThinking...", end="", flush=True)
            ans = reasoner.query(q)
            print(f"\r[Answer]:\n{ans}")
    except FileNotFoundError as e:
        print(f"❌ Index file not found: {e}")
    except Exception as e:
        print(f"❌ Error during consultation: {e}")

def main() -> None:
    _ensure_data_directories()
    
    pdf_files = _find_pdf_files()
    pdf_filename = _select_pdf(pdf_files)
    
    if not pdf_filename:
        return
    
    pdf_path = os.path.join(Config.RAW_DATA_DIR, pdf_filename)
    index_filename = pdf_filename.replace('.pdf', '_index.json')
    index_path = os.path.join(Config.INDEX_DIR, index_filename)

    if not os.path.exists(index_path):
        if not _process_document(pdf_path, pdf_filename, index_filename):
            return
    else:
        print(f"\n--- ✅ Found existing index: {index_filename} ---")

    _run_consultation(index_filename)

if __name__ == "__main__":
    main()
