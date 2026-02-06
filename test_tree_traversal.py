
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.tree_traversal import TreeNavigator, format_traversal_results
from src.core.reasoner import TreeRAGReasoner
from src.config import Config
import json


def create_mock_tree():
    """Create a mock document tree for testing."""
    return {
        "id": "root",
        "title": "Sample Medical Device Regulation Document",
        "summary": "Comprehensive guide for medical device approval process",
        "page_ref": "1-100",
        "children": [
            {
                "id": "chapter1",
                "title": "Chapter 1: General Requirements",
                "summary": "Basic regulatory requirements for all medical devices",
                "page_ref": "1-20",
                "children": [
                    {
                        "id": "section1.1",
                        "title": "Section 1.1: Device Classification",
                        "summary": "How to classify your device (Class I, II, III)",
                        "page_ref": "5-10",
                        "text": "Medical devices are classified into three classes based on risk. Class I devices pose minimal risk, Class II moderate risk, and Class III high risk. The classification determines regulatory requirements."
                    },
                    {
                        "id": "section1.2",
                        "title": "Section 1.2: Registration Process",
                        "summary": "Steps for device registration with regulatory bodies",
                        "page_ref": "11-15"
                    }
                ]
            },
            {
                "id": "chapter2",
                "title": "Chapter 2: Software Requirements",
                "summary": "Specific requirements for software as a medical device (SaMD)",
                "page_ref": "21-50",
                "children": [
                    {
                        "id": "section2.1",
                        "title": "Section 2.1: Software Validation",
                        "summary": "How to validate medical device software according to IEC 62304",
                        "page_ref": "25-35",
                        "text": "Software validation must demonstrate that the software meets user needs and intended uses. Testing must cover all specified requirements including functional, performance, and safety requirements."
                    },
                    {
                        "id": "section2.2",
                        "title": "Section 2.2: Cybersecurity",
                        "summary": "Cybersecurity requirements for connected medical devices",
                        "page_ref": "36-45"
                    }
                ]
            },
            {
                "id": "chapter3",
                "title": "Chapter 3: Clinical Evaluation",
                "summary": "Clinical evidence requirements and evaluation procedures",
                "page_ref": "51-80",
                "children": [
                    {
                        "id": "section3.1",
                        "title": "Section 3.1: Clinical Trials",
                        "summary": "When and how to conduct clinical trials",
                        "page_ref": "55-70"
                    }
                ]
            }
        ]
    }


def test_tree_navigator():
    print("\n" + "="*80)
    print("TEST 1: TreeNavigator Basic Functionality")
    print("="*80)
    
    tree = create_mock_tree()
    doc_name = "Mock_Medical_Device_Regulation"
    
    # Test Query 1: Software validation
    query1 = "How do I validate software for a medical device?"
    print(f"\n📝 Query: {query1}")
    
    navigator = TreeNavigator(tree, doc_name)
    results = navigator.search(query1, max_depth=3, max_branches=2)
    
    print(f"\n✅ Found {len(results)} relevant sections:")
    for result in results:
        node = result['node']
        print(f"   - {result['path']}")
        print(f"     Page: {node.get('page_ref', 'N/A')}")
        print(f"     Depth: {result['depth']}")
    
    # Test Query 2: Device classification
    query2 = "What is device classification and how does it work?"
    print(f"\n📝 Query: {query2}")
    
    navigator2 = TreeNavigator(tree, doc_name)
    results2 = navigator2.search(query2, max_depth=3, max_branches=2)
    
    print(f"\n✅ Found {len(results2)} relevant sections:")
    for result in results2:
        node = result['node']
        print(f"   - {result['path']}")
        print(f"     Page: {node.get('page_ref', 'N/A')}")


def test_context_size_comparison():
    print("\n" + "="*80)
    print("TEST 2: Context Size Comparison")
    print("="*80)
    
    tree = create_mock_tree()
    doc_name = "Mock_Medical_Device_Regulation"
    
    flat_context = json.dumps([{"document": doc_name, "content": tree}], ensure_ascii=False)
    flat_size = len(flat_context)
    
    query = "How to validate medical device software?"
    navigator = TreeNavigator(tree, doc_name)
    results = navigator.search(query, max_depth=3, max_branches=2)
    traversal_context = format_traversal_results(results, doc_name)
    traversal_size = len(traversal_context)
    
    print(f"\n📊 Context Size Comparison:")
    print(f"   Flat Context:      {flat_size:,} characters")
    print(f"   Traversal Context: {traversal_size:,} characters")
    print(f"   Reduction:         {flat_size - traversal_size:,} characters ({100 * (1 - traversal_size/flat_size):.1f}%)")
    
    if traversal_size < flat_size:
        print("\n✅ SUCCESS: Deep traversal reduces context size")
    else:
        print("\n⚠️  WARNING: Traversal context is larger (might need tuning)")


def test_format_output():
    print("\n" + "="*80)
    print("TEST 3: Formatted Output")
    print("="*80)
    
    tree = create_mock_tree()
    doc_name = "Mock_Medical_Device_Regulation"
    
    query = "Software validation requirements"
    navigator = TreeNavigator(tree, doc_name)
    results = navigator.search(query, max_depth=3, max_branches=2)
    
    formatted = format_traversal_results(results, doc_name)
    print("\n📄 Formatted Context for LLM:")
    print("-" * 80)
    print(formatted)
    print("-" * 80)


def test_mock_end_to_end():

    print("\n" + "="*80)
    print("TEST 4: End-to-End Logic Test (Mock)")
    print("="*80)
    
    tree = create_mock_tree()
    temp_file = os.path.join(Config.INDEX_DIR, "test_mock_index.json")
    
    os.makedirs(Config.INDEX_DIR, exist_ok=True)
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created mock index: {temp_file}")
    
    try:
        print("\n🌲 Testing with deep traversal enabled...")
        reasoner = TreeRAGReasoner(["test_mock_index.json"], use_deep_traversal=True)
        print("✅ TreeRAGReasoner initialized successfully with deep traversal")
        
        print("\n📄 Testing with flat context (legacy mode)...")
        reasoner_flat = TreeRAGReasoner(["test_mock_index.json"], use_deep_traversal=False)
        print("✅ TreeRAGReasoner initialized successfully with flat context")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"\n🧹 Cleaned up: {temp_file}")


def main():
    print("\n" + "="*80)
    print("🧪 DEEP TREE TRAVERSAL TEST SUITE")
    print("="*80)
    
    try:
        test_tree_navigator()
        test_context_size_comparison()
        test_format_output()
        test_mock_end_to_end()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)
        print("\nNote: These are logic tests with mock data.")
        print("To test with real documents, upload PDFs and run the full system.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
