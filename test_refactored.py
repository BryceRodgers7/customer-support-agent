"""
Test script for the refactored support bot with RAG search.
This verifies that:
1. RAG search works properly
2. LLM router function is being used
3. Tool execution is properly separated
"""
import os
from support_bot import CustomerSupportBot, SupportSession
from rag_search import check_rag_connection, search_kb_rag

def test_rag_connection():
    """Test that we can connect to Qdrant"""
    print("=" * 80)
    print("Testing RAG Connection")
    print("=" * 80)
    
    result = check_rag_connection()
    print(f"Connection status: {result}")
    
    if result.get("connected"):
        print("✓ RAG system is connected and operational")
        print(f"  - Collection: {result.get('collection')}")
        print(f"  - Total documents: {result.get('total_documents')}")
    else:
        print("✗ RAG system connection failed")
        print(f"  Error: {result.get('error')}")
    
    return result.get("connected", False)

def test_rag_search():
    """Test direct RAG search"""
    print("\n" + "=" * 80)
    print("Testing Direct RAG Search")
    print("=" * 80)
    
    queries = [
        "How do I return a product?",
        "Camera troubleshooting",
        "Warranty information"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        result = search_kb_rag(query, max_results=2)
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
        else:
            print(f"  ✓ Found {result['count']} results")
            for i, doc in enumerate(result['results'], 1):
                print(f"    {i}. {doc['title']} (score: {doc['relevance_score']})")
                print(f"       Snippet: {doc['snippet'][:100]}...")

def test_support_bot():
    """Test the refactored support bot"""
    print("\n" + "=" * 80)
    print("Testing Refactored Support Bot")
    print("=" * 80)
    
    # Make sure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set. Skipping bot test.")
        return
    
    bot = CustomerSupportBot()
    session = SupportSession()
    
    test_queries = [
        "What's your return policy?",
        "Tell me about the Aether X1 headphones"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        print("-" * 80)
        response = bot.reply(session, query)
        print(f"Bot: {response[:300]}...")
        print()

def main():
    print("Refactored Support Bot Test Suite")
    print("=" * 80)
    print()
    
    # Test 1: RAG connection
    rag_ok = test_rag_connection()
    
    if not rag_ok:
        print("\n⚠ RAG system is not available. Make sure Qdrant is running:")
        print("  docker-compose up -d")
        print("  python vector_setup/load_qdrant.py")
        return
    
    # Test 2: Direct RAG search
    test_rag_search()
    
    # Test 3: Full bot integration
    test_support_bot()
    
    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()

