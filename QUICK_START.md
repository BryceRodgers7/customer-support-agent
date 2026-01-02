# Quick Start Guide: Refactored Support Bot

## Overview
The support bot now uses RAG (Retrieval Augmented Generation) for knowledge base search and has a cleaner architecture with explicit routing and tool execution functions.

## Prerequisites
1. Python 3.8+
2. Docker (for Qdrant)
3. OpenAI API key

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- openai
- streamlit
- qdrant-client with fastembed
- Other dependencies

### 2. Start Qdrant Vector Database
```bash
docker-compose up -d
```

Verify it's running:
```bash
curl http://localhost:6333/health
```

### 3. Load Knowledge Base into Qdrant
```bash
python vector_setup/load_qdrant.py
```

This will:
- Read knowledge_base.txt
- Chunk the content
- Generate embeddings
- Store in Qdrant collection "support_kb"

### 4. Verify RAG Setup
```bash
python vector_setup/check_qdrant.py
```

You should see:
- Collection info (number of points, vector size)
- Sample search results

### 5. Set OpenAI API Key
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-key-here"
```

### 6. Test the Refactored Bot
```bash
python test_refactored.py
```

This will test:
- RAG connection
- Semantic search
- Full bot integration

### 7. Run the Application
```bash
streamlit run app.py
```

## Usage Examples

### Direct RAG Search
```python
from rag_search import search_kb_rag

results = search_kb_rag("How do I return a product?", max_results=3)
for doc in results['results']:
    print(f"{doc['title']}: {doc['snippet']}")
    print(f"Relevance: {doc['relevance_score']}")
```

### Using the Support Bot
```python
from support_bot import CustomerSupportBot, SupportSession

bot = CustomerSupportBot()
session = SupportSession()

# The bot will automatically use the LLM router and RAG search
response = bot.reply(session, "What's your return policy?")
print(response)
```

### Check RAG Status
```python
from rag_search import check_rag_connection

status = check_rag_connection()
if status['connected']:
    print(f"Connected! {status['total_documents']} documents available")
else:
    print(f"Error: {status['error']}")
```

## Understanding the New Architecture

### LLM Router
The bot now has an explicit `route_with_llm()` function that:
1. Analyzes the user's message
2. Looks at conversation history
3. Decides which tools (if any) to call
4. Logs the routing decision

### Tool Executor
The `execute_tool_call()` function:
1. Parses tool arguments
2. Executes the tool function
3. Handles errors gracefully
4. Logs execution time and results
5. Returns structured results

### RAG Search
The `search_kb()` tool now uses `rag_search.py` which:
1. Embeds the query using fastembed
2. Performs semantic search in Qdrant
3. Returns relevant chunks with scores
4. Provides full text for LLM context

## Troubleshooting

### "Cannot connect to Qdrant"
- Check if Qdrant is running: `docker ps`
- Restart: `docker-compose restart`
- Check logs: `docker-compose logs qdrant`

### "Collection 'support_kb' not found"
- Run: `python vector_setup/load_qdrant.py`

### "No search results"
- Verify collection has data: `python vector_setup/check_qdrant.py`
- Check if knowledge_base.txt exists and has content

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

## Development

### Adding a New Tool
1. Add the function to `support_bot.py`
2. Add to `TOOL_FUNCS` dictionary
3. Add schema to `TOOLS_SCHEMA` list
4. The router will automatically make it available

### Modifying Routing Logic
Edit the `route_with_llm()` function in `support_bot.py`

### Modifying Tool Execution
Edit the `execute_tool_call()` function in `support_bot.py`

### Changing RAG Behavior
Edit functions in `rag_search.py`:
- Change embedding model
- Modify search parameters
- Add filters or re-ranking

## File Structure

```
cust-support-agent/
├── support_bot.py          # Main bot (refactored)
├── rag_search.py           # RAG/vector search (NEW)
├── sample_data.py          # Sample data (products, orders, etc.)
├── app.py                  # Streamlit UI
├── test_refactored.py      # Test suite (NEW)
├── requirements.txt        # Dependencies (updated)
├── REFACTORING_NOTES.md    # Detailed refactoring notes (NEW)
├── ARCHITECTURE.md         # Architecture diagram (NEW)
├── QUICK_START.md          # This file (NEW)
├── knowledge_base.txt      # Knowledge base content
└── vector_setup/
    ├── load_qdrant.py      # Load KB into Qdrant
    ├── check_qdrant.py     # Verify Qdrant data
    └── chunker.py          # Text chunking logic
```

## Next Steps

1. **Customize Knowledge Base**: Edit `knowledge_base.txt` and reload
2. **Add More Tools**: Extend `support_bot.py` with new capabilities
3. **Tune RAG**: Experiment with different embedding models or search parameters
4. **Monitor Performance**: Check logs for tool execution times
5. **A/B Test Routing**: Create alternative routing strategies

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed Models](https://qdrant.github.io/fastembed/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

## Support

For questions about the refactored architecture, see:
- `REFACTORING_NOTES.md` - Detailed change log
- `ARCHITECTURE.md` - Visual architecture diagrams
- `test_refactored.py` - Working examples

