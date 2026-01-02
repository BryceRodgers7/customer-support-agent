# Refactoring Summary: Support Bot with RAG and Improved Architecture

## Overview
The support bot has been refactored to use a vector database (RAG) for knowledge base search and to have a clearer separation of concerns with dedicated routing and tool execution functions.

## Changes Made

### 1. New File: `rag_search.py`
This file implements RAG (Retrieval Augmented Generation) using Qdrant vector database:

**Key Functions:**
- `search_kb_rag(query, max_results)`: Semantic search using vector embeddings
- `get_kb_document(doc_id)`: Retrieve a specific document by ID
- `check_rag_connection()`: Verify Qdrant connection and status

**Features:**
- Uses `fastembed` with `BAAI/bge-small-en-v1.5` model for embeddings
- Lazy initialization of Qdrant client
- Comprehensive error handling and logging
- Returns results with relevance scores and full text for LLM context

### 2. Updated: `support_bot.py`

#### a) Knowledge Base Search
- **Old**: `search_kb()` used simple keyword matching from `sample_data.py` dictionary
- **New**: `search_kb()` now calls `search_kb_rag()` for semantic search

#### b) LLM Router Function (NEW)
Added `route_with_llm()` function that encapsulates the routing logic:
```python
def route_with_llm(client, model, messages, tools_schema) -> Any:
    """
    LLM Router: Uses the language model to decide which tool(s) to call.
    """
```

**Purpose**: 
- Clearly separates the "decision making" step where the LLM analyzes the conversation
- Determines which tools are needed to respond to the user
- Logs routing decisions for debugging

#### c) Tool Execution Function (NEW)
Added `execute_tool_call()` function that handles individual tool execution:
```python
def execute_tool_call(tool_call, tool_funcs) -> tuple[str, Dict[str, Any], int]:
    """
    Execute a single tool call and return the result with timing information.
    """
```

**Features**:
- Parses tool arguments
- Handles errors gracefully
- Logs tool calls and results
- Times execution
- Returns structured results

#### d) Simplified `reply()` Function
The main `reply()` function is now cleaner:
- Calls `route_with_llm()` to decide what tools to use
- Calls `execute_tool_call()` for each tool
- Focuses on conversation flow rather than implementation details

### 3. Updated: `requirements.txt`
Added new dependencies:
- `qdrant-client[fastembed]`: Vector database client with embedding support
- `fastembed`: Fast embedding generation

### 4. New File: `test_refactored.py`
Comprehensive test script that verifies:
- RAG connection to Qdrant
- Direct RAG search functionality
- Full bot integration with new architecture

## Architecture Benefits

### 1. Separation of Concerns
- **Routing Logic**: Now in `route_with_llm()` - LLM decides what to do
- **Tool Execution**: Now in `execute_tool_call()` - Handles the "doing"
- **Conversation Flow**: In `reply()` - Orchestrates the conversation

### 2. Better RAG Integration
- Semantic search instead of keyword matching
- More relevant results using vector similarity
- Scales better with large knowledge bases

### 3. Improved Testability
- Each component can be tested independently
- Easier to debug routing vs execution issues
- Clear logging at each step

### 4. Maintainability
- Functions have single, clear responsibilities
- Easier to modify routing logic without touching tool execution
- Easier to add new tools or change how they're executed

## How the LLM Router Works

The refactored code explicitly separates the "routing" decision:

1. **User sends message** → Added to conversation history
2. **`route_with_llm()` is called** → LLM analyzes the conversation and available tools
3. **LLM decides**:
   - No tools needed → Returns direct response
   - Tools needed → Returns list of tool calls
4. **`execute_tool_call()` runs each tool** → Gets results
5. **Results added to conversation** → Loop continues if needed

This makes it clear that the LLM is acting as a "router" that decides which tools to invoke based on the conversation context.

## Migration Guide

### For Existing Code
The API remains the same:
```python
bot = CustomerSupportBot()
session = SupportSession()
response = bot.reply(session, "user message")
```

### For New Installations
1. Ensure Qdrant is running: `docker-compose up -d`
2. Load the knowledge base: `python vector_setup/load_qdrant.py`
3. Install new dependencies: `pip install -r requirements.txt`
4. Test: `python test_refactored.py`

### For Development
- To modify routing logic: Edit `route_with_llm()`
- To modify tool execution: Edit `execute_tool_call()`
- To add new tools: Add to `TOOL_FUNCS` and `TOOLS_SCHEMA`
- To modify RAG behavior: Edit `rag_search.py`

## Testing

Run the test suite:
```bash
python test_refactored.py
```

This will:
1. Check RAG connection
2. Test semantic search
3. Run end-to-end bot tests

## Notes

- The old `search_kb()` interface is preserved but now uses RAG internally
- All logging statements are now in dedicated functions for easier debugging
- The LLM router pattern makes it explicit that tool selection is an AI decision
- Tool execution is now reusable and easier to extend with retries, caching, etc.

