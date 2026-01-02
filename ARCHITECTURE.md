# Architecture Diagram: Refactored Support Bot

## Before Refactoring

```
reply() function
├── Get user message
├── Call OpenAI API (routing mixed with execution)
├── If tool calls:
│   ├── Parse tool name and args
│   ├── Look up function
│   ├── Execute function
│   ├── Log (mixed throughout)
│   └── Handle errors
└── Return response

search_kb()
└── Simple keyword matching from dictionary
```

## After Refactoring

```
reply() function (Orchestrator)
├── Get user message
├── route_with_llm()  ← EXPLICIT ROUTING STEP
│   ├── Analyze conversation
│   ├── Decide which tools (if any)
│   └── Log routing decision
│
├── For each tool call:
│   └── execute_tool_call()  ← CLEAN EXECUTION
│       ├── Parse args
│       ├── Execute function
│       ├── Log call & result
│       ├── Handle errors
│       └── Return result
│
└── Return response


search_kb()  ← NOW USES RAG
└── search_kb_rag()  (in rag_search.py)
    ├── Initialize Qdrant & embedder
    ├── Embed query
    ├── Vector search
    └── Return semantic results with scores
```

## Data Flow

```
User Input
    ↓
reply() orchestrates the conversation flow
    ↓
route_with_llm() - LLM analyzes and decides
    ↓
[If tools needed]
    ↓
execute_tool_call() - Executes each tool
    ↓
[Tool function, e.g., search_kb()]
    ↓
[If search_kb: uses RAG]
    ↓
search_kb_rag() in rag_search.py
    ↓
Qdrant Vector Database
    ↓
Returns semantic results
    ↓
Back to reply()
    ↓
LLM uses tool results to generate response
    ↓
User receives response
```

## Key Components

### 1. LLM Router (`route_with_llm`)
**Purpose**: Decision-making layer
**Input**: Conversation history, available tools
**Output**: Which tools to call (if any)
**Benefits**: 
- Explicit routing logic
- Easy to add routing strategies
- Clear logging of decisions

### 2. Tool Executor (`execute_tool_call`)
**Purpose**: Execution layer
**Input**: Tool call specification
**Output**: Tool result + timing
**Benefits**:
- Reusable execution logic
- Centralized error handling
- Consistent logging
- Can add features like retry, caching

### 3. RAG Search (`rag_search.py`)
**Purpose**: Knowledge base search
**Components**:
- `search_kb_rag()`: Main search function
- `check_rag_connection()`: Health check
- `get_kb_document()`: Direct retrieval

**Benefits**:
- Semantic search (understands meaning)
- Scales to large knowledge bases
- Relevance scoring
- Isolated from main bot logic

## Separation of Concerns

| Component | Responsibility | Can Change Without Affecting |
|-----------|---------------|------------------------------|
| `reply()` | Conversation flow | Tool execution, routing strategy |
| `route_with_llm()` | Tool selection | Tool implementation, execution |
| `execute_tool_call()` | Tool execution | Routing logic, tool implementation |
| `rag_search.py` | Vector search | Bot logic, other tools |
| Individual tools | Domain logic | Everything else |

## Extension Points

Want to add...?

**Retry logic**: Modify `execute_tool_call()`
**Different routing strategy**: Modify `route_with_llm()`
**New tool**: Add to `TOOL_FUNCS` and `TOOLS_SCHEMA`
**Different embedding model**: Modify `rag_search.py`
**Tool caching**: Add to `execute_tool_call()`
**A/B test routing**: Wrap `route_with_llm()`

