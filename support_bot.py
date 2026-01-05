# support_bot.py
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import math
import re
import logging
import time

from openai import OpenAI

from sample_data import PRODUCTS, ORDERS, RETURN_POLICY, SUPPORT_GUIDANCE, SHIPPING_RATES, PROMO_CODES, KB_ARTICLES, WAREHOUSES, INVENTORY_BY_WAREHOUSE
from rag_search import search_kb_rag
from tools import TOOL_FUNCS, TOOLS_SCHEMA

DB_PATH = "tickets.sqlite3"

logger = logging.getLogger("support_bot")
logger.setLevel(logging.INFO)

# Console handler (works fine for Streamlit logs too)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            sku TEXT NOT NULL,
            reason TEXT NOT NULL,
            email TEXT,
            status TEXT NOT NULL DEFAULT 'requested'
        )
        """
    )
    conn.commit()
    conn.close()

# ----------------------------
# Tool Execution & Routing
# ----------------------------

def execute_tool_call(tool_call, tool_funcs: Dict[str, Any]) -> tuple[str, Dict[str, Any], int]:
    """
    Execute a single tool call and return the result with timing information.
    
    Args:
        tool_call: The tool call object from OpenAI API
        tool_funcs: Dictionary mapping tool names to their functions
    
    Returns:
        Tuple of (tool_name, result_dict, elapsed_ms)
    """
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        args = {}
    
    fn = tool_funcs.get(name)
    
    start = time.time()
    logger.info("TOOL CALL -> %s args=%s", name, args)
    
    if not fn:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = fn(**args)
        except TypeError as e:
            result = {"error": f"Bad arguments for {name}: {str(e)}"}
        except Exception as e:
            result = {"error": f"Tool {name} failed: {str(e)}"}
    
    elapsed_ms = int((time.time() - start) * 1000)
    preview = str(result)
    if len(preview) > 500:
        preview = preview[:500] + "...(truncated)"
    logger.info("TOOL RESULT <- %s (%d ms) result=%s", name, elapsed_ms, preview)
    
    return name, result, elapsed_ms


def route_with_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools_schema: List[Dict[str, Any]]
) -> Any:
    """
    LLM Router: Uses the language model to decide which tool(s) to call.
    
    This function encapsulates the routing logic where the LLM analyzes
    the conversation and determines which tools are needed to respond.
    
    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: Conversation history
        tools_schema: Available tools schema
    
    Returns:
        The API response containing the LLM's decision on tool usage
    """
    logger.info("LLM Router: Analyzing request and determining tool usage...")
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
    )
    
    message = resp.choices[0].message
    
    if message.tool_calls:
        tool_names = [tc.function.name for tc in message.tool_calls]
        logger.info(f"LLM Router: Selected tools: {tool_names}")
    else:
        logger.info("LLM Router: No tools needed, generating direct response")
    
    return resp


# ----------------------------
# Agent core
# ----------------------------

@dataclass
class SupportSession:
    conversation_id: Optional[str] = None


SYSTEM_INSTRUCTIONS = """
You are a customer support agent for a small e-commerce store.

Goals:
- Answer questions about products, stock, pricing, features, warranty, and returns.
- Help customers check order status.
- Use internal support guidance for troubleshooting steps and escalation rules.
- If the issue requires human intervention, create a support ticket.

Rules:
- Do not invent order status or product details. Use tools when you need facts.
- Ask a brief clarifying question if required info is missing (e.g., order id, sku).
- Keep responses concise and helpful.
"""


class CustomerSupportBot:
    def __init__(self, model: str = "gpt-4o"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY env var")
        if not os.getenv("QDRANT_API_KEY"):
            raise RuntimeError("Missing QDRANT_API_KEY env var")
        if not os.getenv("QDRANT_URL"):
            raise RuntimeError("Missing QDRANT_URL env var")
        self.client = OpenAI()
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        init_db()

    def reply(self, session: SupportSession, user_text: str) -> str:
        """
        Single turn with tool-loop, maintaining a stateful conversation via messages history.
        """
        # Initialize messages if this is a new session
        if not self.messages:
            self.messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        
        # Add user message
        self.messages.append({"role": "user", "content": user_text})
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Use LLM router to determine which tools to call (if any)
            resp = route_with_llm(
                client=self.client,
                model=self.model,
                messages=self.messages,
                tools_schema=TOOLS_SCHEMA
            )

            message = resp.choices[0].message
            
            # Add assistant message to history
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls if message.tool_calls else None
            })
            
            # Check if there are tool calls
            if not message.tool_calls:
                return message.content or ""

            # Process each tool call using the executor
            for tool_call in message.tool_calls:
                name, result, elapsed_ms = execute_tool_call(tool_call, TOOL_FUNCS)
                
                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
        
        return "I apologize, but I've reached the maximum number of processing steps. Please try rephrasing your question."
