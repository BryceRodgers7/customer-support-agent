# support_bot.py
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from openai import OpenAI

from sample_data import PRODUCTS, ORDERS, RETURN_POLICY, SUPPORT_GUIDANCE

DB_PATH = "tickets.sqlite3"


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
    conn.commit()
    conn.close()


# ----------------------------
# Tools
# ----------------------------

def search_products(query: str) -> Dict[str, Any]:
    q = query.lower().strip()
    matches = []
    for p in PRODUCTS:
        hay = f"{p['sku']} {p['name']} {' '.join(p.get('key_features', []))}".lower()
        if q in hay:
            matches.append(p)

    if not matches:
        tokens = [t for t in q.split() if t]
        for p in PRODUCTS:
            hay = f"{p['sku']} {p['name']} {' '.join(p.get('key_features', []))}".lower()
            if any(t in hay for t in tokens):
                matches.append(p)

    return {"matches": matches[:5]}


def get_product_details(sku: str) -> Dict[str, Any]:
    for p in PRODUCTS:
        if p["sku"].upper() == sku.upper():
            return {"product": p}
    return {"error": f"Unknown SKU: {sku}"}


def get_order_status(order_id: str) -> Dict[str, Any]:
    order = ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"Order not found: {order_id}"}
    return {"order_id": order_id.upper(), **order}


def get_return_policy(product_sku: Optional[str] = None) -> Dict[str, Any]:
    policy = dict(RETURN_POLICY)
    if product_sku:
        policy["note"] = f"Policy shown is general; SKU={product_sku} is covered unless marked final sale."
    return policy


def get_support_guidance(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Tool to retrieve internal guidance. This lets you grow into
    more complex support playbooks without hardcoding everything in the prompt.
    """
    if not topic:
        return {"guidance": SUPPORT_GUIDANCE}
    # simple path-ish lookup
    node: Any = SUPPORT_GUIDANCE
    for part in topic.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {"error": f"Unknown guidance topic path: {topic}"}
    return {"guidance": node}


def create_support_ticket(subject: str, description: str, email: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tickets (email, subject, description) VALUES (?, ?, ?)",
        (email, subject, description),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return {"ticket_id": ticket_id, "status": "open"}


TOOL_FUNCS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_order_status": get_order_status,
    "get_return_policy": get_return_policy,
    "get_support_guidance": get_support_guidance,
    "create_support_ticket": create_support_ticket,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by keyword (name, sku, features).",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Fetch full product details by SKU.",
            "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order status by order_id.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_return_policy",
            "description": "Get the return policy. Optionally pass product_sku for sku-specific notes.",
            "parameters": {"type": "object", "properties": {"product_sku": {"type": ["string", "null"]}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_support_guidance",
            "description": "Fetch internal support playbooks / guidance. Optional dot-path topic (e.g., 'troubleshooting.camera_autofocus').",
            "parameters": {"type": "object", "properties": {"topic": {"type": ["string", "null"]}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket when an issue needs human follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "email": {"type": ["string", "null"]},
                },
                "required": ["subject", "description"],
            },
        },
    },
]


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
            
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
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

            # Process tool calls
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                fn = TOOL_FUNCS.get(name)

                if not fn:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except TypeError as e:
                        result = {"error": f"Bad arguments for {name}: {str(e)}"}
                    except Exception as e:
                        result = {"error": f"Tool {name} failed: {str(e)}"}

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
        
        return "I apologize, but I've reached the maximum number of processing steps. Please try rephrasing your question."
