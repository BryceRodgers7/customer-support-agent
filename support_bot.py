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

def estimate_shipping(zip_code: str, item_count: int = 1, method: str = "Standard") -> Dict[str, Any]:
    method_norm = method.strip().lower()
    rate = next((r for r in SHIPPING_RATES if r["method"].lower() == method_norm), None)
    if not rate:
        return {"error": f"Unknown shipping method: {method}. Options: {[r['method'] for r in SHIPPING_RATES]}"}

    # very simple example: base + per-item bump
    per_item = 1.25
    cost = rate["base_usd"] + max(0, item_count - 1) * per_item

    return {
        "zip_code": zip_code,
        "method": rate["method"],
        "estimated_days": {"min": rate["min_days"], "max": rate["max_days"]},
        "estimated_cost_usd": round(cost, 2),
    }

def validate_promo_code(code: str) -> Dict[str, Any]:
    key = code.strip().upper()
    promo = PROMO_CODES.get(key)
    if not promo:
        return {"valid": False, "reason": "Unknown code"}
    if not promo.get("active", False):
        return {"valid": False, "reason": "Code is not active"}
    return {"valid": True, "code": key, "details": promo}

def recommend_products(need: str, max_results: int = 3) -> Dict[str, Any]:
    q = need.lower().strip()
    tokens = [t for t in re.split(r"\W+", q) if t]

    scored = []
    for p in PRODUCTS:
        hay = f"{p['sku']} {p['name']} {' '.join(p.get('key_features', []))} {p.get('category','')}".lower()
        score = sum(1 for t in tokens if t and t in hay)
        # small boost if in stock
        if p.get("stock", 0) > 0:
            score += 1
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [p for s, p in scored if s > 0][: max(1, max_results)]
    return {"recommendations": picks}

def check_inventory(sku: str, include_by_warehouse: bool = False) -> Dict[str, Any]:
    sku_u = sku.upper()
    prod = next((p for p in PRODUCTS if p["sku"].upper() == sku_u), None)
    if not prod:
        return {"error": f"Unknown SKU: {sku}"}

    out: Dict[str, Any] = {"sku": sku_u, "stock_total": prod.get("stock", 0)}
    if include_by_warehouse:
        by_wh = {}
        for wh in WAREHOUSES:
            wh_id = wh["id"]
            by_wh[wh_id] = {
                "location": f"{wh['city']}, {wh['state']}",
                "qty": INVENTORY_BY_WAREHOUSE.get(wh_id, {}).get(sku_u, 0),
            }
        out["by_warehouse"] = by_wh
    return out

def get_restock_eta(sku: str) -> Dict[str, Any]:
    sku_u = sku.upper()
    prod = next((p for p in PRODUCTS if p["sku"].upper() == sku_u), None)
    if not prod:
        return {"error": f"Unknown SKU: {sku}"}

    if prod.get("stock", 0) > 0:
        return {"sku": sku_u, "restock_needed": False, "message": "Item is currently in stock."}

    # demo: different categories have different lead times
    category = prod.get("category", "").lower()
    eta_days = 14 if category in {"peripherals"} else 7
    return {"sku": sku_u, "restock_needed": True, "estimated_restock_days": eta_days}

def create_return_request(order_id: str, sku: str, reason: str, email: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    # Create table if missing
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO returns (order_id, sku, reason, email) VALUES (?, ?, ?, ?)",
        (order_id.upper(), sku.upper(), reason, email),
    )
    conn.commit()
    rma_id = cur.lastrowid
    conn.close()

    return {"rma_id": rma_id, "status": "requested", "order_id": order_id.upper(), "sku": sku.upper()}

def search_kb(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search the knowledge base using RAG (vector database).
    This now uses semantic search via Qdrant instead of simple keyword matching.
    """
    return search_kb_rag(query, max_results)

def get_troubleshooting_steps(issue: str) -> Dict[str, Any]:
    issue_norm = issue.strip().lower()

    # Map a few common issues to internal guidance paths
    mapping = {
        "autofocus": "troubleshooting.camera_autofocus",
        "camera_autofocus": "troubleshooting.camera_autofocus",
    }
    path = mapping.get(issue_norm)

    if not path:
        # fall back to the KB search for unknowns
        return {"note": "No direct playbook match. Returning relevant KB results instead.", "kb": search_kb(issue_norm)}

    # Walk SUPPORT_GUIDANCE dict (same logic as get_support_guidance)
    node: Any = SUPPORT_GUIDANCE
    for part in path.split("."):
        node = node.get(part, None) if isinstance(node, dict) else None
        if node is None:
            return {"error": f"Guidance path missing: {path}"}

    return {"issue": issue_norm, "steps": node}

def classify_request(user_message: str) -> Dict[str, Any]:
    text = user_message.lower()
    labels = []
    if any(k in text for k in ["refund", "return", "rma", "exchange"]):
        labels.append("returns")
    if any(k in text for k in ["where is my order", "tracking", "shipped", "delivered", "order"]):
        labels.append("order_status")
    if any(k in text for k in ["broken", "won't", "doesn't", "issue", "problem", "troubleshoot"]):
        labels.append("troubleshooting")
    if any(k in text for k in ["price", "in stock", "feature", "compare", "recommend"]):
        labels.append("product_help")

    if not labels:
        labels = ["general_support"]

    # naive confidence
    confidence = min(0.95, 0.55 + 0.1 * len(labels))
    return {"labels": labels, "confidence": confidence}

def should_escalate(message: str) -> Dict[str, Any]:
    text = message.lower()
    triggers = ["chargeback", "lawyer", "legal", "injury", "fraud", "scam"]
    matched = [t for t in triggers if t in text]

    # Additionally incorporate your SUPPORT_GUIDANCE escalation rules
    # (for demo we keep it simple)
    return {
        "escalate": len(matched) > 0,
        "matched_triggers": matched,
        "note": "Escalate if sensitive/legal/billing triggers are present.",
    }

TOOL_FUNCS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_order_status": get_order_status,
    "get_return_policy": get_return_policy,
    "get_support_guidance": get_support_guidance,
    "create_support_ticket": create_support_ticket,
    "estimate_shipping": estimate_shipping,
    "validate_promo_code": validate_promo_code,
    "recommend_products": recommend_products,
    "check_inventory": check_inventory,
    "get_restock_eta": get_restock_eta,
    "create_return_request": create_return_request,
    "search_kb": search_kb,
    "get_troubleshooting_steps": get_troubleshooting_steps,
    "classify_request": classify_request,
    "should_escalate": should_escalate,
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
    {
        "type": "function",
        "function": {
            "name": "estimate_shipping",
            "description": "Estimate shipping cost and delivery window for a zip code, item count, and method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_code": {"type": "string"},
                    "item_count": {"type": "integer", "minimum": 1},
                    "method": {"type": "string", "enum": ["Standard", "Expedited", "Overnight"]},
                },
                "required": ["zip_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_promo_code",
            "description": "Validate a promo code and return discount details if valid.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "Recommend products based on a short need statement (e.g., 'wireless headphones for travel').",
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check stock for a SKU. Optionally include by-warehouse quantities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "include_by_warehouse": {"type": "boolean"},
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restock_eta",
            "description": "If a SKU is out of stock, return an estimated restock timeline (demo).",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return_request",
            "description": "Create a return request (RMA) for an order and SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "sku": {"type": "string"},
                    "reason": {"type": "string"},
                    "email": {"type": ["string", "null"]},
                },
                "required": ["order_id", "sku", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search internal knowledge base using semantic search (RAG). Returns relevant documentation chunks from the company's support knowledge base with relevance scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query - can be a question, topic, or keywords"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum number of results to return"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_troubleshooting_steps",
            "description": "Get troubleshooting steps for a known issue (falls back to KB search if unknown).",
            "parameters": {
                "type": "object",
                "properties": {"issue": {"type": "string"}},
                "required": ["issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_request",
            "description": "Classify a user message into request labels like returns, order_status, troubleshooting, product_help.",
            "parameters": {
                "type": "object",
                "properties": {"user_message": {"type": "string"}},
                "required": ["user_message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "should_escalate",
            "description": "Check if a message should be escalated to a human (legal/billing/safety triggers).",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
]


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
