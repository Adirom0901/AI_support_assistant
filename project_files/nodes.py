import os
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from model import generate_response
from state import ChatState
from tools import tools, WRITE_TOOLS, PASSTHROUGH_TOOLS, CONFIRM

# --- deterministic routing (a small model can't be trusted to pick tools by itself) ---
FORCE_ROUTING = os.getenv("FORCE_LOOKUP", "true").lower() == "true"
MAX_TOOL_ROUNDS = 2

ORDER_ID_RE = re.compile(r"\bORD\d+\b", re.I)
YES_RE = re.compile(r"^\s*(yes|y|yeah|yep|sure|ok|okay|confirm|go ahead|proceed)\b", re.I)
BARE_YES_RE = re.compile(r"^\s*(yes|y|yeah|yep|confirm|go ahead|proceed)\W*$", re.I)
NO_RE = re.compile(r"^\s*(no|n|nope|stop|never ?mind|don'?t)\b", re.I)

CANCEL_RE = re.compile(r"\bcancel", re.I)
RETURN_RE = re.compile(r"\b(return|refund|replace|exchange)", re.I)
TRACK_RE = re.compile(r"\b(track|tracking|where is|where's|shipment)", re.I)
HANDOFF_RE = re.compile(
    r"\b(human|agent|representative|complain|complaint|escalat|speak to|talk to|manager)", re.I)
POLICY_RE = re.compile(
    r"\b(policy|policies|return|refund|exchange|shipping|delivery|warranty|cancel|charges?|fee|how long|how many days)",
    re.I)


def _ai_call(name: str, **args):
    call = {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}
    return {"messages": [AIMessage(content="", tool_calls=[call])]}


def _route(text: str):
    """Map a user message to (tool, args), or None to let the model handle it."""
    m = ORDER_ID_RE.search(text)
    oid = m.group(0).upper() if m else None

    if HANDOFF_RE.search(text):
        args = {"issue": text[:500]}
        if oid:
            args["order_id"] = oid
        return "create_support_ticket", args
    if oid:
        if CANCEL_RE.search(text):
            return "cancel_order", {"order_id": oid}
        if RETURN_RE.search(text):
            return "initiate_return", {"order_id": oid, "reason": text[:300]}
        if TRACK_RE.search(text):
            return "track_shipment", {"order_id": oid}
        return "lookup_order", {"order_id": oid}
    if POLICY_RE.search(text):
        return "search_policy", {"query": text}
    return None


def _pending_confirmation(messages):
    """If the bot just asked 'reply yes to confirm', return that original tool call."""
    for m in reversed(messages[:-1]):          # skip the current user message
        if isinstance(m, HumanMessage):
            return None                        # a newer user turn happened in between
        if isinstance(m, ToolMessage):
            if m.name in WRITE_TOOLS and str(m.content).startswith(CONFIRM):
                for ai in reversed(messages):
                    if isinstance(ai, AIMessage):
                        for tc in ai.tool_calls:
                            if tc["id"] == m.tool_call_id:
                                return tc
            return None
    return None


def _tool_rounds_since_last_human(messages) -> int:
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            n += 1
    return n


def generate_node(state: ChatState):
    messages = state["messages"]
    last = messages[-1]

    # 1) Customer-ready tool output (confirmations, results, tickets): return as-is,
    #    so the LLM can never rewrite or botch a write-action message.
    if isinstance(last, ToolMessage) and last.name in PASSTHROUGH_TOOLS:
        return {"messages": [AIMessage(content=str(last.content).removeprefix(CONFIRM).strip())]}

    # 2) Deterministic routing on a fresh user message
    if FORCE_ROUTING and isinstance(last, HumanMessage):
        pending = _pending_confirmation(messages)
        if pending:
            if YES_RE.match(last.content):
                # Only this path ever sets confirmed=True, never the model.
                return _ai_call(pending["name"], **{**pending["args"], "confirmed": True})
            if NO_RE.match(last.content):
                return {"messages": [AIMessage(
                    content="No problem, I haven't made any changes. Anything else I can help with?")]}
        routed = _route(last.content)
        if routed:
            return _ai_call(routed[0], **routed[1])
        if BARE_YES_RE.match(last.content):
            # "yes" with nothing pending: never let the model improvise an action
            return {"messages": [AIMessage(content=(
                "I don't have any action waiting for your confirmation. "
                "Tell me what you'd like to do, for example 'cancel ORD1006'."))]}

    # 3) Normal generation; the model may still call tools itself
    allow_tools = _tool_rounds_since_last_human(messages) < MAX_TOOL_ROUNDS
    text, calls = generate_response(messages, tools if allow_tools else None)
    for c in calls:                            # the model may never self-confirm
        if c["name"] in WRITE_TOOLS:
            c["args"]["confirmed"] = False

    if not text and not calls:
        text = "Sorry, I couldn't process that. Could you rephrase or share your order ID?"
    return {"messages": [AIMessage(content=text, tool_calls=calls)]}