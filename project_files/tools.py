from langchain_core.tools import tool

import database as db
from rag import search_policies

CONFIRM = "[CONFIRM] "                                   # prefix on "are you sure?" results
WRITE_TOOLS = {"cancel_order", "initiate_return"}        # need explicit user confirmation
# Tools whose output is already customer-ready; nodes.py returns it as-is (no LLM rewrite)
PASSTHROUGH_TOOLS = WRITE_TOOLS | {"create_support_ticket", "track_shipment"}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its exact order ID.
    Use this when a customer asks about an order's
    status, delivery date, product, or tracking number.
    """
    order = db.get_order(order_id)
    if order is None:
        return {"found": False, "order_id": order_id, "message": "Order not found"}
    return {"found": True, "order": order}


_TRACK_TEXT = {
    "Processing": "has not shipped yet, so there is no tracking number. It is expected by {eta}.",
    "Shipped": "is in transit with tracking number {trk}. Expected delivery: {eta}.",
    "Out for Delivery": "is out for delivery today (tracking {trk}).",
    "Delivered": "was delivered (tracking {trk}).",
    "Cancelled": "was cancelled and will not be shipped.",
}


@tool
def track_shipment(order_id: str) -> str:
    """Get the shipping/tracking status of an order. Use when the customer asks
    where their order is or for a tracking number."""
    o = db.get_order(order_id)
    if o is None:
        return f"I couldn't find order {order_id}."
    # A real courier API call would go here, keyed on o["tracking_number"].
    tmpl = _TRACK_TEXT.get(o["order_status"], "is currently '{status}'.")
    return f"Your order {order_id} ({o['product_name']}) " + tmpl.format(
        eta=o["expected_delivery"], trk=o["tracking_number"], status=o["order_status"]
    )


@tool
def search_policy(query: str) -> str:
    """Search the store's policy documents: returns, refunds, cancellation,
    shipping charges and times, warranty. Use for any general 'what is your policy /
    how does it work' question that does not need a specific order."""
    hits = search_policies(query)
    if not hits:
        return ("No matching policy found. Tell the customer you don't have that information "
                "and offer to create a support ticket. Do not guess.")
    return "\n\n".join(f"[{h['source']} / {h['heading']}]\n{h['text']}" for h in hits)


@tool
def cancel_order(order_id: str, confirmed: bool = False) -> str:
    """Cancel an order that has not shipped yet. Use when the customer asks to cancel
    and gives an order ID. Always call with confirmed=False."""
    order = db.get_order(order_id)
    if order is None:
        return f"I couldn't find order {order_id}."
    ok, why = db.cancel_eligibility(order)
    if not ok:
        return why
    if not confirmed:
        return CONFIRM + (f"You're about to cancel {order_id} ({order['product_name']}). "
                          "Reply 'yes' to confirm or 'no' to keep the order.")
    if db.cancel_order_record(order_id):
        return (f"Done. Order {order_id} ({order['product_name']}) has been cancelled. "
                "Your refund will arrive in 5 to 7 business days.")
    return f"Sorry, {order_id} could no longer be cancelled."


@tool
def initiate_return(order_id: str, reason: str = "", confirmed: bool = False) -> str:
    """Start a return for a delivered order. Use when the customer wants to return,
    replace or get a refund for an item and gives an order ID. Always call with confirmed=False."""
    ok, why, order = db.return_eligibility(order_id)
    if not ok:
        return why
    if not confirmed:
        return CONFIRM + (f"I can start a return for {order_id} ({order['product_name']}). "
                          "Reply 'yes' to confirm or 'no' to cancel.")
    return_id = db.create_return(order_id, reason or "Not specified")
    if return_id is None:
        return f"Sorry, I couldn't start a return for {order_id}."
    return (f"Your return request {return_id} has been created for {order['product_name']}. "
            "Pickup will be arranged within 2 business days and your refund follows "
            "5 to 7 business days after we receive the item.")


@tool
def create_support_ticket(issue: str, order_id: str = "") -> str:
    """Escalate to a human agent by creating a support ticket. Use for complaints,
    requests to talk to a human, warranty claims, or anything you cannot resolve."""
    ticket_id = db.create_ticket(issue, order_id.upper() or None)
    return (f"I've created support ticket {ticket_id} and passed it to our team. "
            "An agent will get back to you shortly.")


tools = [lookup_order, track_shipment, search_policy, cancel_order, initiate_return, create_support_ticket]
tools_by_name = {t.name: t for t in tools}