import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "orders.db"
RETURN_WINDOW_DAYS = 7  # keep in sync with policies/returns.md


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_order(order_id: str):
    with _db() as conn:
        row = conn.execute(
            """
            SELECT order_id, customer_name, product_name,
                   order_status, order_date,
                   expected_delivery, tracking_number
            FROM orders WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
        return dict(row) if row else None




def cancel_eligibility(order: dict):
    """Returns (ok, message)."""
    status = order["order_status"]
    if status == "Processing":
        return True, ""
    if status == "Cancelled":
        return False, f"Order {order['order_id']} is already cancelled."
    return False, (f"Order {order['order_id']} is '{status}', so it can no longer be cancelled. "
                   "Once it arrives you can return it within 7 days of delivery.")


def cancel_order_record(order_id: str) -> bool:
    """Atomic: only flips the status if it is still Processing."""
    with _db() as conn:
        cur = conn.execute(
            "UPDATE orders SET order_status = 'Cancelled' "
            "WHERE order_id = ? AND order_status = 'Processing'",
            (order_id,),
        )
        return cur.rowcount == 1




def return_eligibility(order_id: str):
    """Returns (ok, message, order). Uses expected_delivery as the delivery date
    (add a delivered_at column if you want this to be exact)."""
    order = get_order(order_id)
    if order is None:
        return False, f"I couldn't find order {order_id}.", None

    status = order["order_status"]
    if status != "Delivered":
        return False, (f"Returns can only be started for delivered orders. "
                       f"Order {order_id} is currently '{status}'."), order

    days = (date.today() - date.fromisoformat(order["expected_delivery"])).days
    if days > RETURN_WINDOW_DAYS:
        return False, (f"The {RETURN_WINDOW_DAYS}-day return window for {order_id} "
                       f"ended {days - RETURN_WINDOW_DAYS} day(s) ago."), order

    with _db() as conn:
        existing = conn.execute("SELECT id FROM returns WHERE order_id = ?", (order_id,)).fetchone()
    if existing:
        return False, f"A return has already been requested for {order_id}.", order

    return True, "", order


def create_return(order_id: str, reason: str) -> str | None:
    ok, _, _ = return_eligibility(order_id)
    if not ok:
        return None
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO returns (order_id, reason, created_at) VALUES (?, ?, ?)",
            (order_id, reason, datetime.now().isoformat(timespec="seconds")),
        )
        return f"RET{cur.lastrowid:04d}"



def create_ticket(issue: str, order_id: str | None = None) -> str:
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (order_id, issue, created_at) VALUES (?, ?, ?)",
            (order_id, issue, datetime.now().isoformat(timespec="seconds")),
        )
        return f"TKT{cur.lastrowid:04d}"

    