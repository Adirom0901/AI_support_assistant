import sqlite3
import os
from datetime import date, timedelta

DB_PATH = os.getenv("ORDERS_DB", "orders.db")
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
 
conn = sqlite3.connect(DB_PATH)

conn.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_name TEXT,
    product_name TEXT,
    order_status TEXT,
    order_date TEXT,
    expected_delivery TEXT,
    tracking_number TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    reason TEXT,
    status TEXT DEFAULT 'Requested',
    created_at TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT,
    issue TEXT NOT NULL,
    status TEXT DEFAULT 'Open',
    created_at TEXT
)
""")

# Dates are relative to today so the test data never goes stale
d = lambda n: (date.today() + timedelta(days=n)).isoformat()

orders = [
    ("ORD1001", "Rahul", "Wireless Headphones",   "Processing",       d(-2),  d(5),   None),
    ("ORD1002", "Priya", "Mechanical Keyboard",   "Shipped",          d(-3),  d(2),   "TRK78231"),
    ("ORD1003", "Aman",  "Gaming Mouse",          "Delivered",        d(-8),  d(-3),  "TRK91542"),  # returnable
    ("ORD1004", "Neha",  "Laptop Stand",          "Out for Delivery", d(-5),  d(0),   "TRK66312"),
    ("ORD1005", "Karan", "USB-C Hub",             "Delivered",        d(-20), d(-15), "TRK55120"),  # return window expired
    ("ORD1006", "Sneha", "Webcam",                "Processing",       d(-1),  d(6),   None),        # second cancellable order
]

conn.executemany("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
conn.commit()
conn.close()
print("Database initialized!")