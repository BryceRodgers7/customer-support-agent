# sample_data.py
from __future__ import annotations

PRODUCTS = [
    {
        "sku": "HDPH-001",
        "name": "Auraluxe Wireless Headphones",
        "price_usd": 129.99,
        "stock": 42,
        "key_features": ["Bluetooth 5.3", "ANC", "30h battery"],
        "warranty_months": 12,
        "category": "audio",
    },
    {
        "sku": "HDPH-002",
        "name": "Auraluxe Wireless Headphones 2",
        "price_usd": 159.99,
        "stock": 22,
        "key_features": ["Bluetooth 5.3", "ANC", "32h battery", "Noise-cancelling"],
        "warranty_months": 12,
        "category": "audio",
    },
    {
        "sku": "MONLCD-001",
        "name": "Lightbend 4K Monitor",
        "price_usd": 599.99,
        "stock": 5,
        "key_features": ["4K", "32-inch", "IPS", "144Hz", "HDR", "G-Sync", "FreeSync"],
        "warranty_months": 24,
        "category": "monitor",
    },
    {
        "sku": "MONLCD-002",
        "name": "Lightbend 4K Monitor 35",
        "price_usd": 899.99,
        "stock": 3,
        "key_features": ["4K", "35-inch", "IPS", "144Hz", "HDR", "G-Sync", "FreeSync", "QHD"],
        "warranty_months": 24,
        "category": "monitor",
    },
    {
        "sku": "KBMX-002",
        "name": "MechaPro Keyboard (MX Brown)",
        "price_usd": 99.00,
        "stock": 0,
        "key_features": ["Hot-swappable", "RGB", "USB-C"],
        "warranty_months": 24,
        "category": "peripherals",
    },
    {
        "sku": "CAM4K-003",
        "name": "StreamCam 4K",
        "price_usd": 149.50,
        "stock": 8,
        "key_features": ["4K30", "1080p60", "Auto-focus"],
        "warranty_months": 12,
        "category": "camera",
    },
]

ORDERS = {
    "A1001": {"status": "Shipped", "carrier": "UPS", "eta_days": 2},
    "A1002": {"status": "Processing", "carrier": None, "eta_days": 5},
    "A1003": {"status": "Delivered", "carrier": "USPS", "eta_days": 0},
}

RETURN_POLICY = {
    "window_days": 30,
    "condition": "Like-new condition with original packaging where possible.",
    "refund_timing": "Refunds processed 3–5 business days after inspection.",
    "exceptions": ["Final sale items", "Gift cards"],
}

# Example internal support guidance you can expand later:
SUPPORT_GUIDANCE = {
    "tone": "Friendly, concise, and solution-oriented.",
    "escalation_rules": [
        "Escalate if customer mentions chargeback, legal threats, personal injury, or repeated failures after troubleshooting.",
        "Escalate if order contains high-value items and is missing after delivery scan.",
    ],
    "troubleshooting": {
        "camera_autofocus": [
            "Confirm lighting conditions and lens cleanliness.",
            "Try a different USB port/cable; avoid unpowered hubs.",
            "Update firmware and camera app/driver.",
            "Test on another device to isolate host issues.",
        ]
    },
}

SHIPPING_RATES = [
    {"method": "Standard", "min_days": 3, "max_days": 7, "base_usd": 5.99},
    {"method": "Expedited", "min_days": 2, "max_days": 3, "base_usd": 12.99},
    {"method": "Overnight", "min_days": 1, "max_days": 1, "base_usd": 24.99},
]

PROMO_CODES = {
    "WELCOME10": {"percent_off": 10, "active": True},
    "FREESHIP": {"free_shipping": True, "active": True},
    "EXPIRED5": {"percent_off": 5, "active": False},
}

# Example: minimal KB articles
KB_ARTICLES = [
    {
        "id": "kb_autofocus",
        "title": "Camera autofocus troubleshooting",
        "tags": ["camera", "autofocus", "streamcam", "usb"],
        "content": "Try cleaning the lens, improving lighting, using a powered USB port, updating firmware/drivers, and testing on another device.",
    },
    {
        "id": "kb_bluetooth",
        "title": "Bluetooth pairing tips",
        "tags": ["bluetooth", "headphones", "pairing"],
        "content": "Reset pairing mode, forget the device on your phone, restart Bluetooth, and ensure no other device is connecting first.",
    },
]

# Optional: pretend we track inventory per warehouse
WAREHOUSES = [
    {"id": "CHI-1", "city": "Chicago", "state": "IL"},
    {"id": "DAL-1", "city": "Dallas", "state": "TX"},
]

INVENTORY_BY_WAREHOUSE = {
    "CHI-1": {"HDPH-001": 20, "HDPH-002": 22, "MONLCD-001": 5, "MONLCD-002": 3, "KBMX-002": 0, "CAM4K-003": 3},
    "DAL-1": {"HDPH-001": 22, "HDPH-002": 24, "MONLCD-001": 6, "MONLCD-002": 4, "KBMX-002": 0, "CAM4K-003": 6},
}