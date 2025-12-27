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
