"""
utils/helper.py
SAP T-code dictionary, FAQ list, and chat export utilities.
"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional
import re


# ── SAP Transaction Code Dictionary ───────────────────────
# Add more T-codes here as needed for your organisation
SAP_TCODES: Dict[str, Dict[str, str]] = {
    # Materials Management (MM)
    "ME21N": {"module": "MM", "desc": "Create Purchase Order"},
    "ME22N": {"module": "MM", "desc": "Change Purchase Order"},
    "ME23N": {"module": "MM", "desc": "Display Purchase Order"},
    "ME51N": {"module": "MM", "desc": "Create Purchase Requisition"},
    "ME52N": {"module": "MM", "desc": "Change Purchase Requisition"},
    "ME53N": {"module": "MM", "desc": "Display Purchase Requisition"},
    "MIGO":  {"module": "MM", "desc": "Goods Movement (Goods Receipt / Issue)"},
    "MIRO":  {"module": "MM", "desc": "Enter Incoming Invoice (LIV)"},
    "XK01":  {"module": "MM", "desc": "Create Vendor (Centrally)"},
    "XK02":  {"module": "MM", "desc": "Change Vendor (Centrally)"},
    "XK03":  {"module": "MM", "desc": "Display Vendor (Centrally)"},
    "MK01":  {"module": "MM", "desc": "Create Vendor (Purchasing only)"},
    "MM01":  {"module": "MM", "desc": "Create Material Master"},
    "MM02":  {"module": "MM", "desc": "Change Material Master"},
    "MB51":  {"module": "MM", "desc": "Material Document List"},
    "ME2M":  {"module": "MM", "desc": "Purchase Orders by Material"},
    "ME5A":  {"module": "MM", "desc": "List Display of Purchase Requisitions"},
    "MD01":  {"module": "MM", "desc": "MRP Run"},
    "MD01N": {"module": "MM", "desc": "MRP Run (New)"},
    # Finance (FI)
    "FB60":  {"module": "FI", "desc": "Enter Vendor Invoice"},
    "FB65":  {"module": "FI", "desc": "Enter Vendor Credit Memo"},
    "FB70":  {"module": "FI", "desc": "Enter Customer Invoice"},
    "F-02":  {"module": "FI", "desc": "General Posting"},
    "F-53":  {"module": "FI", "desc": "Post Outgoing Payments"},
    "F-28":  {"module": "FI", "desc": "Post Incoming Payments"},
    "FS00":  {"module": "FI", "desc": "G/L Account Master Data"},
    "FBL1N": {"module": "FI", "desc": "Vendor Line Items"},
    "FBL3N": {"module": "FI", "desc": "G/L Account Line Items"},
    "FBL5N": {"module": "FI", "desc": "Customer Line Items"},
    "F110":  {"module": "FI", "desc": "Automatic Payment Program"},
    "FK01":  {"module": "FI", "desc": "Create Vendor (Accounting)"},
    # Sales & Distribution (SD)
    "VA01":  {"module": "SD", "desc": "Create Sales Order"},
    "VA02":  {"module": "SD", "desc": "Change Sales Order"},
    "VA03":  {"module": "SD", "desc": "Display Sales Order"},
    "VF01":  {"module": "SD", "desc": "Create Billing Document"},
    "VL01N": {"module": "SD", "desc": "Create Outbound Delivery"},
    "XD01":  {"module": "SD", "desc": "Create Customer (Centrally)"},
    # Plant Maintenance (PM)
    "IW31":  {"module": "PM", "desc": "Create Maintenance Order"},
    "IW41":  {"module": "PM", "desc": "Confirm Maintenance Order"},
    # Human Resources (HCM)
    "PA40":  {"module": "HCM", "desc": "Personnel Actions"},
    "PA30":  {"module": "HCM", "desc": "Maintain HR Master Data"},
    # Basis
    "SU01":  {"module": "BASIS", "desc": "User Maintenance"},
    "SE38":  {"module": "BASIS", "desc": "ABAP Editor"},
    "SM30":  {"module": "BASIS", "desc": "Maintain Table Views"},
}


def lookup_tcode(tcode: str) -> Optional[Dict[str, str]]:
    """Return details for a T-code, or None if not found."""
    return SAP_TCODES.get(tcode.upper().strip())


def extract_tcodes_from_text(text: str) -> List[str]:
    """Scan text and return any known SAP T-codes found in it."""
    candidates = re.findall(r'\b([A-Z]{1,4}[0-9A-Z]{1,4})\b', text.upper())
    return [c for c in set(candidates) if c in SAP_TCODES]


# ── Frequently Asked Questions ─────────────────────────────
FAQ_LIST: List[Dict[str, str]] = [
    {"question": "How do I create a purchase order?",
     "hint": "Transaction ME21N"},
    {"question": "What is transaction code ME21N?",
     "hint": "Create Purchase Order — MM module"},
    {"question": "How do I create a vendor master?",
     "hint": "XK01 (centrally) or MK01 (purchasing)"},
    {"question": "What are the steps for goods receipt?",
     "hint": "MIGO → Goods Receipt → Purchase Order"},
    {"question": "Explain the invoice verification process.",
     "hint": "MIRO — logistics invoice verification"},
    {"question": "What is the purpose of transaction FB60?",
     "hint": "Enter vendor invoice in FI"},
    {"question": "What is the approval workflow for purchase requests?",
     "hint": "ME51N → release strategy"},
    {"question": "How do I run MRP?",
     "hint": "MD01 or MD01N"},
]


# ── Chat Export ────────────────────────────────────────────

def export_chat_as_txt(chat_history: List[Dict[str, str]]) -> str:
    """Convert chat history to a plain text string for download."""
    lines = [
        "SAP Knowledge Assistant — Conversation Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    for turn in chat_history:
        role = "You" if turn["role"] == "user" else "SAP Assistant"
        lines.append(f"[{role}]")
        lines.append(turn["content"])
        lines.append("")
    return "\n".join(lines)


def export_chat_as_csv(chat_history: List[Dict[str, str]]) -> str:
    """Convert chat history to CSV format for download."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "role", "message"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for turn in chat_history:
        writer.writerow([ts, turn["role"], turn["content"]])
    return output.getvalue()


def format_sources(sources: List[Dict[str, str]]) -> str:
    """Format source citations into a readable markdown list."""
    if not sources:
        return ""
    lines = ["📎 **Sources used:**"]
    for s in sources:
        lines.append(f"- `{s['file']}` — Page {s['page']}")
    return "\n".join(lines)