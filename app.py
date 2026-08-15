"""
app.py — SAP Knowledge Assistant v4.1
Features: Streaming, PDF viewer, Persistent search history,
          Typing indicator, Professional formatting, Dark mode toggle
"""

import os, io, re, logging, tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from utils.pdf_loader import load_pdf, split_documents
from utils.vector_db  import (build_vectorstore, save_vectorstore,
                               load_vectorstore, merge_vectorstores, vectorstore_exists)
from utils.rag_chain  import (configure_gemini, ask_question_stream,
                               summarise_document, SUPPORTED_LANGUAGES,
                               _split_followups, format_answer,
                               compare_documents_stream)
from utils.helper     import (lookup_tcode, extract_tcodes_from_text,
                               FAQ_LIST, export_chat_as_txt, export_chat_as_csv)
from utils.feedback       import (save_feedback, save_usage, load_feedback, load_usage,
                               save_search_history, load_search_history, clear_search_history)
from utils.table_extractor import (extract_tables_from_pdf, table_to_markdown,
                                   table_to_text, summarise_table)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SAP Knowledge Assistant",
                   page_icon="📋", layout="wide",
                   initial_sidebar_state="expanded")

# ── Session state ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history":     [],
        "vectorstore":      None,
        "uploaded_files":   [],
        "pdf_bytes_map":    {},
        "gemini_ready":     False,
        "language":         "English",
        "page":             "chat",
        "pending_followup": None,
        "pdf_viewer":       None,
        "dark_mode":        False,
        "bookmarks":        [],
        "extracted_tables": [],
        "compare_result":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Theme variables ────────────────────────────────────────────────────────────
if st.session_state.dark_mode:
    T = {
        "app_bg":         "#0F1117",
        "sidebar_bg":     "#1A1D27",
        "sidebar_border": "#2D3148",
        "card_bg":        "#1E2130",
        "card_border":    "#2D3148",
        "header_bg":      "#1A1D27",
        "header_border":  "#2D3148",
        "text_primary":   "#E2E8F0",
        "text_secondary": "#94A3B8",
        "text_muted":     "#64748B",
        "input_bg":       "#252837",
        "msg_bot_bg":     "#1E2130",
        "msg_bot_border": "#2D3148",
        "msg_bot_color":  "#E2E8F0",
        "source_bg":      "#161924",
        "source_border":  "#2D3148",
        "source_color":   "#94A3B8",
        "feat_chip_bg":   "#252837",
        "feat_chip_bdr":  "#2D3148",
        "feat_chip_clr":  "#94A3B8",
        "faq_bg":         "#1E2130",
        "faq_border":     "#2D3148",
        "faq_color":      "#CBD5E1",
        "faq_hover_bg":   "#1D2D44",
        "faq_hover_bdr":  "#3B5998",
        "faq_hover_clr":  "#93C5FD",
        "tips_bg":        "#161924",
        "tips_border":    "#2D3148",
        "tcode_bg":       "#2D1A0A",
        "tcode_border":   "#7C3A1A",
        "tcode_color":    "#FB923C",
        "tcode_card_bg":  "#1A1208",
        "tcode_card_bdr": "#2D1A0A",
        "dash_card_bg":   "#1E2130",
        "dash_card_bdr":  "#2D3148",
        "typing_bg":      "#1E2130",
        "typing_border":  "#2D3148",
        "scrollbar":      "#2D3148",
        "divider":        "#2D3148",
        "feedback_bdr":   "#2D3148",
        "sidebar_all":    "#E2E8F0",
    }
else:
    T = {
        "app_bg":         "#F0F2F5",
        "sidebar_bg":     "#FFFFFF",
        "sidebar_border": "#E2E8F0",
        "card_bg":        "#FFFFFF",
        "card_border":    "#E2E8F0",
        "header_bg":      "#FFFFFF",
        "header_border":  "#E2E8F0",
        "text_primary":   "#0F172A",
        "text_secondary": "#475569",
        "text_muted":     "#94A3B8",
        "input_bg":       "#F8FAFC",
        "msg_bot_bg":     "#FFFFFF",
        "msg_bot_border": "#E2E8F0",
        "msg_bot_color":  "#1E293B",
        "source_bg":      "#F8FAFC",
        "source_border":  "#E2E8F0",
        "source_color":   "#64748B",
        "feat_chip_bg":   "#FFFFFF",
        "feat_chip_bdr":  "#E2E8F0",
        "feat_chip_clr":  "#475569",
        "faq_bg":         "#FFFFFF",
        "faq_border":     "#E2E8F0",
        "faq_color":      "#334155",
        "faq_hover_bg":   "#EFF6FF",
        "faq_hover_bdr":  "#BFDBFE",
        "faq_hover_clr":  "#1D4ED8",
        "tips_bg":        "#F8FAFC",
        "tips_border":    "#E2E8F0",
        "tcode_bg":       "#FFF7ED",
        "tcode_border":   "#FED7AA",
        "tcode_color":    "#C2410C",
        "tcode_card_bg":  "#F8FAFC",
        "tcode_card_bdr": "#E2E8F0",
        "dash_card_bg":   "#FFFFFF",
        "dash_card_bdr":  "#E2E8F0",
        "typing_bg":      "#FFFFFF",
        "typing_border":  "#E2E8F0",
        "scrollbar":      "#CBD5E1",
        "divider":        "#E2E8F0",
        "feedback_bdr":   "#F1F5F9",
        "sidebar_all":    "#1E293B",
    }

# ── Dynamic CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif !important; }}
#MainMenu, footer, header   {{ visibility: hidden; }}
.stApp                       {{ background: {T['app_bg']} !important; }}

section[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['sidebar_border']} !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.08) !important;
}}
section[data-testid="stSidebar"] * {{ color: {T['sidebar_all']} !important; }}

.logo-container {{ display:flex; align-items:center; gap:12px; padding:22px 8px 20px; border-bottom:1px solid {T['sidebar_border']}; margin-bottom:4px; }}
.logo-mark {{ width:36px; height:36px; border-radius:7px; background:#0F52BA; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:800; color:white; font-family:'JetBrains Mono',monospace; flex-shrink:0; }}
.logo-title {{ font-size:13.5px !important; font-weight:700 !important; color:{T['text_primary']} !important; line-height:1.2; }}
.logo-sub   {{ font-size:10px !important; font-weight:500 !important; color:{T['text_muted']} !important; letter-spacing:0.06em; text-transform:uppercase; }}

.sb-label {{ font-size:10px !important; font-weight:700 !important; color:{T['text_muted']} !important; text-transform:uppercase; letter-spacing:0.12em; margin:18px 0 8px !important; padding-left:2px; display:block; }}

.status-badge {{ display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:6px; font-size:11.5px; font-weight:600; margin-bottom:6px; }}
.s-ok   {{ background:#F0FDF4; color:#16A34A; border:1px solid #BBF7D0; }}
.s-warn {{ background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; }}

.stButton > button {{
    background:#0F52BA !important; color:#FFFFFF !important; border:none !important;
    border-radius:7px !important; font-weight:500 !important; font-size:12.5px !important;
    box-shadow:0 1px 3px rgba(15,82,186,0.2) !important; transition:all 0.15s !important;
    white-space:nowrap !important; padding:8px 6px !important;
}}
.stButton > button:hover {{
    background:#0D47A1 !important; box-shadow:0 3px 10px rgba(15,82,186,0.28) !important;
    transform:translateY(-1px) !important;
}}

/* ── View p.X button ── */
div[data-testid="stButton"]:has(button[kind="secondary"]) button,
.view-btn button, .view-btn .stButton > button {{
    background:{T['input_bg']} !important;
    color:#1D4ED8 !important;
    border:1px solid #BFDBFE !important;
    box-shadow:none !important;
    font-size:11.5px !important;
    font-weight:500 !important;
    padding:3px 12px !important;
    min-height:0px !important;
    height:30px !important;
    transform:none !important;
    width:auto !important;
}}

.stSelectbox > div > div {{ border-radius:7px !important; border-color:{T['card_border']} !important; font-size:13px !important; background:{T['input_bg']} !important; color:{T['text_primary']} !important; }}

.page-header {{
    background: linear-gradient(270deg, {T['header_bg']}, #0a1628, #0F52BA22, {T['header_bg']});
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
    border-bottom:1px solid {T['header_border']}; padding:18px 28px 16px; margin:-1rem -1rem 20px -1rem;
    position: relative; overflow: hidden;
}}
.page-header::after {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(15,82,186,0.06), transparent);
    animation: shimmer 3s ease infinite;
}}
@keyframes gradientShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes shimmer {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
.page-header h1 {{ font-size:17px !important; font-weight:700 !important; color:{T['text_primary']} !important; margin:0 !important; letter-spacing:-0.02em; position:relative; z-index:1; }}
.page-header p  {{ font-size:12px !important; color:{T['text_secondary']} !important; margin:3px 0 0 !important; position:relative; z-index:1; }}
.lang-badge {{ display:inline-flex; align-items:center; background:#EFF6FF; border:1px solid #BFDBFE; border-radius:5px; padding:3px 9px; font-size:11px; font-weight:600; color:#1D4ED8; }}

.msg-meta {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.meta-user {{ justify-content:flex-end; }}
.meta-bot  {{ justify-content:flex-start; }}

.avatar {{
    width:26px; height:26px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:700; flex-shrink:0;
}}
.avatar-user {{
    background:#0F52BA; color:#FFFFFF;
}}
.avatar-bot {{
    background:linear-gradient(135deg,#0F52BA,#3B82F6);
    color:#FFFFFF; font-size:9px; letter-spacing:-0.5px;
}}
.meta-label {{
    font-size:10px; font-weight:700; letter-spacing:0.08em;
    text-transform:uppercase;
}}
.meta-label-user {{ color:{T['text_muted']}; }}
.meta-label-bot  {{ color:#0F52BA; }}
.meta-time {{
    font-size:10px; color:{T['text_muted']};
    font-family:'JetBrains Mono',monospace; margin-left:4px;
}}

@keyframes slideUp {{
    from {{ opacity:0; transform:translateY(12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.msg-user {{
    background:#0F52BA; border-radius:12px 12px 3px 12px;
    padding:12px 17px; margin:2px 0 10px 80px;
    color:#FFFFFF; font-size:14px; line-height:1.6;
    box-shadow:0 2px 8px rgba(15,82,186,0.18);
    animation: slideUp 0.25s ease forwards;
}}
.msg-bot {{
    background:{T['msg_bot_bg']}; border:1px solid {T['msg_bot_border']}; border-radius:3px 12px 12px 12px;
    padding:15px 19px; margin:2px 0 4px 0;
    color:{T['msg_bot_color']}; font-size:14px; line-height:1.75;
    box-shadow:0 1px 4px rgba(0,0,0,0.08);
    animation: slideUp 0.25s ease forwards;
}}
.msg-bot code {{
    background:{T['input_bg']}; border:1px solid {T['card_border']}; border-radius:4px;
    padding:1px 6px; font-family:'JetBrains Mono',monospace;
    font-size:12.5px; color:#C2410C; font-weight:600;
}}
.msg-bot strong {{ color:#3B82F6; font-weight:600; }}

.source-box {{
    background:{T['source_bg']}; border:1px solid {T['source_border']};
    border-left:3px solid #0F52BA; border-radius:0 6px 6px 0;
    padding:9px 13px; margin-top:8px;
    font-size:11.5px; color:{T['source_color']};
    font-family:'JetBrains Mono',monospace; line-height:2;
}}

.tcode-pill {{
    display:inline-block; background:{T['tcode_bg']}; border:1px solid {T['tcode_border']};
    color:{T['tcode_color']}; border-radius:4px; padding:2px 8px;
    font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace; margin:2px;
}}

.typing-wrap {{ display:flex; flex-direction:column; margin:2px 0 10px 0; }}
.typing-box {{
    display:inline-flex; align-items:center; gap:5px;
    padding:14px 18px; background:{T['typing_bg']}; border:1px solid {T['typing_border']};
    border-radius:3px 12px 12px 12px; box-shadow:0 1px 4px rgba(0,0,0,0.05); width:fit-content;
}}
.typing-dot {{ width:7px; height:7px; border-radius:50%; background:{T['text_muted']}; animation:bounce 1.2s infinite ease-in-out; }}
.typing-dot:nth-child(2) {{ animation-delay:0.2s; }}
.typing-dot:nth-child(3) {{ animation-delay:0.4s; }}
@keyframes bounce {{ 0%,60%,100%{{transform:translateY(0);background:{T['text_muted']};}} 30%{{transform:translateY(-6px);background:#0F52BA;}} }}

.pdf-viewer-bar {{
    background:{T['input_bg']}; border:1px solid {T['card_border']}; border-radius:8px 8px 0 0;
    padding:10px 16px; font-size:12.5px; font-weight:600; color:{T['text_secondary']};
    display:flex; justify-content:space-between; align-items:center;
}}

.welcome {{ text-align:center; padding:60px 20px 44px; }}
.welcome h2 {{ font-size:22px; font-weight:700; color:{T['text_primary']}; letter-spacing:-0.02em; margin-bottom:10px; }}
.welcome p  {{ font-size:13.5px; color:{T['text_secondary']}; max-width:440px; margin:0 auto 26px; line-height:1.6; }}
.feat-chip  {{ display:inline-block; background:{T['feat_chip_bg']}; border:1px solid {T['feat_chip_bdr']}; border-radius:6px; padding:5px 12px; font-size:12px; font-weight:500; color:{T['feat_chip_clr']}; margin:3px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }}

.faq-btn .stButton > button {{
    background:{T['faq_bg']} !important; color:{T['faq_color']} !important;
    border:1px solid {T['faq_border']} !important; box-shadow:none !important;
    font-size:12px !important; font-weight:400 !important;
    text-align:left !important; transform:none !important;
    white-space:normal !important; word-break:break-word !important;
    height:auto !important; min-height:0 !important; line-height:1.4 !important;
    padding:8px 12px !important;
}}
.faq-btn .stButton > button:hover {{
    background:{T['faq_hover_bg']} !important; border-color:{T['faq_hover_bdr']} !important;
    color:{T['faq_hover_clr']} !important; transform:none !important; box-shadow:none !important;
}}

.dash-card {{ background:{T['dash_card_bg']}; border:1px solid {T['dash_card_bdr']}; border-radius:10px; padding:20px 16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05); }}
.dash-number {{ font-size:32px; font-weight:800; color:#0F52BA; letter-spacing:-0.03em; }}
.dash-label  {{ font-size:10.5px; color:{T['text_muted']}; margin-top:5px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }}

.tcode-card {{ background:{T['tcode_card_bg']}; border:1px solid {T['tcode_card_bdr']}; border-left:3px solid #0F52BA; border-radius:0 8px 8px 0; padding:12px 14px; margin-top:8px; }}
.tcode-card .tc {{ font-size:17px; font-weight:800; color:{T['text_primary']}; font-family:'JetBrains Mono',monospace; }}
.tcode-card .td {{ font-size:13px; color:{T['text_secondary']}; margin-top:4px; }}
.tcode-card .tm {{ font-size:11px; color:{T['text_muted']}; margin-top:3px; }}

/* ── Dark mode toggle — subtle ── */
.dark-toggle .stButton > button {{
    background:{"#1E2130" if st.session_state.dark_mode else "#F8FAFC"} !important;
    color:{"#94A3B8" if st.session_state.dark_mode else "#64748B"} !important;
    border:1px solid {T['card_border']} !important;
    box-shadow:none !important; font-size:12px !important;
    font-weight:500 !important; width:100% !important;
}}
.dark-toggle .stButton > button:hover {{
    background:{"#252837" if st.session_state.dark_mode else "#EFF6FF"} !important;
    color:{"#CBD5E1" if st.session_state.dark_mode else "#1D4ED8"} !important;
    transform:none !important; box-shadow:none !important;
}}

/* ── View page button — small, outlined ── */
.view-btn .stButton > button {{
    background:{"#1E2130" if st.session_state.dark_mode else "#F8FAFC"} !important;
    color:{"#93C5FD" if st.session_state.dark_mode else "#1D4ED8"} !important;
    border:1px solid {"#3B5998" if st.session_state.dark_mode else "#BFDBFE"} !important;
    box-shadow:none !important; font-size:11.5px !important;
    font-weight:500 !important; border-radius:6px !important; transform:none !important;
}}
.view-btn .stButton > button:hover {{
    background:{"#1D2D44" if st.session_state.dark_mode else "#EFF6FF"} !important;
    transform:none !important; box-shadow:none !important;
}}

/* ── Feedback thumbs — small inline ── */
.feedback-btn .stButton > button {{
    background:transparent !important; color:{T['text_muted']} !important;
    border:1px solid {T['card_border']} !important;
    box-shadow:none !important; font-size:13px !important;
    padding:2px 10px !important; border-radius:6px !important;
    height:28px !important; transform:none !important; width:auto !important;
}}
.feedback-btn .stButton > button:hover {{
    background:{T['input_bg']} !important;
    transform:none !important; box-shadow:none !important;
}}

/* ── Follow-up chips — outlined pill style ── */
.followup-btn .stButton > button {{
    background:{"#1E2130" if st.session_state.dark_mode else "#F8FAFC"} !important;
    color:{"#93C5FD" if st.session_state.dark_mode else "#1D4ED8"} !important;
    border:1px solid {"#3B5998" if st.session_state.dark_mode else "#BFDBFE"} !important;
    box-shadow:none !important; font-size:12px !important;
    font-weight:400 !important; border-radius:20px !important; transform:none !important;
}}
.followup-btn .stButton > button:hover {{
    background:{"#1D2D44" if st.session_state.dark_mode else "#EFF6FF"} !important;
    transform:none !important; box-shadow:none !important;
}}

/* ── Search history — plain text, no blue ── */
.hist-btn .stButton > button {{
    background:{T['input_bg']} !important; color:{T['text_secondary']} !important;
    border:1px solid {T['card_border']} !important;
    box-shadow:none !important; font-size:11.5px !important; font-weight:400 !important;
    text-align:left !important; border-radius:6px !important;
    padding:6px 10px !important; transform:none !important;
    white-space:normal !important; word-break:break-word !important;
    height:auto !important; min-height:0 !important; line-height:1.4 !important;
}}
.hist-btn .stButton > button:hover {{
    background:{T['faq_hover_bg']} !important; color:#0F52BA !important;
    border-color:#BFDBFE !important;
    transform:none !important; box-shadow:none !important;
}}

::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{T['scrollbar']}; border-radius:3px; }}

/* ── Bookmark + Copy action buttons ── */
.action-bar {{
    display:flex; align-items:center; gap:6px;
    margin-top:8px;
}}
.copy-btn, .bookmark-btn {{
    display:inline-flex; align-items:center; gap:4px;
    padding:3px 10px; border-radius:5px; font-size:11px;
    font-weight:500; cursor:pointer; border:1px solid {T['card_border']};
    background:{T['input_bg']}; color:{T['text_secondary']};
    transition: all 0.15s; user-select:none;
}}
.copy-btn:hover {{ background:#EFF6FF; color:#1D4ED8; border-color:#BFDBFE; }}
.bookmark-btn:hover {{ background:#FFF7ED; color:#C2410C; border-color:#FED7AA; }}
.bookmark-btn.saved {{ background:#FFF7ED; color:#C2410C; border-color:#FED7AA; }}

/* Keyboard shortcut hint */
.kb-hint {{
    position:fixed; bottom:80px; right:20px;
    background:{T['card_bg']}; border:1px solid {T['card_border']};
    border-radius:8px; padding:6px 12px; font-size:11px;
    color:{T['text_muted']}; font-family:'JetBrains Mono',monospace;
    opacity:0; animation: fadeHint 3s ease 1s forwards;
    z-index:999; pointer-events:none;
}}
@keyframes fadeHint {{
    0%   {{ opacity:0; }}
    20%  {{ opacity:1; }}
    80%  {{ opacity:1; }}
    100% {{ opacity:0; }}
}}

hr {{ border:none !important; border-top:1px solid {T['divider']} !important; margin:16px 0 !important; }}

[data-testid="stChatInput"] textarea {{
    background:{T['input_bg']} !important;
    color:{T['text_primary']} !important;
}}

/* ── Small action buttons: 👍 👎 and Open page X ── */
button[kind="secondary"] {{
    background:{T['input_bg']} !important;
    color:{T['text_secondary']} !important;
    border:1px solid {T['card_border']} !important;
    box-shadow:none !important;
    font-size:12px !important;
    padding:2px 10px !important;
    min-height:0 !important;
    height:30px !important;
    transform:none !important;
    border-radius:6px !important;
    font-weight:400 !important;
}}
button[kind="secondary"]:hover {{
    background:{T['input_bg']} !important;
    border-color:{T['text_muted']} !important;
    transform:none !important;
    box-shadow:none !important;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def init_gemini():
    try:
        configure_gemini(); return True
    except EnvironmentError as e:
        return str(e)

gemini_status = init_gemini()
if gemini_status is True:
    st.session_state.gemini_ready = True

if st.session_state.vectorstore is None and vectorstore_exists():
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = load_vectorstore()


# ── PDF page renderer ──────────────────────────────────────────────────────────
def render_pdf_page(filename: str, page_num):
    try:
        from pdf2image import convert_from_bytes
        pdf_bytes = st.session_state.pdf_bytes_map.get(filename)
        if not pdf_bytes:
            st.error(f"'{filename}' not in memory — please re-upload the PDF.")
            return
        page_idx = max(0, int(page_num) - 1)
        images = convert_from_bytes(pdf_bytes, first_page=page_idx+1,
                                    last_page=page_idx+1, dpi=150)
        if images:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            buf.seek(0)
            st.markdown(f"""
            <div class="pdf-viewer-bar">
                <span>📄 {filename}</span>
                <span>Page {page_num}</span>
            </div>""", unsafe_allow_html=True)
            st.image(buf, use_container_width=True)
        else:
            st.warning("Could not render this page.")
    except ImportError:
        st.error("Run: pip install pdf2image pillow")
    except Exception as e:
        st.error(f"PDF viewer error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"""
    <div class="logo-container">
        <div class="logo-mark">SAP</div>
        <div>
            <div class="logo-title">SAP Knowledge</div>
            <div class="logo-sub">AI Assistant · v4.1</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<span class="sb-label">Navigation</span>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    with n1:
        if st.button("Chat", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
    with n2:
        if st.button("Tables", use_container_width=True):
            st.session_state.page = "tables"; st.rerun()
    n3, n4 = st.columns(2)
    with n3:
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"; st.rerun()
    with n4:
        if st.button("Compare", use_container_width=True):
            st.session_state.page = "compare"; st.rerun()

    st.divider()

    # Dark mode toggle
    st.markdown('<span class="sb-label">Appearance</span>', unsafe_allow_html=True)
    toggle_label = "☀️  Switch to Light Mode" if st.session_state.dark_mode else "🌙  Switch to Dark Mode"
    st.markdown('<div class="dark-toggle">', unsafe_allow_html=True)
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Bookmarks panel
    if st.session_state.bookmarks:
        st.markdown('<span class="sb-label">Bookmarks</span>', unsafe_allow_html=True)
        for bi, bm in enumerate(reversed(st.session_state.bookmarks[-5:])):
            q = bm.get("question", "Answer")
            short = q[:30] + "…" if len(q) > 30 else q
            with st.expander(f"★ {short}", expanded=False):
                st.markdown(bm["content"][:300] + "…" if len(bm["content"]) > 300 else bm["content"])
                if st.button("Remove", key=f"rm_bm_{bi}", type="secondary"):
                    st.session_state.bookmarks = [
                        b for b in st.session_state.bookmarks
                        if b.get("content") != bm["content"]
                    ]
                    st.rerun()
        st.divider()

    # Search history
    st.markdown('<span class="sb-label">Search History</span>', unsafe_allow_html=True)
    history_rows = load_search_history(limit=20)
    if history_rows:
        seen_q, unique_rows = set(), []
        for row in history_rows:
            q = row.get("question","").strip().lower()
            if q not in seen_q:
                seen_q.add(q)
                unique_rows.append(row)
            if len(unique_rows) == 6:
                break
        for row in unique_rows:
            q = row.get("question", "")
            short = q[:28] + "…" if len(q) > 28 else q
            if st.button(short, key=f"hist_{row['timestamp']}", use_container_width=True):
                st.session_state.pending_followup = q
                st.session_state.page = "chat"
                st.rerun()
        if st.button("Clear History", use_container_width=True):
            clear_search_history(); st.rerun()
    else:
        st.caption("No searches yet.")

    st.divider()

    # Language
    st.markdown('<span class="sb-label">Response Language</span>', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        "Language", options=list(SUPPORTED_LANGUAGES.keys()),
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.language),
        label_visibility="collapsed")
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang; st.rerun()

    st.divider()

    # System status
    st.markdown('<span class="sb-label">System Status</span>', unsafe_allow_html=True)
    if st.session_state.gemini_ready:
        st.markdown('<span class="status-badge s-ok">● Gemini Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge s-warn">● Gemini Not Set</span>', unsafe_allow_html=True)
        st.caption(str(gemini_status))
    if st.session_state.vectorstore:
        st.markdown('<span class="status-badge s-ok">● Knowledge Base Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge s-warn">● No Documents Indexed</span>', unsafe_allow_html=True)

    st.divider()

    # PDF Upload
    st.markdown('<span class="sb-label">Upload Documents</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PDFs", type=["pdf"],
                                accept_multiple_files=True, label_visibility="collapsed")
    with st.expander("Chunking settings"):
        chunk_size    = st.slider("Chunk size",    500, 2000, 1000, 50)
        chunk_overlap = st.slider("Chunk overlap",  50,  500,  200, 25)
        top_k         = st.slider("Results (K)",     1,   10,    5)

    if uploaded:
        if st.button("Index Documents", use_container_width=True):
            new_docs = []
            progress = st.progress(0, text="Processing...")
            for idx, pdf_file in enumerate(uploaded):
                pdf_bytes = pdf_file.read()
                st.session_state.pdf_bytes_map[pdf_file.name] = pdf_bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes); tmp_path = tmp.name
                try:
                    docs, pages = load_pdf(tmp_path)
                    for d in docs: d.metadata["source_file"] = pdf_file.name
                    new_docs.extend(docs)
                    if pdf_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(pdf_file.name)
                    st.success(f"{pdf_file.name} — {pages} pages")
                except Exception as e:
                    st.error(f"{pdf_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)
                progress.progress((idx + 1) / len(uploaded))
            if new_docs:
                with st.spinner("Building search index..."):
                    chunks = split_documents(new_docs, chunk_size, chunk_overlap)
                    new_vs = build_vectorstore(chunks)
                    if st.session_state.vectorstore:
                        st.session_state.vectorstore = merge_vectorstores(
                            st.session_state.vectorstore, new_vs)
                    else:
                        st.session_state.vectorstore = new_vs
                    save_vectorstore(st.session_state.vectorstore)
                # Also extract tables
                all_tables = []
                for pdf_file in uploaded:
                    try:
                        import tempfile, os as _os
                        pdf_bytes = st.session_state.pdf_bytes_map.get(pdf_file.name, b"")
                        if pdf_bytes:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(pdf_bytes); tmp_path = tmp.name
                            tbls = extract_tables_from_pdf(tmp_path)
                            for t in tbls:
                                t["source"] = pdf_file.name
                            all_tables.extend(tbls)
                            _os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Table extraction failed for {pdf_file.name}: {e}")
                st.session_state.extracted_tables = all_tables
                progress.empty()
                st.success(f"{len(chunks)} chunks indexed · {len(all_tables)} tables found")
                st.rerun()

    if st.session_state.uploaded_files:
        st.markdown('<span class="sb-label">Indexed Documents</span>', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_files:
            st.markdown(
                f"<span style='font-size:12px;color:{T['text_secondary']};font-family:JetBrains Mono,monospace'>📄 {fname}</span>",
                unsafe_allow_html=True)

    st.divider()

    # Summariser
    st.markdown('<span class="sb-label">Document Summariser</span>', unsafe_allow_html=True)
    if st.button("Summarise Documents", use_container_width=True):
        if st.session_state.vectorstore:
            with st.spinner("Generating summary..."):
                sample  = st.session_state.vectorstore.similarity_search(
                    "SAP process overview modules transaction codes", k=10)
                summary = summarise_document(sample)
            with st.expander("Summary", expanded=True):
                st.markdown(summary)
        else:
            st.warning("Upload documents first.")

    st.divider()

    # T-Code lookup
    st.markdown('<span class="sb-label">T-Code Reference</span>', unsafe_allow_html=True)
    tcode_input = st.text_input("T-Code", placeholder="e.g. ME21N", label_visibility="collapsed")
    if tcode_input:
        info = lookup_tcode(tcode_input.strip())
        if info:
            st.markdown(f"""
            <div class="tcode-card">
                <div class="tc">{tcode_input.upper()}</div>
                <div class="td">{info['desc']}</div>
                <div class="tm">Module: {info['module']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.caption(f"`{tcode_input.upper()}` not in reference dictionary.")

    st.divider()

    # Controls
    st.markdown('<span class="sb-label">Controls</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []; st.rerun()
    with c2:
        if st.button("Reset DB", use_container_width=True):
            import shutil
            st.session_state.vectorstore    = None
            st.session_state.uploaded_files = []
            st.session_state.pdf_bytes_map  = {}
            if Path("vectorstore").exists(): shutil.rmtree("vectorstore")
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Summarise Session", use_container_width=True):
            msgs = [t for t in st.session_state.chat_history if t["role"] == "user"]
            questions = "\n".join(f"- {m['content']}" for m in msgs[:10])
            answers   = [t for t in st.session_state.chat_history if t["role"] == "assistant"]
            ans_text  = "\n".join(a["content"][:200] for a in answers[:5])
            from utils.rag_chain import _call_groq
            summary = _call_groq([
                {"role":"system","content":"You are a concise summariser."},
                {"role":"user","content":
                    f"Summarise this SAP support session in 4-6 bullet points:\n\nQuestions asked:\n{questions}\n\nKey answers:\n{ans_text}"}
            ])
            with st.expander("Session Summary", expanded=True):
                st.markdown(summary)

    if st.session_state.chat_history:
        st.divider()
        st.markdown('<span class="sb-label">Export Conversation</span>', unsafe_allow_html=True)
        chat_simple = [{"role": t["role"], "content": t["content"]}
                       for t in st.session_state.chat_history]
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("TXT", data=export_chat_as_txt(chat_simple),
                file_name=f"sap_chat_{ts}.txt", mime="text/plain", use_container_width=True)
        with e2:
            st.download_button("CSV", data=export_chat_as_csv(chat_simple),
                file_name=f"sap_chat_{ts}.csv", mime="text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  TABLES PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "tables":
    st.markdown(f"""
    <div class="page-header">
        <h1>Document Tables</h1>
        <p>All tables extracted from your SAP documents — browse, search and get AI explanations</p>
    </div>""", unsafe_allow_html=True)

    tables = st.session_state.extracted_tables

    if not tables:
        st.info("No tables found yet. Upload and index a PDF first, then come back here.")
        st.stop()

    # Summary row
    st.markdown(f"""
    <div style="display:flex;gap:16px;margin-bottom:20px;">
        <div class="dash-card" style="flex:1;padding:16px;">
            <div class="dash-number">{len(tables)}</div>
            <div class="dash-label">Tables Found</div>
        </div>
        <div class="dash-card" style="flex:1;padding:16px;">
            <div class="dash-number">{len(set(t['page'] for t in tables))}</div>
            <div class="dash-label">Pages with Tables</div>
        </div>
        <div class="dash-card" style="flex:1;padding:16px;">
            <div class="dash-number">{sum(len(t['rows']) for t in tables)}</div>
            <div class="dash-label">Total Rows</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Search filter
    search = st.text_input("Search tables", placeholder="e.g. material, vendor, price...",
                           label_visibility="collapsed")

    # Filter tables by search
    filtered = tables
    if search:
        filtered = []
        for t in tables:
            table_str = table_to_text(t).lower()
            if search.lower() in table_str:
                filtered.append(t)
        st.caption(f"{len(filtered)} of {len(tables)} tables match '{search}'")

    if not filtered:
        st.warning("No tables match your search.")
        st.stop()

    # Display each table
    for t in filtered:
        title = t.get("title", f"Table {t['index']}")
        with st.expander(
            f"📊 {title} — {t['shape']} · {t['source']}",
            expanded=(len(filtered) <= 2)
        ):
            # Context
            if t.get("context"):
                st.markdown(f"<div style='font-size:12px;color:#64748B;margin-bottom:8px;'>"
                           f"📍 Context: {t['context'][:120]}...</div>",
                           unsafe_allow_html=True)

            # Table display
            st.dataframe(
                t["df"],
                use_container_width=True,
                hide_index=True,
            )

            col1, col2 = st.columns([1, 1])

            # Explain button
            with col1:
                if st.button(f"Explain this table", key=f"explain_{t['source']}_{t['page']}_{t['index']}",
                             use_container_width=True):
                    with st.spinner("Analysing table..."):
                        from utils.rag_chain import _call_groq
                        prompt = summarise_table(t)
                        explanation = _call_groq([
                            {"role": "system", "content":
                                "You are an SAP documentation expert. Explain tables clearly and professionally."},
                            {"role": "user", "content": prompt}
                        ])
                    st.markdown(f"""
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;
                                border-left:3px solid #0F52BA;border-radius:0 8px 8px 0;
                                padding:14px;margin-top:10px;font-size:13.5px;line-height:1.7;">
                        {explanation}
                    </div>""", unsafe_allow_html=True)

            # Download as CSV
            with col2:
                csv = t["df"].to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    data=csv,
                    file_name=f"table_{t['index']}_page{t['page']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{t['source']}_{t['page']}_{t['index']}"
                )

    st.stop()

if st.session_state.page == "dashboard":
    st.markdown(f"""
    <div class="page-header">
        <h1>Usage Dashboard</h1>
        <p>Analytics, feedback, and documentation gap insights</p>
    </div>""", unsafe_allow_html=True)

    usage    = load_usage()
    feedback = load_feedback()
    total_q     = len(usage)
    answered    = sum(1 for u in usage if u.get("answered") == "yes")
    unanswered  = total_q - answered
    thumbs_up   = sum(1 for f in feedback if f.get("rating") == "positive")
    thumbs_down = sum(1 for f in feedback if f.get("rating") == "negative")
    satisfaction = f"{round(thumbs_up/(thumbs_up+thumbs_down)*100)}%" if (thumbs_up+thumbs_down) > 0 else "N/A"

    cols = st.columns(5)
    for col, num, lbl in [
        (cols[0], total_q, "Total Questions"),
        (cols[1], answered, "Answered"),
        (cols[2], unanswered, "Not Found"),
        (cols[3], thumbs_up, "Positive Feedback"),
        (cols[4], satisfaction, "Satisfaction Rate"),
    ]:
        with col:
            st.markdown(f'<div class="dash-card"><div class="dash-number">{num}</div><div class="dash-label">{lbl}</div></div>',
                        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if usage:
        import pandas as pd
        df = pd.DataFrame(usage)
        l, r = st.columns(2)
        with l:
            st.markdown("#### Questions Over Time")
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            st.line_chart(df.groupby("date").size().reset_index(name="q").set_index("date"))
        with r:
            st.markdown("#### Language Breakdown")
            if "language" in df.columns:
                lc = df["language"].value_counts().reset_index()
                lc.columns = ["Language", "Count"]
                st.bar_chart(lc.set_index("Language"))

        st.markdown("#### Unanswered Questions")
        udf = df[df["answered"] == "no"][["timestamp","question"]]
        if not udf.empty:
            st.dataframe(udf.rename(columns={"timestamp":"Time","question":"Question"}),
                         use_container_width=True, hide_index=True)
        else:
            st.success("No unanswered questions yet.")

        if feedback:
            st.markdown("#### Recent Feedback")
            fb_df = pd.DataFrame(feedback)[["timestamp","rating","question","comment"]]
            fb_df["rating"] = fb_df["rating"].map({"positive":"Positive","negative":"Negative"})
            st.dataframe(fb_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Start asking questions in the Chat tab.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "compare":
    st.markdown(f"""
    <div class="page-header">
        <h1>Document Comparison</h1>
        <p>Compare how different documents address the same topic side by side</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    available_docs = st.session_state.uploaded_files

    if len(available_docs) < 2:
        st.warning(
            f"You need at least 2 indexed documents to compare. "
            f"Currently you have {len(available_docs)}. "
            f"Upload another PDF in the sidebar and index it."
        )
        st.stop()

    if not st.session_state.gemini_ready:
        st.warning("Add your API key / start Ollama to enable comparisons.")
        st.stop()

    st.markdown("##### Select documents to compare")
    selected_docs = st.multiselect(
        "Documents",
        options=available_docs,
        default=available_docs[:2],
        label_visibility="collapsed",
    )

    compare_question = st.text_input(
        "What would you like to compare?",
        placeholder="e.g. How does the Purchase Order process differ between these documents?",
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_compare = st.button("Compare", use_container_width=True, type="secondary")

    if run_compare and compare_question and len(selected_docs) >= 2:
        st.markdown('<div class="msg-meta meta-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-user">{compare_question}</div>', unsafe_allow_html=True)

        typing_ph = st.empty()
        typing_ph.markdown("""
        <div class="msg-meta meta-bot">SAP Assistant</div>
        <div class="typing-wrap">
            <div class="typing-box">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        result = compare_documents_stream(
            question=compare_question,
            vectorstore=st.session_state.vectorstore,
            doc_names=selected_docs,
            k_per_doc=5,
        )

        typing_ph.empty()
        st.markdown('<div class="msg-meta meta-bot">SAP Assistant</div>', unsafe_allow_html=True)
        stream_box = st.empty()
        full_text  = ""
        for chunk in result["stream"]:
            full_text += chunk
            stream_box.markdown(full_text)

        formatted = format_answer(full_text)
        stream_box.markdown(formatted)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.session_state.compare_result = {
            "question":       compare_question,
            "answer":         formatted,
            "docs":           selected_docs,
            "per_doc_chunks": result["per_doc_chunks"],
        }

    elif run_compare and len(selected_docs) < 2:
        st.error("Select at least 2 documents.")
    elif run_compare and not compare_question:
        st.error("Enter a question or topic to compare.")

    # Show source breakdown if a comparison was run
    if st.session_state.compare_result:
        with st.expander("View source excerpts per document"):
            for doc_name, chunks in st.session_state.compare_result["per_doc_chunks"].items():
                st.markdown(f"**{doc_name}**")
                if not chunks:
                    st.caption("No relevant content found.")
                else:
                    for ch in chunks:
                        page = ch.metadata.get("page", "?")
                        st.markdown(
                            f"<div class='source-box'>Page {page}: {ch.page_content[:200]}...</div>",
                            unsafe_allow_html=True
                        )
                st.markdown("---")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
top_k = 5

# PDF viewer panel
if st.session_state.pdf_viewer:
    v = st.session_state.pdf_viewer
    cl, ct = st.columns([1, 9])
    with cl:
        if st.button("✕ Close", key="close_pdf"):
            st.session_state.pdf_viewer = None; st.rerun()
    with ct:
        st.markdown(f"<div style='padding:7px 0;font-size:13px;font-weight:600;color:{T['text_secondary']}'>"
                    f"📄 {v['file']} — Page {v['page']}</div>", unsafe_allow_html=True)
    render_pdf_page(v["file"], v["page"])
    st.divider()

main_col, faq_col = st.columns([3, 1])

with main_col:
    st.markdown(f"""
    <div class="page-header">
        <h1>SAP Knowledge Assistant</h1>
        <p>Responding in <strong>{st.session_state.language}</strong> &nbsp;·&nbsp;
           Ollama (LLaMA 3.1) &nbsp;·&nbsp;
           <span class="lang-badge">{"Knowledge Base Active" if st.session_state.vectorstore else "No Documents Loaded"}</span>
        </p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.vectorstore:
        st.markdown(f"""
        <div class="welcome">
            <h2>Upload your SAP documents to begin</h2>
            <p>Add PDF manuals in the sidebar, then ask any question in natural language.</p>
            <div>
                <span class="feat-chip">Streaming responses</span>
                <span class="feat-chip">PDF page viewer</span>
                <span class="feat-chip">Search history</span>
                <span class="feat-chip">Dark mode</span>
                <span class="feat-chip">10 languages</span>
                <span class="feat-chip">T-Code reference</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Render chat history
    for i, turn in enumerate(st.session_state.chat_history):
        if turn["role"] == "user":
            ts = turn.get("timestamp","")
            time_str = ts[11:16] if len(ts) >= 16 else ""
            st.markdown(f'''<div class="msg-meta meta-user">
                <span class="meta-time">{time_str}</span>
                <span class="meta-label meta-label-user">You</span>
                <div class="avatar avatar-user">D</div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{turn["content"]}</div>', unsafe_allow_html=True)
        else:
            ts = turn.get("timestamp","")
            time_str = ts[11:16] if len(ts) >= 16 else ""
            st.markdown(f'''<div class="msg-meta meta-bot">
                <div class="avatar avatar-bot">SAP</div>
                <span class="meta-label meta-label-bot">SAP Assistant</span>
                <span class="meta-time">{time_str}</span>
            </div>''', unsafe_allow_html=True)
            with st.container(border=False):
                st.markdown(turn["content"])

            if turn.get("sources"):
                src_html = "".join(
                    f'<div>📎 {s["file"]} — Page {s["page"]}</div>'
                    for s in turn["sources"])
                st.markdown(f'<div class="source-box">{src_html}</div>', unsafe_allow_html=True)
                # Small inline view buttons
                view_html = ""
                for src in turn["sources"][:4]:
                    view_html += f'''<span style="
                        display:inline-block;margin:4px 4px 0 0;
                        background:{T["input_bg"]};
                        border:1px solid {"#3B5998" if st.session_state.dark_mode else "#BFDBFE"};
                        color:{"#93C5FD" if st.session_state.dark_mode else "#1D4ED8"};
                        border-radius:6px;padding:3px 10px;font-size:11.5px;
                        font-weight:500;font-family:JetBrains Mono,monospace;
                        cursor:default;">↗ p.{src["page"]}</span>'''
                st.markdown(view_html, unsafe_allow_html=True)
                # Real clickable buttons hidden below for functionality
                btn_cols = st.columns(min(len(turn["sources"]), 4))
                for ci, src in enumerate(turn["sources"][:4]):
                    with btn_cols[ci]:
                        if st.button(f"↗ p.{src['page']}", key=f"view_{i}_{ci}",
                                     type="secondary", use_container_width=False):
                            st.session_state.pdf_viewer = {"file": src["file"], "page": src["page"]}
                            st.rerun()

            if turn.get("tcodes"):
                pills = "".join(f'<span class="tcode-pill">{tc}</span>' for tc in turn["tcodes"])
                st.markdown(
                    f'<div style="margin-top:8px;">'
                    f'<span style="font-size:10px;color:{T["text_muted"]};font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">T-CODES &nbsp;</span>'
                    f'{pills}</div>', unsafe_allow_html=True)

            # ── Action bar: Copy + Bookmark + Feedback ──
            is_bookmarked = any(b.get("content") == turn["content"]
                                for b in st.session_state.bookmarks)
            safe_text = turn["content"].replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            st.markdown(f"""
            <div class="action-bar">
                <span class="copy-btn"
                    onclick="navigator.clipboard.writeText(`{safe_text}`);
                             this.innerHTML='✓ Copied';
                             setTimeout(()=>this.innerHTML='⎘ Copy',1500);">
                    ⎘ Copy
                </span>
                <span style="font-size:11px;color:{T['text_muted']};margin-left:8px;font-weight:500;">
                    Was this helpful?
                </span>
            </div>""", unsafe_allow_html=True)
            fb1, fb2, fb3, _ = st.columns([1, 1, 1, 6])
            with fb1:
                if st.button("👍", key=f"up_{i}", type="secondary"):
                    uq = st.session_state.chat_history[i-1]["content"] if i > 0 else ""
                    save_feedback(uq, turn["content"], "positive")
                    st.toast("Thanks for the feedback.")
            with fb2:
                if st.button("👎", key=f"dn_{i}", type="secondary"):
                    uq = st.session_state.chat_history[i-1]["content"] if i > 0 else ""
                    save_feedback(uq, turn["content"], "negative")
                    st.toast("Feedback noted.")
            with fb3:
                bm_label = "★" if is_bookmarked else "☆"
                if st.button(bm_label, key=f"bm_{i}", type="secondary"):
                    if not is_bookmarked:
                        st.session_state.bookmarks.append({
                            "content":  turn["content"],
                            "question": st.session_state.chat_history[i-1]["content"] if i > 0 else "",
                            "saved_at": turn.get("timestamp", ""),
                        })
                        st.toast("Bookmarked!")
                    else:
                        st.session_state.bookmarks = [
                            b for b in st.session_state.bookmarks
                            if b.get("content") != turn["content"]
                        ]
                        st.toast("Bookmark removed.")
                    st.rerun()

            if turn.get("followups"):
                st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{T["text_muted"]};font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">Suggested follow-ups</div>',
                            unsafe_allow_html=True)
                fu_cols = st.columns(len(turn["followups"]))
                for fi, (fcol, fq) in enumerate(zip(fu_cols, turn["followups"])):
                    with fcol:
                        if st.button(fq, key=f"fu_{i}_{fi}", use_container_width=True):
                            st.session_state.pending_followup = fq; st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Keyboard shortcut: press / to focus chat input
    st.markdown("""
    <div class="kb-hint">Press / to focus chat</div>
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === '/' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            var inp = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (inp) inp.focus();
        }
    });
    </script>
    """, unsafe_allow_html=True)

    user_input = st.chat_input(
        f"Ask an SAP question... (responding in {st.session_state.language})",
        disabled=(not st.session_state.vectorstore or not st.session_state.gemini_ready))

    if st.session_state.pending_followup:
        user_input = st.session_state.pending_followup
        st.session_state.pending_followup = None

    if not st.session_state.gemini_ready:
        st.warning("Add your GEMINI_API_KEY to the .env file to enable AI responses.")
    elif not st.session_state.vectorstore:
        st.info("Upload SAP documents in the sidebar to start chatting.")

    if user_input and st.session_state.vectorstore and st.session_state.gemini_ready:
        save_search_history(user_input, "")
        st.session_state.chat_history.append({
            "role": "user", "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        history = [{"role": t["role"], "content": t["content"]}
                   for t in st.session_state.chat_history[:-1]]

        st.markdown('<div class="msg-meta meta-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-user">{user_input}</div>', unsafe_allow_html=True)

        typing_ph = st.empty()
        typing_ph.markdown(f"""
        <div class="msg-meta meta-bot">SAP Assistant</div>
        <div class="typing-wrap">
            <div class="typing-box">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        result = ask_question_stream(
            question=user_input,
            vectorstore=st.session_state.vectorstore,
            k=top_k,
            chat_history=history,
            language=st.session_state.language,
        )

        typing_ph.empty()
        st.markdown('<div class="msg-meta meta-bot">SAP Assistant</div>', unsafe_allow_html=True)
        stream_box = st.empty()
        full_text  = ""

        for chunk in result["stream"]:
            full_text += chunk
            display = re.sub(r'\nFOLLOWUPS:.*$', '', full_text, flags=re.IGNORECASE | re.DOTALL)
            stream_box.markdown(display)

        answer, followups = _split_followups(full_text)
        formatted = format_answer(answer)
        stream_box.markdown(formatted)

        sources = result["sources"]
        tcodes  = extract_tcodes_from_text(formatted)
        answered_flag = "not available" not in formatted.lower()
        save_usage(user_input, answered_flag, sources, st.session_state.language)

        st.session_state.chat_history.append({
            "role":      "assistant",
            "content":   formatted,
            "sources":   sources,
            "tcodes":    tcodes,
            "followups": followups,
            "timestamp": datetime.now().isoformat(),
        })
        st.rerun()


# ── FAQ column ─────────────────────────────────────────────────────────────────
with faq_col:
    st.markdown("""
    <div class="page-header" style="margin-bottom:14px;">
        <h1 style="font-size:14px!important;">Quick Questions</h1>
        <p>Click to ask instantly</p>
    </div>""", unsafe_allow_html=True)

    for faq in FAQ_LIST:
        st.markdown('<div class="faq-btn">', unsafe_allow_html=True)
        if st.button(faq["question"], key=f"faq_{faq['question'][:25]}", use_container_width=True):
            if st.session_state.vectorstore and st.session_state.gemini_ready:
                st.session_state.pending_followup = faq["question"]; st.rerun()
            else:
                st.warning("Upload documents and set API key first.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px;padding:14px;background:{T['tips_bg']};border:1px solid {T['tips_border']};border-radius:8px;">
        <div style="font-size:10px;font-weight:700;color:{T['text_muted']};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Tips</div>
        <div style="font-size:12px;color:{T['text_secondary']};line-height:2;">
            Ask step-by-step instructions<br>
            Mention T-Codes directly<br>
            Click <b>View p.X</b> to open PDF page<br>
            Use follow-up suggestions<br>
            Toggle dark mode in sidebar
        </div>
    </div>""", unsafe_allow_html=True)