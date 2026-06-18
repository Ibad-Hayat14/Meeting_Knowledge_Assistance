"""
app.py
------
Streamlit frontend for the Meeting Knowledge Assistant.

Pages (sidebar navigation):
  📤 Upload Meeting  – file uploader → POST /meetings/process
  📋 My Meetings     – list indexed meetings → GET /meetings
  🔍 Ask a Question  – Q&A interface → POST /ask or /meetings/{id}/ask
"""

import os
import requests
import streamlit as st
from datetime import date

# Helper to read Streamlit secrets safely without throwing error if secrets.toml is missing
def get_secret(key, default=None):
    try:
        val = st.secrets.get(key)
        return val if val is not None else default
    except Exception:
        return default

API_BASE = get_secret("API_BASE_URL", "http://localhost:8000")
API_TOKEN = get_secret("API_APP_TOKEN") or os.getenv("API_APP_TOKEN")

st.set_page_config(
    page_title="Meeting Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – premium futuristic dark glass look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }

    /* Ambient radial glow background */
    .stApp {
        background: radial-gradient(circle at top right, #1a103c 0%, #0a071b 50%, #030209 100%) !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(8, 5, 18, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
    }

    /* Premium glassmorphic cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 12px 40px 0 rgba(139, 92, 246, 0.18);
        transform: translateY(-3px);
    }

    /* Glowing header title */
    .glow-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc 0%, #f472b6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .glow-sub {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 2.2rem;
        line-height: 1.6;
    }

    /* AI Answer container styling */
    .ai-response-box {
        background: rgba(99, 102, 241, 0.04);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 5px solid #6366f1;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
        margin-top: 1.2rem;
        backdrop-filter: blur(8px);
    }
    .ai-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 0.75rem;
        font-size: 1.15rem;
    }
    .ai-text {
        color: #e2e8f0;
        font-size: 1.05rem;
        line-height: 1.75;
    }

    /* Custom badges */
    .badge-indigo {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.4rem;
        transition: all 0.2s;
    }
    .badge-indigo:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
        transform: scale(1.05);
    }
    .badge-cyan {
        display: inline-block;
        background: rgba(6, 182, 212, 0.15);
        color: #67e8f9;
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Glowing health indicator */
    .status-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
    }
    .status-pulse.online {
        background-color: #10b981;
    }
    .status-pulse.offline {
        background-color: #ef4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
    }

    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }

    /* Interactive button overrides */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #9061f9 0%, #5c52eb 100%) !important;
        box-shadow: 0 6px 22px rgba(124, 58, 237, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Form customization */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f3f4f6 !important;
    }
    
    /* Metrics display improvement */
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #c084fc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Password Protection
# ---------------------------------------------------------------------------


def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        expected_pass = get_secret("APP_PASSWORD", "resume2026")
        if st.session_state["password"] == expected_pass:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show login screen
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 2.5rem 2rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 12px rgba(167, 139, 250, 0.5));">🧠</div>
                <h2 style="margin: 0; background: linear-gradient(135deg, #c084fc 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.8rem;">Meeting KA</h2>
                <p style="color: #9ca3af; margin-top: 0.5rem; font-size: 0.95rem; margin-bottom: 2rem;">Secure Resume Project Showcase</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            st.text_input(
                "Access Password",
                type="password",
                key="password",
                placeholder="Enter password to access..."
            )
            submitted = st.form_submit_button("🔓 Unlock Dashboard", use_container_width=True)
            if submitted:
                password_entered()
                if not st.session_state.get("password_correct", False):
                    st.error("😕 Invalid password. Please try again.")
                else:
                    st.rerun()
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_headers():
    headers = {}
    if API_TOKEN:
        headers["X-App-Token"] = API_TOKEN
    return headers


def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", headers=get_headers(), timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API. Make sure the backend server is running."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def api_post(path: str, **kwargs):
    try:
        headers = kwargs.pop("headers", {})
        headers.update(get_headers())
        r = requests.post(f"{API_BASE}{path}", headers=headers, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API. Make sure the backend server is running."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", headers=get_headers(), timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style='padding: 1.5rem 0; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 1.5rem;'>
        <div style='font-size: 2.5rem; filter: drop-shadow(0 0 10px rgba(167, 139, 250, 0.4));'>🧠</div>
        <div style='font-weight: 800; font-size: 1.3rem; letter-spacing: -0.01em; background: linear-gradient(135deg, #c084fc 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Meeting KA</div>
        <div style='font-size: 0.8rem; color: #71717a; font-weight: 500;'>Knowledge Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE = st.sidebar.radio(
    "Navigate",
    ["📤 Upload Meeting", "📋 My Meetings", "🔍 Ask a Question"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# check health in sidebar
health, err = api_get("/")
if health:
    st.sidebar.markdown(
        f"""
        <div class="status-container">
            <span class="status-pulse online"></span>
            <span style="color: #34d399;">API Online <code style="background: rgba(52, 211, 153, 0.1); color: #34d399; font-size: 0.75rem; padding: 0.1rem 0.3rem; border-radius: 4px;">v{health.get('version', '')}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div class="status-container">
            <span class="status-pulse offline"></span>
            <span style="color: #f87171;">API Offline</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page 1 – Upload Meeting
# ---------------------------------------------------------------------------

if PAGE == "📤 Upload Meeting":
    st.markdown("<div class='glow-title'>📤 Upload a Meeting</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-sub'>Upload audio/video recordings. The pipeline extracts audio, transcribes it, extracts key summaries, and indexes it for QA queries.</div>",
        unsafe_allow_html=True,
    )

    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Upload Recording File",
            type=["mp3", "wav", "mp4", "mov", "avi", "m4a", "webm", "ogg"],
            help="Supported: MP3, WAV, MP4, MOV, AVI, M4A, WebM, OGG",
        )
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Meeting Title *", placeholder="e.g. Sprint Review #12")
        with col2:
            meeting_date = st.date_input("Meeting Date *", value=date.today())

        col3, col4 = st.columns(2)
        with col3:
            meeting_id = st.text_input(
                "Meeting ID (Optional)",
                placeholder="Auto-generated if left blank",
            )
        with col4:
            language = st.text_input(
                "Language Hint (Optional)",
                placeholder="e.g. en, ur, es",
                help="ISO code. Leave blank for auto-detection.",
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Process Recording", use_container_width=True)

    if submitted:
        if not uploaded_file:
            st.error("Please upload a file first.")
        elif not title.strip():
            st.error("Please specify a meeting title.")
        else:
            with st.spinner("⏳ Extracting audio, executing transcription & generating summaries..."):
                data = {
                    "title": title.strip(),
                    "date": str(meeting_date),
                }
                if meeting_id.strip():
                    data["meeting_id"] = meeting_id.strip()
                if language.strip():
                    data["language"] = language.strip()

                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result, err = api_post("/meetings/process", data=data, files=files)

            if err:
                st.error(err)
            else:
                st.balloons()
                st.success(f"✅ Processed successfully!")
                
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                
                # Stats grid
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom: 0;">
                            <span style="color: #9ca3af; font-size: 0.85rem; font-weight: 500;">Meeting ID</span>
                            <h3 style="margin: 0.3rem 0; color: #fff; font-size: 1.4rem;">{result['meeting_id']}</h3>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                with col_b:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom: 0;">
                            <span style="color: #9ca3af; font-size: 0.85rem; font-weight: 500;">Indexed Chunks</span>
                            <h3 style="margin: 0.3rem 0; color: #c084fc; font-size: 1.4rem;">{result['chunks_stored']}</h3>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                with col_c:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom: 0;">
                            <span style="color: #9ca3af; font-size: 0.85rem; font-weight: 500;">Date</span>
                            <h3 style="margin: 0.3rem 0; color: #fff; font-size: 1.4rem;">{result['date']}</h3>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

                st.markdown("### 📝 Structured Meeting Output")
                summary = result.get("summary", {})
                st.info(summary.get("summary", ""))

                col_kp, col_ai = st.columns(2)
                with col_kp:
                    st.markdown(
                        """
                        <div class="glass-card" style="height: 100%;">
                            <h4 style="margin-top:0; color:#a78bfa;">🔑 Key Points</h4>
                        """,
                        unsafe_allow_html=True
                    )
                    for kp in summary.get("key_points", []):
                        st.markdown(f"- {kp}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_ai:
                    st.markdown(
                        """
                        <div class="glass-card" style="height: 100%;">
                            <h4 style="margin-top:0; color:#10b981;">✅ Action Items</h4>
                        """,
                        unsafe_allow_html=True
                    )
                    ai_list = summary.get("action_items", [])
                    if ai_list:
                        for ai in ai_list:
                            st.markdown(f"- {ai}")
                    else:
                        st.markdown("_No action items identified._")
                    st.markdown("</div>", unsafe_allow_html=True)

                decisions = summary.get("decisions", [])
                if decisions:
                    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div class="glass-card">
                            <h4 style="margin-top:0; color:#3b82f6;">⚖️ Decisions Made</h4>
                        """,
                        unsafe_allow_html=True
                    )
                    for d in decisions:
                        st.markdown(f"- {d}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("📄 Full Extracted Transcript"):
                    st.text_area("Transcript Raw Content", value=result.get("transcript", ""), height=250, disabled=True)

# ---------------------------------------------------------------------------
# Page 2 – My Meetings
# ---------------------------------------------------------------------------

elif PAGE == "📋 My Meetings":
    st.markdown("<div class='glow-title'>📋 My Meetings</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-sub'>Manage and browse all meeting documents currently indexed in the vector store.</div>",
        unsafe_allow_html=True,
    )

    meetings, err = api_get("/meetings")

    if err:
        st.error(err)
    elif not meetings:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🗂️</div>
                <h4 style="margin: 0; color: #a78bfa;">No meetings indexed yet</h4>
                <p style="color: #71717a; margin-top: 0.5rem; font-size: 0.95rem;">Head over to the <b>📤 Upload Meeting</b> section to index your first recording.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<span class='badge-cyan'>{len(meetings)} Meeting(s) Registered</span>", unsafe_allow_html=True)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        for m in meetings:
            col_content, col_act = st.columns([5, 1])
            with col_content:
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0;">
                        <div class="meeting-title">{m['title']}</div>
                        <div class="meeting-date">📅 {m['date']}</div>
                        <span class="meeting-id">ID: {m['meeting_id']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_act:
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", key=f"del_{m['meeting_id']}"):
                    _, del_err = api_delete(f"/meetings/{m['meeting_id']}")
                    if del_err:
                        st.error(del_err)
                    else:
                        st.success("Deleted!")
                        st.rerun()
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page 3 – Ask a Question
# ---------------------------------------------------------------------------

elif PAGE == "🔍 Ask a Question":
    st.markdown("<div class='glow-title'>🔍 Ask a Question</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glow-sub'>Search and ask questions across all indexed transcripts. The RAG engine fetches source segments and summarizes answers.</div>",
        unsafe_allow_html=True,
    )

    meetings, _ = api_get("/meetings")
    meeting_options = {f"{m['title']} ({m['date']})": m["meeting_id"] for m in (meetings or [])}

    scope = st.selectbox(
        "Select Search Scope",
        ["🌐 Search across all meetings"] + list(meeting_options.keys()),
        help="Select a specific meeting context or search everything.",
    )

    question = st.text_area(
        "Enter Your Question",
        placeholder="e.g. What were the key conclusions regarding the deadline? Who was assigned the task of frontend design?",
        height=100,
    )

    n_context = st.slider("Context chunks to retrieve", min_value=1, max_value=15, value=5)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    ask_btn = st.button("💬 Ask Assistant", use_container_width=True)

    if ask_btn:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            payload = {"question": question.strip(), "n_context": n_context}

            with st.spinner("🔎 Retrieving text segments and running LLM generation..."):
                if scope == "🌐 Search across all meetings":
                    result, err = api_post("/ask", json=payload)
                else:
                    mid = meeting_options[scope]
                    result, err = api_post(f"/meetings/{mid}/ask", json=payload)

            if err:
                st.error(err)
            else:
                st.markdown("---")
                
                # Answer Box
                st.markdown(
                    f"""
                    <div class="ai-response-box">
                        <div class="ai-header">
                            <span>🤖</span> AI Response
                        </div>
                        <div class="ai-text">{result['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                citations = result.get("citations", [])
                if citations:
                    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                    st.markdown("#### 📚 Sourced Citations")
                    
                    pills = "".join(
                        f"<span class='badge-indigo'>📌 {c['title']} — {c['date']} (Chunk {c['chunk_index']})</span>"
                        for c in citations
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                    
                    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

                    with st.expander("View Retrieved Transcript Excerpts"):
                        for i, c in enumerate(citations, 1):
                            st.markdown(
                                f"**[{i}] {c['title']} ({c['date']}) — chunk {c['chunk_index']}**"
                            )
                            st.caption(c["text"])
                            st.markdown("---")
