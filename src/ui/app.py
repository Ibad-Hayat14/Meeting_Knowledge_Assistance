"""
app.py
------
Streamlit frontend for the Meeting Knowledge Assistant.

Pages (sidebar navigation):
  📤 Upload Meeting  – file uploader → POST /meetings/process
  📋 My Meetings     – list indexed meetings → GET /meetings
  🔍 Ask a Question  – Q&A interface → POST /ask or /meetings/{id}/ask
"""

import requests
import streamlit as st
from datetime import date

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Meeting Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – premium dark glass look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: box-shadow 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 0 20px rgba(120, 80, 255, 0.25);
    }

    /* Hero header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #a0a0c0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Answer block */
    .answer-box {
        background: rgba(96, 165, 250, 0.08);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #dde6ff;
    }

    /* Citation pill */
    .citation {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.82rem;
        margin: 0.25rem 0.25rem 0 0;
        color: #c4b5fd;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08); }

    /* Metric badge */
    .badge {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }

    /* Button overrides */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.6rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API. Make sure `uvicorn src.api.main:app --reload` is running."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API. Make sure `uvicorn src.api.main:app --reload` is running."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API."
    except requests.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style='padding: 1rem 0 1.5rem 0; text-align: center;'>
        <div style='font-size:2rem;'>🧠</div>
        <div style='font-weight:700; font-size:1.1rem; color:#c4b5fd;'>Meeting KA</div>
        <div style='font-size:0.75rem; color:#6b6b8a;'>Knowledge Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE = st.sidebar.radio(
    "Navigate",
    ["📤 Upload Meeting", "📋 My Meetings", "🔍 Ask a Question"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.78rem; color:#6b6b8a;'>API: <code>localhost:8000</code></div>",
    unsafe_allow_html=True,
)

# check health in sidebar
health, err = api_get("/")
if health:
    st.sidebar.success(f"API online ✓  v{health.get('version', '')}")
else:
    st.sidebar.error("API offline")

# ---------------------------------------------------------------------------
# Page 1 – Upload Meeting
# ---------------------------------------------------------------------------

if PAGE == "📤 Upload Meeting":
    st.markdown("<div class='hero-title'>📤 Upload a Meeting</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Upload an audio/video recording to transcribe, summarise and index it for Q&amp;A.</div>",
        unsafe_allow_html=True,
    )

    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Recording file",
            type=["mp3", "wav", "mp4", "mov", "avi", "m4a", "webm", "ogg"],
            help="Supported: MP3, WAV, MP4, MOV, AVI, M4A, WebM, OGG",
        )
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Meeting title *", placeholder="e.g. Sprint Review #12")
        with col2:
            meeting_date = st.date_input("Meeting date *", value=date.today())

        col3, col4 = st.columns(2)
        with col3:
            meeting_id = st.text_input(
                "Meeting ID (optional)",
                placeholder="Auto-generated if blank",
            )
        with col4:
            language = st.text_input(
                "Language code (optional)",
                placeholder="e.g. en, ur, de",
                help="ISO-639-1 code. Leave blank for auto-detect.",
            )

        submitted = st.form_submit_button("🚀 Process Meeting", use_container_width=True)

    if submitted:
        if not uploaded_file:
            st.error("Please upload a file.")
        elif not title.strip():
            st.error("Please enter a meeting title.")
        else:
            with st.spinner("⏳ Processing… transcription + summarisation + indexing. This may take a minute."):
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
                st.success(f"✅ Meeting **{result['title']}** processed successfully!")
                st.markdown("---")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Meeting ID", result["meeting_id"])
                col_b.metric("Chunks indexed", result["chunks_stored"])
                col_c.metric("Date", result["date"])

                st.markdown("### 📝 Summary")
                summary = result.get("summary", {})
                st.info(summary.get("summary", ""))

                col_kp, col_ai = st.columns(2)
                with col_kp:
                    st.markdown("**🔑 Key Points**")
                    for kp in summary.get("key_points", []):
                        st.markdown(f"- {kp}")

                with col_ai:
                    st.markdown("**✅ Action Items**")
                    ai_list = summary.get("action_items", [])
                    if ai_list:
                        for ai in ai_list:
                            st.markdown(f"- {ai}")
                    else:
                        st.markdown("_No action items identified._")

                decisions = summary.get("decisions", [])
                if decisions:
                    st.markdown("**⚖️ Decisions**")
                    for d in decisions:
                        st.markdown(f"- {d}")

                with st.expander("📄 Full Transcript"):
                    st.text_area("Transcript", value=result.get("transcript", ""), height=300, disabled=True)

# ---------------------------------------------------------------------------
# Page 2 – My Meetings
# ---------------------------------------------------------------------------

elif PAGE == "📋 My Meetings":
    st.markdown("<div class='hero-title'>📋 My Meetings</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>All meetings currently indexed in the knowledge base.</div>",
        unsafe_allow_html=True,
    )

    meetings, err = api_get("/meetings")

    if err:
        st.error(err)
    elif not meetings:
        st.markdown(
            "<div class='card'><b>No meetings indexed yet.</b><br>Use the <b>📤 Upload Meeting</b> page to add your first one.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<span class='badge'>{len(meetings)} meeting(s)</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for m in meetings:
            with st.container():
                st.markdown(
                    f"""
                    <div class='card'>
                        <b style='font-size:1.1rem;'>{m['title']}</b>
                        <span style='color:#6b6b8a; font-size:0.85rem; margin-left:0.8rem;'>{m['date']}</span><br>
                        <code style='font-size:0.8rem; color:#a78bfa;'>{m['meeting_id']}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_del, _ = st.columns([1, 5])
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{m['meeting_id']}"):
                        _, del_err = api_delete(f"/meetings/{m['meeting_id']}")
                        if del_err:
                            st.error(del_err)
                        else:
                            st.success(f"Deleted **{m['title']}**.")
                            st.rerun()

# ---------------------------------------------------------------------------
# Page 3 – Ask a Question
# ---------------------------------------------------------------------------

elif PAGE == "🔍 Ask a Question":
    st.markdown("<div class='hero-title'>🔍 Ask a Question</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Ask anything about your indexed meetings. The AI will retrieve the most relevant context and generate a sourced answer.</div>",
        unsafe_allow_html=True,
    )

    # Fetch available meetings for optional scoping
    meetings, _ = api_get("/meetings")
    meeting_options = {f"{m['title']} ({m['date']})": m["meeting_id"] for m in (meetings or [])}

    scope = st.selectbox(
        "Scope (optional)",
        ["🌐 All meetings"] + list(meeting_options.keys()),
        help="Restrict the search to a single meeting or search across all.",
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What did we decide about the launch date? Who is responsible for the budget?",
        height=100,
    )

    n_context = st.slider("Context chunks to retrieve", min_value=1, max_value=15, value=5)

    ask_btn = st.button("💬 Ask", use_container_width=True)

    if ask_btn:
        if not question.strip():
            st.error("Please type a question.")
        else:
            payload = {"question": question.strip(), "n_context": n_context}

            with st.spinner("🔎 Searching transcripts and generating answer…"):
                if scope == "🌐 All meetings":
                    result, err = api_post("/ask", json=payload)
                else:
                    mid = meeting_options[scope]
                    result, err = api_post(f"/meetings/{mid}/ask", json=payload)

            if err:
                st.error(err)
            else:
                st.markdown("---")
                st.markdown("### 💡 Answer")
                st.markdown(
                    f"<div class='answer-box'>{result['answer']}</div>",
                    unsafe_allow_html=True,
                )

                citations = result.get("citations", [])
                if citations:
                    st.markdown("#### 📚 Sources")
                    pills = "".join(
                        f"<span class='citation'>📌 {c['title']} — {c['date']}</span>"
                        for c in citations
                    )
                    st.markdown(pills, unsafe_allow_html=True)

                    with st.expander("View retrieved transcript excerpts"):
                        for i, c in enumerate(citations, 1):
                            st.markdown(
                                f"**[{i}] {c['title']} ({c['date']}) — chunk {c['chunk_index']}**"
                            )
                            st.caption(c["text"])
                            st.markdown("---")
