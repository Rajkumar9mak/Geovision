"""
GeoInsight-RigX (eRTMAC-NWIS)
Page 5: Well OCR Intelligence
Completely isolated Well PDF Intelligence System:
PDF → Real OCR → Well Information Extraction → Important Text → Simple Summary → Well Q&A Chatbot
Guarantees 100% document isolation: zero contamination from previous or global documents.
"""

import streamlit as st
from core_engine.ui_theme import inject_industrial_theme_css
from core_engine.well_pdf_ocr import WellPDFOCREngine
from core_engine.well_extractor import WellExtractorEngine
from core_engine.well_summary import WellSummaryEngine
from core_engine.well_chatbot import WellChatbotEngine

# Ensure persistent dark industrial theme
inject_industrial_theme_css()

# ==============================================================================
# PAGE HEADER
# ==============================================================================
col_head, col_badge = st.columns([3.2, 1.3])
with col_head:
    st.markdown("# 📄 Well OCR Intelligence")
    st.markdown(
        "<div style='color:#8B949E; font-size:0.95rem; margin-top:-8px; margin-bottom:12px;'>"
        "Upload a well report, extract the important information, understand it in simple words, and ask questions about the report."
        "</div>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px solid #238636; border-radius:18px; padding:6px 14px; text-align:center; margin-top:8px;">
            <span style="color:#3FB950; font-weight:600; font-size:0.85rem;">🟢 Well Document Assistant Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:10px 0 20px 0;'>", unsafe_allow_html=True)

# Initialize Session State Variables
if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None
if "well_pages_data" not in st.session_state:
    st.session_state.well_pages_data = None
if "well_info" not in st.session_state:
    st.session_state.well_info = None
if "well_key_params" not in st.session_state:
    st.session_state.well_key_params = None
if "well_valuable_info" not in st.session_state:
    st.session_state.well_valuable_info = None
if "well_extracts" not in st.session_state:
    st.session_state.well_extracts = None
if "well_summary_text" not in st.session_state:
    st.session_state.well_summary_text = None
if "well_chat_messages" not in st.session_state:
    st.session_state.well_chat_messages = []
if "last_query_debug" not in st.session_state:
    st.session_state.last_query_debug = None

# Cached Engine Instances
@st.cache_resource
def get_ocr_engine():
    return WellPDFOCREngine()

@st.cache_resource
def get_chatbot_engine():
    return WellChatbotEngine()

ocr_engine = get_ocr_engine()
chatbot_engine = get_chatbot_engine()

# ==============================================================================
# SECTION 1 — UPLOAD WELL REPORT
# ==============================================================================
st.markdown("### 📄 Upload Well Report")
st.markdown(
    "<div style='color:#8B949E; font-size:0.9rem; margin-bottom:10px;'>"
    "Upload a drilling, geological, completion, or well operation PDF."
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a Well PDF",
    type=["pdf"],
    accept_multiple_files=False,
    key="well_intelligence_uploader",
    help="Upload a single PDF document. Digital and scanned pages are extracted automatically.",
    label_visibility="collapsed",
)

# Handle file change & Absolute Document Isolation
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_doc_id = ocr_engine.calculate_document_id(file_bytes)

    # 1. Reset everything when a new/different PDF is uploaded
    if st.session_state.active_document_id != current_doc_id:
        st.session_state.active_document_id = current_doc_id
        st.session_state.well_pages_data = None
        st.session_state.well_info = None
        st.session_state.well_key_params = None
        st.session_state.well_valuable_info = None
        st.session_state.well_extracts = None
        st.session_state.well_summary_text = None
        st.session_state.well_chat_messages = []
        st.session_state.last_query_debug = None

    # Inspect the PDF
    is_valid, err_msg, total_pages, doc_id, file_size_kb = ocr_engine.inspect_pdf(file_bytes, uploaded_file.name)

    if not is_valid:
        st.error(f"⚠️ {err_msg}")
    else:
        is_processed = st.session_state.well_pages_data is not None

        # Document Status Card
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:14px 18px; margin:12px 0 16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Current Document</div>
                        <div style="color:#F0F6FC; font-size:1.05rem; font-weight:700; margin-top:2px;">📄 {uploaded_file.name}</div>
                        <div style="color:#8B949E; font-size:0.82rem; margin-top:4px;">
                            Pages: <b style="color:#58A6FF;">{total_pages}</b> &bull; 
                            Size: <b style="color:#E6EDF3;">{file_size_kb} KB</b> &bull; 
                            Document ID: <code style="color:#E28743; font-size:0.8rem;">{doc_id}</code>
                        </div>
                    </div>
                    <div>
                        <span style="background-color:{'#1F6FEB22' if not is_processed else '#23863622'}; color:{'#E28743' if not is_processed else '#3FB950'}; border:1px solid {'#E28743' if not is_processed else '#238636'}; border-radius:6px; padding:6px 12px; font-weight:600; font-size:0.85rem;">
                            {'🟡 Ready to process' if not is_processed else '🟢 Processed & Ready'}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Process Button if not yet processed
        if not is_processed:
            col_proc, _ = st.columns([1.2, 2.8])
            with col_proc:
                process_clicked = st.button("🔍 Process Well Report", type="primary", use_container_width=True)

            if process_clicked:
                progress_text = st.empty()
                progress_bar = st.progress(0.0)

                def on_page_progress(current, total, status_msg):
                    progress_text.markdown(f"**📖 Reading well report... Page {current} / {total}**")
                    progress_bar.progress(min(0.90, current / max(total, 1)))

                try:
                    pages_data = ocr_engine.process_pdf(file_bytes, uploaded_file.name, progress_callback=on_page_progress)
                    progress_text.markdown("**🔍 Extracting information & checking OCR quality...**")
                    progress_bar.progress(0.95)

                    # Extract information strictly from current pages_data
                    w_info = WellExtractorEngine.extract_well_information(pages_data, uploaded_file.name)
                    k_params = WellExtractorEngine.extract_key_parameters(pages_data)
                    val_info = WellExtractorEngine.extract_valuable_information(pages_data)
                    extracts = WellExtractorEngine.extract_important_passages(pages_data)

                    # Index strictly into private collection well_ocr_{current_doc_id}
                    chatbot_engine.index_document(pages_data, uploaded_file.name, current_doc_id)

                    # Store in session state
                    st.session_state.well_pages_data = pages_data
                    st.session_state.well_info = w_info
                    st.session_state.well_key_params = k_params
                    st.session_state.well_valuable_info = val_info
                    st.session_state.well_extracts = extracts

                    progress_text.markdown("**✅ Well report processed successfully**")
                    progress_bar.progress(1.0)
                    st.rerun()
                except Exception as ex:
                    st.error("⚠️ An unexpected error occurred while processing the PDF. Please ensure the file is not corrupted.")

# ==============================================================================
# PROCESSED DOCUMENT DASHBOARD
# ==============================================================================
if st.session_state.well_pages_data is not None:
    pages_data = st.session_state.well_pages_data
    well_info = st.session_state.well_info
    key_params = st.session_state.well_key_params
    valuable_info = st.session_state.well_valuable_info
    extracts = st.session_state.well_extracts
    active_filename = uploaded_file.name if uploaded_file else "Well_Report.pdf"
    doc_id = st.session_state.active_document_id

    # Check for any low confidence OCR pages
    low_conf_pages = [p["page_number"] for p in pages_data if p.get("low_confidence")]
    if low_conf_pages:
        st.warning(f"⚠️ Low OCR confidence on Page(s) {', '.join(str(p) for p in low_conf_pages)}. Text may be degraded in scanned imagery.")

    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:16px 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 3 — WELL INFORMATION
    # --------------------------------------------------------------------------
    st.markdown("## 🛢️ Well Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Well Name</div>
                <b style="color:#F0F6FC; font-size:0.95rem;">{well_info.get('well_name', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Total Well Depth</div>
                <b style="color:#00E5FF; font-size:0.95rem;">{well_info.get('total_depth', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Rig</div>
                <b style="color:#E6EDF3; font-size:0.95rem;">{well_info.get('rig_name', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Operator</div>
                <b style="color:#E28743; font-size:0.95rem;">{well_info.get('operator', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Well Type</div>
                <b style="color:#E6EDF3; font-size:0.95rem;">{well_info.get('well_type', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Spud Date</div>
                <b style="color:#E6EDF3; font-size:0.95rem;">{well_info.get('spud_date', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Field</div>
                <b style="color:#7EE787; font-size:0.95rem;">{well_info.get('field', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Location</div>
                <b style="color:#E6EDF3; font-size:0.95rem;">{well_info.get('location', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px;">
                <div style="color:#8B949E; font-size:0.75rem; text-transform:uppercase;">Well ID / UWI</div>
                <b style="color:#E6EDF3; font-size:0.95rem;">{well_info.get('uwi', 'Not found in report')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 4 — KEY WELL PARAMETERS
    # --------------------------------------------------------------------------
    if key_params:
        st.markdown("## 📊 Key Well Parameters")
        p_cols = st.columns(min(len(key_params), 6))
        for idx, (p_name, p_val) in enumerate(key_params.items()):
            c = p_cols[idx % len(p_cols)]
            with c:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 12px; margin-bottom:8px; text-align:center;">
                        <div style="color:#8B949E; font-size:0.72rem; text-transform:uppercase;">{p_name}</div>
                        <div style="color:#58A6FF; font-weight:700; font-size:0.92rem; margin-top:2px;">{p_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:16px 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 5 — VALUABLE EXTRACTED TEXT
    # --------------------------------------------------------------------------
    st.markdown("## 🔎 Valuable Information Found")
    v_tab1, v_tab2, v_tab3, v_tab4, v_tab5 = st.tabs([
        "📍 Important Depths",
        "⚠️ Drilling Problems",
        "🪨 Geological Findings",
        "🛠️ Actions Taken",
        "⚠️ Hazards",
    ])

    with v_tab1:
        depths_data = valuable_info.get("depths", [])
        if depths_data:
            for item in depths_data:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:3px solid #58A6FF; border-radius:6px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <div><b style="color:#58A6FF;">{item['depth']}</b> &bull; <span style="color:#E6EDF3; font-size:0.85rem;">{item['description']}</span></div>
                        <span style="color:#8B949E; font-size:0.75rem; font-family:monospace;">Page {item['page']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No depth records found in the uploaded document.")

    with v_tab2:
        problems_data = valuable_info.get("problems", [])
        if problems_data:
            for item in problems_data:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:3px solid #F85149; border-radius:6px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="color:#E6EDF3; font-size:0.88rem;">⚠️ {item['description']}</div>
                        <span style="color:#8B949E; font-size:0.75rem; font-family:monospace; margin-left:12px; white-space:nowrap;">Page {item['page']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No drilling problems or loss events recorded in this document.")

    with v_tab3:
        geology_data = valuable_info.get("geology", [])
        if geology_data:
            for item in geology_data:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:3px solid #7EE787; border-radius:6px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="color:#E6EDF3; font-size:0.88rem;">🪨 {item['description']}</div>
                        <span style="color:#8B949E; font-size:0.75rem; font-family:monospace; margin-left:12px; white-space:nowrap;">Page {item['page']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No geological formation descriptions found in this document.")

    with v_tab4:
        actions_data = valuable_info.get("actions", [])
        if actions_data:
            for item in actions_data:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:3px solid #E28743; border-radius:6px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="color:#E6EDF3; font-size:0.88rem;">🛠️ {item['description']}</div>
                        <span style="color:#8B949E; font-size:0.75rem; font-family:monospace; margin-left:12px; white-space:nowrap;">Page {item['page']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No operational mitigation actions recorded in this document.")

    with v_tab5:
        hazards_data = valuable_info.get("hazards", [])
        if hazards_data:
            for item in hazards_data:
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:3px solid #E3B341; border-radius:6px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="color:#E6EDF3; font-size:0.88rem;"><b style="color:#E3B341;">{item.get('hazard', 'Hazard')}:</b> {item['description']}</div>
                        <span style="color:#8B949E; font-size:0.75rem; font-family:monospace; margin-left:12px; white-space:nowrap;">Page {item['page']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No explicit hazard alerts identified in this document.")

    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:16px 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 6 — IMPORTANT REPORT EXTRACTS (CLEAN & EXPANDABLE)
    # --------------------------------------------------------------------------
    st.markdown("## 📄 Important Report Extracts")
    if extracts:
        for ext in extracts:
            src_label = "PDF Text" if ext.get("source_type") == "pdf_text" else "OCR"
            q_score = ext.get("quality_score", 90)
            p_num = ext.get("page", 1)
            passage_text = ext.get("clean_passage", ext.get("excerpt", ""))
            full_content = ext.get("full_text", ext.get("text", passage_text))
            st.markdown(
                f"""
                <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:12px 14px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <b style="color:#E28743; font-size:0.9rem;">📄 Page {p_num}</b>
                        <span style="color:#8B949E; font-size:0.75rem;">Source: <b style="color:#58A6FF;">{src_label}</b> &bull; Quality: <b style="color:#7EE787;">{q_score}%</b></span>
                    </div>
                    <div style="color:#E6EDF3; font-size:0.86rem; line-height:1.5;">
                        "{passage_text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"▼ Show full page text (Page {p_num})", expanded=False):
                st.code(full_content, language="text")


    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:16px 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 7 — SIMPLE SUMMARY (CURRENT PDF ONLY)
    # --------------------------------------------------------------------------
    col_sum_btn, col_sum_lbl = st.columns([1.3, 3])
    with col_sum_btn:
        sum_clicked = st.button("📝 Summarize Well Report", type="primary", use_container_width=True)
    with col_sum_lbl:
        st.markdown(
            "<div style='color:#8B949E; font-size:0.85rem; padding-top:8px;'>"
            "Generates a clear summary based strictly on the currently uploaded document."
            "</div>",
            unsafe_allow_html=True,
        )

    if sum_clicked:
        with st.spinner("Writing simple summary for this document..."):
            summary_md = WellSummaryEngine.generate_summary(
                well_info=well_info,
                key_params=key_params,
                valuable_info=valuable_info,
                pages_data=pages_data,
                filename=active_filename,
            )
            st.session_state.well_summary_text = summary_md

    if st.session_state.well_summary_text:
        st.markdown(
            f"""
            <div style="background-color:#161B22; border:1px solid #30363D; border-left:4px solid #58A6FF; border-radius:8px; padding:18px 22px; margin:16px 0;">
                {st.session_state.well_summary_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:20px 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 8 — CHATBOT (STRICT DOCUMENT GROUNDING)
    # --------------------------------------------------------------------------
    st.markdown("## 💬 Ask About This Well")
    st.markdown(
        "<div style='color:#8B949E; font-size:0.95rem; margin-top:-6px; margin-bottom:14px;'>"
        "Ask questions about the uploaded well report."
        "</div>",
        unsafe_allow_html=True,
    )

    # Suggested Questions Buttons
    sq_col1, sq_col2, sq_col3, sq_col4 = st.columns(4)
    sq_col5, sq_col6, sq_col7, sq_col8 = st.columns(4)

    active_prompt = None
    with sq_col1:
        if st.button("What is the total well depth?", use_container_width=True, key="wq_1"):
            active_prompt = "What is the total well depth?"
    with sq_col2:
        if st.button("What happened at 3290 m?", use_container_width=True, key="wq_2"):
            active_prompt = "What happened at 3290 m?"
    with sq_col3:
        if st.button("What were main drilling problems?", use_container_width=True, key="wq_3"):
            active_prompt = "What were the main drilling problems?"
    with sq_col4:
        if st.button("What formations were encountered?", use_container_width=True, key="wq_4"):
            active_prompt = "What formations were encountered?"
    with sq_col5:
        if st.button("What caused the mud loss?", use_container_width=True, key="wq_5"):
            active_prompt = "What caused the mud loss?"
    with sq_col6:
        if st.button("What actions were taken?", use_container_width=True, key="wq_6"):
            active_prompt = "What actions were taken?"
    with sq_col7:
        if st.button("What are the main hazards?", use_container_width=True, key="wq_7"):
            active_prompt = "What are the main hazards?"
    with sq_col8:
        if st.button("Give me important depths", use_container_width=True, key="wq_8"):
            active_prompt = "Give me the important depths."

    # Render Conversation Messages
    for msg in st.session_state.well_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("raw_context") and msg.get("pages"):
                pages_str = ", ".join(str(p) for p in msg["pages"])
                with st.expander("▼ 🔎 Show source text", expanded=False):
                    st.markdown(
                        f"""
                        <div style="background-color:#161B22; border-left:3px solid #E28743; padding:10px 14px; font-size:0.85rem; color:#C9D1D9; font-style:italic; line-height:1.5;">
                            "{msg['raw_context']}"
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#8B949E;">
                            <b>Document:</b> {msg.get('source_file', active_filename)} &bull; <b>Page(s):</b> {pages_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Chat Input
    user_query = st.chat_input("Ask something about this well report...")
    final_query = active_prompt or user_query

    if final_query:
        st.session_state.well_chat_messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        full_doc_text = "\n".join(p["text"] for p in pages_data)

        with st.chat_message("assistant"):
            with st.spinner("Finding answer in current document..."):
                resp = chatbot_engine.answer_query(
                    query=final_query,
                    chat_history=st.session_state.well_chat_messages,
                    filename=active_filename,
                    document_id=doc_id,
                    well_info=well_info,
                    key_params=key_params,
                    full_text=full_doc_text,
                )

            st.markdown(resp["answer"])
            st.session_state.last_query_debug = resp.get("debug_info")

            if resp.get("raw_context") and resp.get("pages"):
                pages_str = ", ".join(str(p) for p in resp["pages"])
                with st.expander("▼ 🔎 Show source text", expanded=False):
                    st.markdown(
                        f"""
                        <div style="background-color:#161B22; border-left:3px solid #E28743; padding:10px 14px; font-size:0.85rem; color:#C9D1D9; font-style:italic; line-height:1.5;">
                            "{resp['raw_context']}"
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#8B949E;">
                            <b>Document:</b> {resp['source_file']} &bull; <b>Page(s):</b> {pages_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.session_state.well_chat_messages.append({
                "role": "assistant",
                "content": resp["answer"],
                "source_file": resp["source_file"],
                "pages": resp.get("pages", []),
                "raw_context": resp.get("raw_context"),
            })
            st.rerun()

    # --------------------------------------------------------------------------
    # SECTION 9 — DOCUMENT DEBUG INFORMATION (EXPANDABLE)
    # --------------------------------------------------------------------------
    st.markdown("<hr style='border:none; border-top:1px solid #30363D; margin:24px 0 12px 0;'>", unsafe_allow_html=True)
    with st.expander("🔧 Document Debug Information", expanded=False):
        n_ocr = sum(1 for p in pages_data if p.get("source_type") == "ocr")
        n_text = sum(1 for p in pages_data if p.get("source_type") == "pdf_text")
        col_dbg1, col_dbg2 = st.columns(2)
        with col_dbg1:
            st.markdown(f"- **Current Filename:** `{active_filename}`")
            st.markdown(f"- **Current Document ID:** `{doc_id}`")
            st.markdown(f"- **Total Page Count:** `{len(pages_data)}`")
            st.markdown(f"- **Digital Text Pages:** `{n_text}`")
            st.markdown(f"- **OCR Scanned Pages:** `{n_ocr}`")
        with col_dbg2:
            st.markdown(f"- **Vector Collection Name:** `well_ocr_{doc_id}`")
            st.markdown(f"- **Document ID Filter:** `where document_id == '{doc_id}'`")
            if st.session_state.last_query_debug:
                dbg = st.session_state.last_query_debug
                st.markdown(f"- **Last Query Retrieved Pages:** `{dbg.get('retrieved_pages')}`")
                st.markdown(f"- **Distances:** `{dbg.get('distances')}`")
            else:
                st.markdown("- **Last Query Status:** No questions asked yet.")

else:
    # Friendly empty state
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px dashed #30363D; border-radius:8px; padding:36px; text-align:center; margin-top:20px;">
            <div style="font-size:2.2rem; margin-bottom:8px;">🛢️</div>
            <div style="color:#F0F6FC; font-size:1.1rem; font-weight:600;">No Well Report Processed</div>
            <div style="color:#8B949E; font-size:0.88rem; max-width:480px; margin:8px auto 0 auto; line-height:1.5;">
                Upload a drilling report, daily drilling report, completion report, or geological PDF above and click <b>Process Well Report</b> to extract well information, view key parameters, and ask questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
