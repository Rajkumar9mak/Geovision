"""
GeoInsight-RigX (eRTMAC-NWIS)
Page 4: Subsurface Knowledge Retrieval Center
Conversational RAG interface powered by LangChain, all-MiniLM-L6-v2 embeddings,
and local ChromaDB for querying historical offset well reports and geohazard archives.
"""

import streamlit as st
from pathlib import Path
from core_engine.rag_pipeline import query_historical_reports, build_or_load_vector_store
from core_engine.ui_theme import inject_industrial_theme_css

# Ensure persistent dark industrial theme and Z-index isolation across page switches
inject_industrial_theme_css()

st.markdown("## 🧠 Subsurface Knowledge Retrieval Center")
st.caption("Local Vector Search (ChromaDB) • all-MiniLM-L6-v2 Embeddings • Open-Source Historical PDF Archives")

# ==============================================================================
# EDGE-CASE FALLBACK: VERIFY TARGET WELL SELECTION FROM GEOSPATIAL HUB
# ==============================================================================
offset_state = st.session_state.get("selected_offset_wells", None)
if offset_state is None:
    st.warning("⚠️ Awaiting Target Well configuration from the Geospatial Hub.")
    st.info(
        "No active spatial offset baseline is loaded in the session state. "
        "Please initiate the offset search from Page 1, or initialize the default baseline below to enable historical knowledge indexing."
    )
    col_nav1, col_nav2 = st.columns([1, 1.5])
    with col_nav1:
        if st.button("🗺️ Go to Geospatial Hub", key="btn_goto_geohub_p4", use_container_width=True):
            st.switch_page("pages/1_🌍_Geospatial_Hub.py")
    with col_nav2:
        if st.button("⚡ Initialize Default Baseline (Mumbai High Offshore)", key="btn_init_def_p4", use_container_width=True):
            from core_engine.database_manager import SpatialDBManager
            from core_engine.visualization_engine import generate_sample_wellbore_data
            if "db_manager" not in st.session_state:
                st.session_state.db_manager = SpatialDBManager()
            df_off = st.session_state.db_manager.get_nearest_offset_wells(19.4215, 71.3510, radius_km=50.0, limit=3)
            st.session_state.offset_wells = df_off
            primary = df_off.iloc[0].to_dict()
            p_name = str(primary.get("well_name", "RIGX-ALPHA-01"))
            p_td = float(primary.get("total_depth_m", 3500.0))
            b_df = generate_sample_wellbore_data(well_name=p_name, total_depth_m=p_td, step_m=1.0)
            b_df = b_df[(b_df["DEPTH_M"] >= 3180.0) & (b_df["DEPTH_M"] <= 3450.0)].copy()
            b_df.rename(columns={"DEPTH_M": "Depth", "GR": "Gamma_Ray"}, inplace=True)
            b_df["RPM"] = np.clip(115.0 + np.sin(b_df["Depth"] / 20.0) * 12.0, 90.0, 140.0)
            st.session_state.selected_offset_wells = {
                "dataframe": df_off,
                "primary_well": primary,
                "primary_well_name": p_name,
                "primary_well_uwi": primary.get("uwi", "IND-OFF-MH-101"),
                "primary_well_distance_km": float(primary.get("distance_km", 0.0)),
                "primary_well_operator": primary.get("operator", "ONGC / GeoInsight"),
                "primary_well_td_m": p_td,
                "baseline_curves_df": b_df,
                "numerical_averages": {
                    "avg_rop": round(float(b_df["ROP"].mean()), 1),
                    "avg_wob": round(float(b_df["WOB"].mean()), 1),
                    "avg_rpm": round(float(b_df["RPM"].mean()), 0),
                    "avg_gr": round(float(b_df["Gamma_Ray"].mean()), 1),
                },
                "target_coordinates": {"lat": 19.4215, "lon": 71.3510, "radius_km": 50.0},
            }
            st.rerun()
    st.stop()

# Top Intelligence Metadata Bar
col_m1, col_m2, col_m3 = st.columns([1.5, 1, 1])
with col_m1:
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; font-size:0.82rem;">
            <div style="color:#8B949E;">Active Vector Store</div>
            <b style="color:#E28743;">ChromaDB (Local Persistent)</b> • <span style="color:#4CAF50;">Indexed & Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; font-size:0.82rem;">
            <div style="color:#8B949E;">Embedding Model</div>
            <b style="color:#58A6FF;">all-MiniLM-L6-v2</b> <span style="color:#8B949E;">(384-dim)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        """
        <div style="background-color:#161B22; border:1px solid #30363D; border-radius:6px; padding:10px 14px; font-size:0.82rem;">
            <div style="color:#8B949E;">Indexed Literature</div>
            <b style="color:#F0F6FC;">3 Technical PDFs</b> • <span style="color:#8B949E;">1000ch Chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# Quick Query Shortcuts
st.markdown("**⚡ Recommended Subsurface Inquiries:**")
q_col1, q_col2, q_col3 = st.columns(3)

quick_prompt = None
with q_col1:
    if st.button("Extraction failure notes in adjacent sandstone", use_container_width=True, key="qp1"):
        quick_prompt = "Show extraction failure notes in the adjacent sandstone layers"
with q_col2:
    if st.button("Fractured carbonate mud loss at 3290m", use_container_width=True, key="qp2"):
        quick_prompt = "What caused the severe mud loss at 3290m in the fractured carbonate and what was the mitigation?"
with q_col3:
    if st.button("Hole cleaning & pack-off in reactive shale", use_container_width=True, key="qp3"):
        quick_prompt = "What are the recommended hole cleaning and RPM practices to avoid pack-off in reactive smectite shale?"

# Initialize conversation history in session state
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Subsurface Knowledge Retrieval System Active.**\n\n"
                "I am equipped with semantically indexed offset well engineering reports, geological end-of-well summaries, "
                "and historical geohazard analyses (Volve Field, Gulf Offshore, HPHT). Ask me any question regarding historical "
                "extraction failures, lost circulation, stuck pipe events, or formation mitigation directives."
            ),
            "sources": [],
        }
    ]

# Render existing chat history
for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("---")
            st.markdown("##### 📚 Verified Historical Source References (Transparent RAG Layer):")
            for idx, src in enumerate(msg["sources"], 1):
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:4px solid #E28743; border-radius:6px; padding:12px 14px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.8rem;">
                            <span style="color:#E28743; font-weight:700;">📄 Source File: <b>{src['source_file']}</b></span>
                            <span style="color:#58A6FF; font-weight:600;">📑 Page: <b>{src['page_number']}</b></span>
                        </div>
                        <div style="font-size:0.88rem; color:#E6EDF3; line-height:1.5; font-style:italic;">
                            "{src['content']}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Chat Input Handler
user_input = st.chat_input("Enter subsurface or hazard query (e.g., 'Show extraction failure notes in the adjacent sandstone layers')...")
active_query = quick_prompt or user_input

if active_query:
    # 1. Display User Message
    st.session_state.rag_messages.append({"role": "user", "content": active_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Retrieve Relevant Chunks via SubsurfaceRAGPipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB semantic vector store with all-MiniLM-L6-v2..."):
            retrieved_chunks = query_historical_reports(active_query, k=3)

        if not retrieved_chunks:
            response_text = "No direct semantic matches were found in the historical report archives. Try rephrasing your query or selecting one of the suggested topics above."
            st.markdown(response_text)
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text, "sources": []})
        else:
            # Construct synthetic summary answer from top chunk
            top_source = retrieved_chunks[0]
            summary_intro = (
                f"Based on historical offset well records in **{top_source['source_file']}** (Page {top_source['page_number']}), "
                f"here are the verified extraction findings and technical directives:"
            )
            st.markdown(summary_intro)

            st.markdown("---")
            st.markdown("##### 📚 Verified Historical Source References (Transparent RAG Layer):")
            for idx, src in enumerate(retrieved_chunks, 1):
                st.markdown(
                    f"""
                    <div style="background-color:#161B22; border:1px solid #30363D; border-left:4px solid #E28743; border-radius:6px; padding:12px 14px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.8rem;">
                            <span style="color:#E28743; font-weight:700;">📄 Source File: <b>{src['source_file']}</b></span>
                            <span style="color:#58A6FF; font-weight:600;">📑 Page: <b>{src['page_number']}</b></span>
                        </div>
                        <div style="font-size:0.88rem; color:#E6EDF3; line-height:1.5; font-style:italic;">
                            "{src['content']}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": summary_intro,
                "sources": retrieved_chunks,
            })
            st.rerun()
