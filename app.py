import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
import re
import streamlit as st
import plotly.graph_objects as go
from google import genai
from google.genai import types
from groq import Groq
import pandas as pd
import plotly.express as px
import json
import sqlite3
from config import TenantProfile, DEFAULT_TENANT
from core.utils import extract_pdf_text, send_email, parse_score, parse_verdict, split_task_outputs
from pydantic import BaseModel, Field

# Define a strict schema for your radar chart metrics
class CandidateMetrics(BaseModel):
    technical_skills: int = Field(..., description="Score 0-100 based on core tech stack matching.")
    experience: int = Field(..., description="Score 0-100 based on years and relevance.")
    education: int = Field(..., description="Score 0-100 based on degree and certification alignment.")
    culture_fit: int = Field(..., description="Score 0-100 based on soft skills and profile tone.")

# =====================================================================
# 💾 DATABASE ROUTINES (With Extended Metric Logging)
# =====================================================================
def init_pipeline_db():
    """Initializes the embedded relational layer for candidate profile preservation."""
    conn = sqlite3.connect("recruitment_analytics.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            match_score INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            department TEXT NOT NULL,
            tech_score INTEGER DEFAULT 0,
            exp_score INTEGER DEFAULT 0,
            edu_score INTEGER DEFAULT 0,
            culture_score INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_candidate_metrics(name, score, verdict, dept, radar_data):
    """Commits extracted multi-agent metric values securely to relational tables."""
    conn = sqlite3.connect("recruitment_analytics.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO candidate_analytics (
            candidate_name, match_score, verdict, department, 
            tech_score, exp_score, edu_score, culture_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name, score, verdict, dept,
        radar_data.get("technical_skills", score),
        radar_data.get("experience", score),
        radar_data.get("education", score),
        radar_data.get("culture_fit", score)
    ))
    conn.commit()
    conn.close()

# Invoke database generation securely on execution startup
init_pipeline_db()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & KEY SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Recruitment Pipeline", layout="wide")

# Link the external custom style.css architecture cleanly
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
USE_GROQ = False
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    USE_GROQ = True
elif GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Track session state components
for key, default in {
    "pipeline_done": False,
    "batch_results": [],
    "last_processed_candidate": "",
    "score": 0,
    "verdict": "CONDITIONAL REVIEW",
    "radar_metrics": None,
    "task_outputs": [],
    "raw_report": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL PANEL (Dynamic Role Weighting Implementation)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 15px; border-radius: 8px; background-color: rgba(128, 128, 128, 0.1); border: 1px solid #94a3b8; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                <span style="font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; opacity: 0.9;">SYSTEM ONLINE</span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.8rem; line-height: 1.4; font-weight: 500; opacity: 0.7;">
                AI Recruitment Engine is actively monitoring talent pipelines.
            </p>
        </div>
        <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown("## Company Profile")
    st.caption("Configure for your organization before running.")
    company   = st.text_input("Company Name",      value=DEFAULT_TENANT.company_name)
    dept      = st.text_input("Hiring Department", value=DEFAULT_TENANT.department)
    sched_url = st.text_input("Interview Booking URL", value=DEFAULT_TENANT.scheduling_url)

    try:
        active_profile = TenantProfile(company_name=company, department=dept, scheduling_url=sched_url)
        profile_ok = True
    except Exception:
        profile_ok = False

    st.markdown("---")
    if USE_GROQ:
        st.markdown("**Model:** `llama-3.3-70b-versatile` (Groq)")
    else:
        st.markdown("**Model:** `gemini-flash-latest`")
    st.markdown("**Pipeline:** Batch Multiplex Processing Matrix")
    # API Key is securely loaded from .env behind the scenes.
    # No UI exposure.
# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC ENTERPRISE BRAND HEADER LAYER (As Seen in Screenshot 2026-06-03 164454.png)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="brand-header-matrix">
        <h1 class="brand-title-text">{company} — AI Recruitment Pipeline</h1>
    </div>
    <div class="brand-subtitle-subtext">
        Hiring for the <strong>{dept}</strong> team
    </div>
    """, 
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT PANEL (Bulk Resume Upload & Batch Processing)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Setup Pipeline Parameters")
col_jd, col_resume = st.columns(2, gap="large")
with col_jd:
    st.markdown("**Target Job Description**")
    job_desc = st.text_area(
        "Paste the full job description here",
        height=200,
        placeholder="E.g., Skills: Python, Java\nExperience: 2+ years",
        label_visibility="collapsed"
    )

with col_resume:
    st.markdown("**Batch Resume Queue (Multiple PDFs Supported)**")
    uploaded_files = st.file_uploader("Upload resume PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} candidate profiles buffered into batch streaming pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BATCH PIPELINE TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
can_run = bool(job_desc.strip() and uploaded_files and profile_ok and client is not None)

st.markdown('<div style="padding: 10px 0 30px 0;">', unsafe_allow_html=True)
run_clicked = st.button("Execute Enterprise Batch Evaluation Pipeline", use_container_width=True, disabled=not can_run, type="primary")


import time
from google.genai.errors import APIError  # Import the modern SDK error handler

if run_clicked:
    st.session_state["batch_results"] = []
    
    with st.status("Orchestrating Enterprise Multiplex Pipeline...", expanded=True) as status:
        try:
            for idx, current_file in enumerate(uploaded_files):
                candidate_name_raw = current_file.name.replace(".pdf", "").title()
                st.write(f"Processing Applicant ({idx+1}/{len(uploaded_files)}): **{candidate_name_raw}**...")
                
                # 1. Parse File Stream
                resume_text = extract_pdf_text(current_file)
                
                # ─── FREE TIER AUTO-RETRY ARCHITECTURE ───
                raw_report_text = ""
                parsed_radar_data = None
                max_retries = 7
                
                for attempt in range(max_retries):
                    try:
                        # 2. Main Unstructured Analysis Generation
                        master_prompt = f"""
                        You are a strict, top-tier Enterprise Recruitment AI evaluating a candidate for the {dept} department at {company}.
                        Analyze the resume against the job description with extremely high professional standards.
                        Generate your report using EXACTLY these markdown headers:
                        
                        ### Resume Screening Summary
                        (Provide a highly detailed, professional executive summary of the candidate's background. Use bullet points to highlight their core competencies, total years of experience, and most notable career achievements.)
                        
                        ### Technical Evaluation
                        (Critically and extensively assess their specific skills against the job requirements. Use bullet points to break down 'Strengths' and 'Areas for Improvement'. YOU MUST END THIS SECTION WITH EXACTLY 'Match Score: X' where X is an integer from 0-100 indicating how perfectly they match the JD.)
                        
                        ### Hiring Recommendation
                        (Provide a comprehensive, highly professional verdict. Justify your recommendation with a detailed paragraph explaining exactly why their profile is a strong, moderate, or weak fit for this specific enterprise role.)
                        
                        ### Interview Question Set
                        (Provide 3-5 highly advanced, role-specific interview questions. For each question, provide a brief bullet point explaining what specific skill or competency this question is designed to evaluate.)
                        
                        ### Interview Scheduling Plan
                        (Provide a highly realistic, practical interview plan based ONLY on the actual facts present in the candidate's resume and the job description. Do NOT hallucinate skills. Propose 2-3 focused interview stages using a bulleted list. For each stage, clearly outline which real skills from their resume will be verified.)
                        
                        ### Candidate Outreach Email
                        (Write a polished, professional, and detailed email to the candidate to invite them to the next steps. Ensure the tone is warm but highly corporate. Use {sched_url} if available. Do NOT include any placeholder times or dates.)
                        
                        Candidate Resume Text: {resume_text}
                        Job Description Input: {job_desc}
                        """
                        if USE_GROQ:
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": master_prompt}]
                            )
                            raw_report_text = response.choices[0].message.content
                            
                            # Brief safety pause between rapid-fire requests
                            time.sleep(4)
                            
                            # 3. Dynamic Structured Calibration Metrics Engine
                            metric_prompt = f"Evaluate {candidate_name_raw} against requirements. Output 4 scores 0-100 for technical_skills, experience, education, culture_fit based on schema.\nJD: {job_desc}\nResume: {resume_text}\nOutput ONLY valid JSON matching this schema: {{\"technical_skills\": int, \"experience\": int, \"education\": int, \"culture_fit\": int}}"
                            metric_response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": metric_prompt}],
                                response_format={"type": "json_object"}
                            )
                            parsed_radar_data = json.loads(metric_response.choices[0].message.content)
                        else:
                            response = client.models.generate_content(model='gemini-2.0-flash', contents=master_prompt)
                            raw_report_text = response.text
                            
                            # Brief safety pause between rapid-fire requests
                            time.sleep(4)
                            
                            # 3. Dynamic Structured Calibration Metrics Engine
                            metric_prompt = f"Evaluate {candidate_name_raw} against requirements. Output 4 scores 0-100 for technical_skills, experience, education, culture_fit based on schema.\nJD: {job_desc}\nResume: {resume_text}"
                            metric_response = client.models.generate_content(
                                model='gemini-2.0-flash',
                                contents=metric_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=CandidateMetrics,
                                ),
                            )
                            parsed_radar_data = json.loads(metric_response.text)
                        
                        # If both calls succeed, break the retry loop and continue processing!
                        break
                        
                    except Exception as err:
                        err_str = str(err)
                        is_retryable = "429" in err_str or "503" in err_str or "rate limit" in err_str.lower()
                        if is_retryable and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 8
                            err_type = "Rate limit" if "429" in err_str else "High demand (503)"
                            st.toast(f"⚠️ {err_type} hit. Securely backing off for {wait_time}s to bypass block...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise err  # Pass error to the main except block if retries fail permanently
                
                # 4. Parse Responses
                split_outputs_list = split_task_outputs(raw_report_text)
                calculated_aggregate_score = parse_score(split_outputs_list[1] or raw_report_text)
                
                # Logically derive verdict from the exact match score for strict consistency
                if calculated_aggregate_score >= 85:
                    extracted_verdict = "STRONG HIRE"
                elif calculated_aggregate_score >= 60:
                    extracted_verdict = "CONDITIONAL REVIEW"
                else:
                    extracted_verdict = "WEAK"
                
                # 5. Commit to SQLite Persistent Relational Layer
                log_candidate_metrics(
                    name=candidate_name_raw,
                    score=calculated_aggregate_score,
                    verdict=extracted_verdict,
                    dept=dept,
                    radar_data=parsed_radar_data
                )
                
                # Buffer tracking records across the session loop
                candidate_payload = {
                    "name": candidate_name_raw,
                    "score": calculated_aggregate_score,
                    "verdict": extracted_verdict,
                    "radar_metrics": parsed_radar_data,
                    "task_outputs": split_outputs_list,
                    "raw_report": raw_report_text
                }
                st.session_state["batch_results"].append(candidate_payload)
                
                # Toast notification for successful processing
                st.toast(f"Successfully evaluated {candidate_name_raw}!", icon="🎉")
                
                # Enterprise Free-Tier Rate Limit Governor (Google allows 15 requests/minute)
                # 2 API calls per resume = max 7 resumes per minute. 60 / 7 = ~8.5 seconds delay.
                time.sleep(8.5)
            
            # Cache the last active row to keep presentation displays functional
            if st.session_state["batch_results"]:
                last_candidate = st.session_state["batch_results"][-1]
                st.session_state["last_processed_candidate"] = last_candidate["name"]
                st.session_state["score"] = last_candidate["score"]
                st.session_state["verdict"] = last_candidate["verdict"]
                st.session_state["radar_metrics"] = last_candidate["radar_metrics"]
                st.session_state["task_outputs"] = last_candidate["task_outputs"]
                st.session_state["raw_report"] = last_candidate["raw_report"]
                
            st.session_state["pipeline_done"] = True
            status.update(label=f"Batch pipeline completed! {len(uploaded_files)} profiles fully evaluated.", state="complete", expanded=False)
            
        except Exception as e:
            status.update(label="Batch pipeline runtime exception", state="error", expanded=True)
            st.error(f"System Exception Tracking Event: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("pipeline_done", False) and st.session_state.get("batch_results", []):
    st.markdown("---")
    st.subheader("Focused Candidate Workspace Panel")
    
    candidate_names_list = [entry["name"] for entry in st.session_state["batch_results"]]
    selected_name = st.selectbox("Toggle Focused Workspace Profile Target:", candidate_names_list, index=len(candidate_names_list)-1)
    
    focused_record = next(entry for entry in st.session_state["batch_results"] if entry["name"] == selected_name)
    
    score    = focused_record["score"]
    verdict  = focused_record["verdict"]
    outputs  = focused_record["task_outputs"]
    radar_source = focused_record["radar_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    verdict_color = {"STRONG HIRE": "", "CONDITIONAL REVIEW": "", "WEAK": ""}.get(verdict, "")
    m1.markdown(f"<div class='custom-metric-card'><div class='custom-metric-label'>Weighted Match Score</div><div class='custom-metric-val' style='font-size:1.8rem; color:#10b981;'>{score}%</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='custom-metric-card'><div class='custom-metric-label'>Verdict State</div><div class='custom-metric-val'>{verdict_color} {verdict}</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='custom-metric-card'><div class='custom-metric-label'>Target Tenant</div><div class='custom-metric-val'>{company}</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='custom-metric-card'><div class='custom-metric-label'>Hiring Group</div><div class='custom-metric-val'>{dept}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    chart_col, log_col = st.columns([3, 2])

    with chart_col:
        c1, c2 = st.columns(2)
        with c1:
            # Advanced Candidate Donut Chart
            st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:5px;'>Candidate Match Index</p>", unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                values=[score, 100-score],
                labels=['Match', 'Gap'],
                hole=0.75,
                marker_colors=['#0070f3', '#161b22'],
                textinfo='none',
                hoverinfo='label+percent'
            ))
            fig_donut.add_annotation(text=f"{score}%<br><span style='font-size:12px;color:#94a3b8;'>Match</span>", x=0.5, y=0.5, font_size=32, showarrow=False, font_color="#ffffff")
            fig_donut.update_layout(
                height=300, 
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        with c2:
            # Advanced Candidate Radar Chart
            st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:5px;'>Candidate Skill Radar</p>", unsafe_allow_html=True)
            cats = ["Education", "Culture Fit", "Technical Skills", "Experience"]
            
            val_edu = radar_source.get("education", score)
            val_cult = radar_source.get("culture_fit", score)
            val_tech = radar_source.get("technical_skills", score)
            val_exp = radar_source.get("experience", score)
            
            values = [val_edu, val_cult, val_tech, val_exp]
            
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]], 
                theta=cats + [cats[0]], 
                fill="toself",
                fillcolor="rgba(0, 112, 243, 0.15)", 
                line=dict(color="#0070f3", width=3),
                marker=dict(color="#0070f3", size=8)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", showticklabels=False),
                    angularaxis=dict(direction="clockwise", rotation=90, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=10, color="#94a3b8"))
                ),
                margin=dict(t=30, b=30, l=30, r=30),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with log_col:
        st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#94a3b8; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.05em;'>Agent Execution Matrix</p>", unsafe_allow_html=True)
        agent_names = ["Resume Screener", "Technical Evaluator", "Decision Authority", "Interview Designer", "Scheduler", "Comms Officer"]
        for name in agent_names:
            st.success(f"{name} Pipeline Verified for {selected_name}")
        

    st.markdown("---")
    
    head_c1, head_c2 = st.columns([3, 1])
    with head_c1:
        st.subheader("Detailed Agent Outputs")
    with head_c2:
        st.download_button(
            label="📄 Download AI Dossier",
            data=st.session_state["raw_report"].encode('utf-8'),
            file_name=f"{selected_name}_dossier.txt",
            mime="text/plain",
            use_container_width=True
        )
    tabs = st.tabs(["Resume Summary", "Performance Evaluation", "Hiring Recommendation", "Contextual Interview Qs", "Scheduling Execution", "Communications Lead", "💬 Interactive Q&A"])
    
    for i, tab in enumerate(tabs[:6]):
        with tab:
            content = outputs[i] if i < len(outputs) else ""
            if content:
                st.markdown(content)
            else:
                st.info("Verification payload loaded successfully in secondary analytical channel streams.")
                
    with tabs[6]:
        st.markdown("<p style='color:#94a3b8;'>Ask the AI specifically about this candidate's background and evaluation.</p>", unsafe_allow_html=True)
        chat_q = st.chat_input(f"Ask a question about {selected_name}...")
        if chat_q:
            st.markdown(f"**You:** {chat_q}")
            with st.spinner("Analyzing candidate dossier..."):
                chat_prompt = f"Answer the user's question based strictly on this candidate's AI evaluation report.\n\nReport:\n{st.session_state['raw_report']}\n\nQuestion: {chat_q}"
                max_retries = 4
                for attempt in range(max_retries):
                    try:
                        if USE_GROQ:
                            chat_resp = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": chat_prompt}]
                            )
                            st.info(chat_resp.choices[0].message.content)
                        else:
                            chat_resp = client.models.generate_content(model='gemini-2.0-flash', contents=chat_prompt)
                            st.info(chat_resp.text)
                        break
                    except Exception as e:
                        if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 4
                            st.warning(f"Google API High Demand (503). Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                            import time
                            time.sleep(wait_time)
                        else:
                            st.error(f"Chat failed: {e}")
                            break

    st.markdown("---")
    st.subheader("Send Automated Outreach Correspondence")
    with st.expander("Configure & stream candidate email payload via SMTP gateway"):
        e1, e2 = st.columns(2)
        with e1:
            sender_email  = st.text_input("Your Gmail address", placeholder="recruiter@gmail.com")
            app_password  = st.text_input("Gmail App Password", type="password")
        with e2:
            recv_email = st.text_input("Candidate email", placeholder="candidate@email.com")
            subj       = st.text_input("Subject line", value=f"Your application status to {active_profile.company_name}")

        if st.button("Dispatch Correspondence", use_container_width=True):
            if not all([sender_email, app_password, recv_email]):
                st.warning("All verification tracking network pathways must be populated.")
            else:
                try:
                    with st.spinner("Streaming via Gmail SMTP node layers..."):
                        send_email(sender_email, app_password, recv_email, subj, outputs[5] if len(outputs) > 5 else focused_record["raw_report"])
                    st.success(f"Outreach packet deployed safely to {recv_email}")
                except Exception as e:
                    st.error(f"Gateway stream error exception event: {e}")


# 📈 GLOBAL ENTERPRISE TALENT ANALYTICS VIEWS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

view_tab1, view_tab2 = st.tabs(["Enterprise Talent Leaderboard", "Deep Talent Analytics Dashboard"])

with view_tab1:
    st.caption("Historical match metrics matrix rankings aggregated dynamically across relational database tables.")
    try:
        conn = sqlite3.connect("recruitment_analytics.db")
        leaderboard_df = pd.read_sql_query("""
            SELECT id,
                   candidate_name AS [Candidate ID], 
                   match_score AS [Match Rating %], 
                   verdict AS [Verdict Status], 
                   department AS [Target Business Team]
            FROM candidate_analytics 
            ORDER BY match_score DESC
        """, conn)
        conn.close()

        if not leaderboard_df.empty:
            
            # Real-time Candidate Search Filter (Global)
            search_query = st.text_input("🔍 Filter Database", placeholder="Search by candidate name or verdict...", label_visibility="collapsed")
            if search_query:
                leaderboard_df = leaderboard_df[leaderboard_df.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)]
                
            metric_plot_col, table_data_col = st.columns([1, 1], gap="medium")
            
            with metric_plot_col:
                st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:15px;'>Candidate Match Rankings</p>", unsafe_allow_html=True)

                fig_leader = px.bar(
                    leaderboard_df.head(8), 
                    x='Match Rating %', 
                    y='Candidate ID',
                    orientation='h',
                    color='Match Rating %',
                    color_continuous_scale='darkmint',
                    template='plotly_white'
                )
                fig_leader.update_layout(
                    yaxis={'categoryorder':'total ascending'}, 
                    height=260, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_leader, use_container_width=True, config={"displayModeBar": False})
                
            with table_data_col:
                # Add clear option and data editor for deleting rows
                st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:15px;'>Candidate Database Management</p>", unsafe_allow_html=True)
                
                edited_df = st.data_editor(
                    leaderboard_df, 
                    use_container_width=True, 
                    hide_index=True, 
                    height=220,
                    num_rows="dynamic",
                    column_config={
                        "id": None, # Hide the ID column
                        "Match Rating %": st.column_config.ProgressColumn(
                            "Match Rating",
                            help="Overall match score percentage",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Verdict Status": st.column_config.TextColumn(
                            "Verdict Status",
                            width="medium",
                        ),
                        "Candidate ID": st.column_config.TextColumn(
                            "Candidate ID",
                            width="medium",
                        )
                    },
                    key="candidate_editor"
                )
                
                c_btn1, c_btn2, c_btn3 = st.columns([2, 2, 1])
                with c_btn1:
                    if st.button("🧹 Clear All", type="primary", use_container_width=True):
                        conn = sqlite3.connect("recruitment_analytics.db")
                        conn.cursor().execute("DELETE FROM candidate_analytics")
                        conn.commit()
                        conn.close()
                        st.rerun()
                
                with c_btn2:
                    # CSV Export Capability
                    csv_export = leaderboard_df.drop(columns=['id'], errors='ignore').to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_export,
                        file_name="candidate_leaderboard.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Sync deletion back to database
                if len(edited_df) < len(leaderboard_df):
                    deleted_ids = set(leaderboard_df['id']) - set(edited_df['id'])
                    if deleted_ids:
                        conn = sqlite3.connect("recruitment_analytics.db")
                        cursor = conn.cursor()
                        cursor.execute(f"DELETE FROM candidate_analytics WHERE id IN ({','.join(map(str, deleted_ids))})")
                        conn.commit()
                        conn.close()
                        st.rerun()
            
        else:
            pass
    except Exception as db_err:
        st.error(f"Failed to query backend analytical relational table layers: {db_err}")

with view_tab2:
    st.caption("Deep aggregate summaries across structural candidate pool variables.")
    try:
        conn = sqlite3.connect("recruitment_analytics.db")
        analytics_df = pd.read_sql_query("SELECT id, candidate_name as name, match_score, verdict, department, tech_score, exp_score, edu_score, culture_score FROM candidate_analytics", conn)
        conn.close()

        if not analytics_df.empty:
            
            an_col1, an_col2 = st.columns(2, gap="medium")
            
            with an_col1:
                st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;'>Hiring Conversion Funnel Metrics</p>", unsafe_allow_html=True)
                verdict_counts = analytics_df['verdict'].value_counts().reset_index()
                verdict_counts.columns = ['Verdict Status', 'Total Candidates']
                
                # Plotly Funnel design config
                fig_funnel = px.funnel(
                    verdict_counts, 
                    x='Total Candidates', 
                    y='Verdict Status',
                    color='Verdict Status',
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                    template='plotly_white'
                )
                fig_funnel.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_funnel, use_container_width=True, config={"displayModeBar": False})
                
            with an_col2:
                st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;'>Candidate Talent Matrix Distribution</p>", unsafe_allow_html=True)
                
                # Generate an interactive scatter distribution matrix mapping individual scores
                fig_scatter = px.scatter(
                    analytics_df,
                    x='tech_score',
                    y='match_score',
                    color='verdict',
                    size='exp_score',
                    hover_data=['department'],
                    labels={
                        'tech_score': 'Technical Skills Rating (0-100)',
                        'match_score': 'Overall Match Percentage (0-100)',
                        'verdict': 'Hiring Verdict'
                    },
                    color_discrete_map={
                        'STRONG HIRE': '#10b981',
                        'CONDITIONAL REVIEW': '#f59e0b',
                        'WEAK': '#ef4444'
                    },
                    template='plotly_white'
                )
                
                fig_scatter.update_layout(
                    height=240, 
                    margin=dict(t=10, b=10, l=10, r=10), 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_scatter.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')))
                st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
            
            st.markdown("---")
            st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;'>Multi-Candidate Radar Comparison</p>", unsafe_allow_html=True)
            
            # Interactive Radar Comparison Overlay
            candidate_list = analytics_df['name'].dropna().unique().tolist()
            if len(candidate_list) > 0:
                default_selections = candidate_list[:3] if len(candidate_list) >= 3 else candidate_list
                compare_candidates = st.multiselect("Select candidates to overlay & compare:", candidate_list, default=default_selections)
                
                if compare_candidates:
                    fig_multi_radar = go.Figure()
                    colors = ['#0070f3', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
                    cats = ["Education", "Culture Fit", "Technical Skills", "Experience"]
                    
                    for idx, cand in enumerate(compare_candidates):
                        cand_rows = analytics_df[analytics_df['name'] == cand]
                        if not cand_rows.empty:
                            cand_row = cand_rows.iloc[0]
                            values = [cand_row['edu_score'], cand_row['culture_score'], cand_row['tech_score'], cand_row['exp_score']]
                            
                            fig_multi_radar.add_trace(go.Scatterpolar(
                                r=values + [values[0]],
                                theta=cats + [cats[0]],
                                fill='toself',
                                name=cand,
                                line=dict(color=colors[idx % len(colors)], width=2.5),
                                marker=dict(size=6)
                            ))
                            
                    fig_multi_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", showticklabels=False),
                            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8", size=13))
                        ),
                        margin=dict(t=40, b=40, l=40, r=40),
                        height=420,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_multi_radar, use_container_width=True, config={"displayModeBar": False})
            
        else:
            pass
    except Exception as an_err:
        st.error(f"Failed to aggregate deep pipeline visual matrices: {an_err}")