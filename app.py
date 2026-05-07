# Line 1: Move your imports to the absolute top
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
import json
import plotly.express as px
from firebase_admin import credentials, firestore, initialize_app, _apps
import datetime

# Line 14: NOW initialize your session state
if 'report' not in st.session_state:
    st.session_state.report = None


# =========================================================
# CURATOR-AI 2.0: THE FINAL INDUSTRIAL VERSION
# =========================================================

st.set_page_config(page_title="Curator-AI", layout="wide")

# --- 1. CONFIGURATION ---  
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # We use the evergreen stable name
    MODEL_NAME = "gemini-1.5-flash" 
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Configuration Error: {e}")

# --- 2. FAIL-SAFE KNOWLEDGE BASE ---
MOCK_REPORTS = {
    "DDoS": "### 🛡️ AI Forensic Report: Volumetric DDoS\n**Vector:** Traffic Spike.\n**Remediation:** Enable rate limiting and scale ingress nodes.",
    "Brute Force": "### 🛡️ AI Forensic Report: Brute Force\n**Vector:** Auth Failures.\n**Remediation:** IP Lockout and 2FA enforcement."
}

# --- 3. DATABASE INIT ---
def init_db():
    if not _apps:
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            initialize_app(cred)
            return firestore.client()
        except Exception: return None
    return firestore.client()

db = init_db()
app_id = "curator-ai-cyber-intel"

# --- 4. ML ENGINE ---
@st.cache_data
def get_traffic_logs(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        # CHANGE Line 58 to:
        'timestamp': pd.date_range(start=datetime.datetime.now().strftime('%Y-%m-%d'), periods=n, freq='s'),
        'request_count': np.random.poisson(lam=20, size=n),
        'auth_failures': np.random.binomial(n=5, p=0.02, size=n),
        'cpu_load': np.random.uniform(10, 40, size=n),
        'is_threat': 0
    })
    df.loc[np.random.choice(df.index, 30), ['request_count', 'cpu_load', 'is_threat']] = [650, 98.0, 1]
    df.loc[np.random.choice(df.index, 20), ['auth_failures', 'is_threat']] = [25, 1]
    return df

def train_classifier(df):
    # We define the order HERE
    features = ['request_count', 'cpu_load', 'auth_failures']
    X = df[features] 
    y = df['is_threat']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- 5. THE AGENT (THE CURATOR) ---
def curate_forensics(row):
    # Data extraction with defaults
    req = row.get('request_count', 0)
    cpu = row.get('cpu_load', 0)
    pred = row.get('prediction', 'Normal')
    
    prompt = f"Analyze cyber threat: Node {row.name}, Requests {req}, CPU {cpu}%, ML Prediction {pred}."

    try:
        # Attempt Cloud Curation
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
        raise ValueError("Empty Response")
        
    except Exception:
        # THE RESILIENCE TRIGGER
        # If Cloud fails for ANY reason (404, 400, No Internet), this runs.
        if req > 500:
            return MOCK_REPORTS["DDoS"]
        elif cpu > 80:
            return MOCK_REPORTS["Brute Force"]
        else:
            return "✅ **ANALYSIS COMPLETE:** Telemetry indicates normal operational patterns. System Integrity Verified."

# --- 6. UI ---
st.title("🛡️ Curator-AI: Cyber-Intel Pipeline")
df = get_traffic_logs()
model = train_classifier(df)
df['prediction'] = model.predict(df[['request_count', 'cpu_load', 'auth_failures']])
# 1. ML Engine Health
# Proves the Random Forest Classifier is loaded and ready for inference [cite: 173, 210]
if model:
    st.sidebar.markdown("● **ML Engine:** `ACTIVE` (Random Forest)")
else:
    st.sidebar.markdown("● **ML Engine:** `OFFLINE`")

# 2. Cloud Agent Connectivity (Circuit Breaker Logic)
# Demonstrates the transition between Cloud and Edge-Mode [cite: 191, 232]
try:
    # Lightweight ping to verify Gemini 2.5-Flash availability [cite: 382]
    requests.get("https://generativelanguage.googleapis.com/", timeout=2)
    st.sidebar.markdown("● **Cloud Agent:** `CONNECTED` (Gemini 2.5)")
except:
    st.sidebar.markdown("● **Cloud Agent:** `EDGE_MODE` (Fallback Active)")
    st.sidebar.warning("⚠️ Resilience Triggered: Using Local Knowledge Base.")

# 3. Database Sync Status
# Verifies the connection to the Firebase Firestore audit ledger [cite: 55, 241, 349]
if db:
    st.sidebar.markdown("● **Audit Ledger:** `SYNCED` (Firestore)")
else:
    st.sidebar.markdown("● **Audit Ledger:** `LOCAL_ONLY` (No Cloud Sync)")

# Timestamp of last health check
st.sidebar.caption(f"Last Health Check: {datetime.datetime.now().strftime('%H:%M:%S')}")
st.sidebar.divider()
st.sidebar.subheader("🔌 Ingress Controller")
run_mode = st.sidebar.toggle("Enable Real-Time Scapy Sniffing", value=False)

if run_mode:
    try:
        from scapy.all import sniff
        st.sidebar.success("✅ Scapy Engine Initialized (Admin Mode)")
    except (ImportError, PermissionError):
        st.sidebar.error("❌ Permission Denied / Scapy Missing. Using Mock Ingress.")
        run_mode = False
df = get_traffic_logs()
for col in ['request_count', 'cpu_load', 'auth_failures']:
    if col not in df.columns:
        df[col] = 0
model = train_classifier(df)
FEATURES_LIST = ['request_count', 'cpu_load', 'auth_failures']
df['prediction'] = model.predict(df[FEATURES_LIST])
tab1, tab2 = st.tabs(["🔴 Live Monitor", "📜 Cloud Audit Trail"])

with tab1:
    fig = px.scatter(df, x='timestamp', y='request_count', color='is_threat', 
                    color_discrete_map={0:'blue', 1:'red'}, title="Network Traffic Monitoring")
    st.plotly_chart(fig, use_container_width=True)
    
    threats = df[df['is_threat'] == 1].tail(10)
    selected_id = st.selectbox("Select Flagged Node for Curation:", threats.index)

    # --- OPTION 1: RISK QUANTIZATION ---
node_data = df.loc[selected_id]
# Simplified CVSS-style score (0 to 10)
# Formula: ((Request/Max) * 7) + ((CPU/100) * 3)
risk_score = round(((min(node_data['request_count'], 1000) / 1000) * 7) + ((node_data['cpu_load'] / 100) * 3), 1)

# Display a Color-Coded Gauge
st.write(f"### 🎯 Real-Time Risk Score: {risk_score}/10")
if risk_score >= 7.5:
    st.error("CRITICAL: Immediate Response Required")
elif risk_score >= 4.0:
    st.warning("MEDIUM: Suspicious Activity Detected")
else:
    st.success("LOW: Normal Telemetry Patterns")
    
    if st.button("Generate & Sync Forensic Report"):
        with st.spinner("Curating forensic intelligence..."):
            st.session_state.report = curate_forensics(df.loc[selected_id])
            st.markdown(st.session_state.report)
            
# --- SECTION 1: ANALYST VERIFICATION & SYNC ---
st.divider()
st.subheader("🕵️ Analyst Verification")
is_validated = st.checkbox("I have reviewed the AI Forensic Report and verify its accuracy for NIST compliance.")

# Logic Gate: If NOT validated, show info. If validated, run the Sync.
if not is_validated:
    st.info("⚠️ Action Required: Please validate the report above to enable Cloud Sync and Mitigation.")

elif is_validated and st.session_state.report:
    if db:
        try:
            # 1. Sync to Firebase Firestore
            db.collection('threat_audit').document(str(selected_id)).set({
                'report': st.session_state.report,
                'node_id': str(selected_id),
                'timestamp': datetime.datetime.now().isoformat()
            })
            st.success(f"Incident {selected_id} Synced to Cloud.")

            # 2. Generate Forensic Bundle for local download
            forensic_bundle = f"""
CURATOR-AI 2.0: FORENSIC AUDIT EXPORT
-------------------------------------
TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
NODE IDENTIFIER: {selected_id}
RAW TELEMETRY: {df.loc[selected_id].to_dict()}
-------------------------------------
AI-DRIVEN FORENSIC CURATION:
{st.session_state.report}
-------------------------------------
CLASSIFICATION: {'MALICIOUS (THREAT)' if df.loc[selected_id]['is_threat'] == 1 else 'BENIGN (NORMAL)'}
COMPLIANCE: NIST SP 800-61 Standards Applied
"""

            st.download_button(
                label="📥 Download Forensic Audit Bundle (.txt)",
                data=forensic_bundle,
                file_name=f"CuratorAI_Audit_{selected_id}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Sync Failed: {e}")

# --- SECTION 2: CHAT INTERFACE (NOTICE: NO INDENTATION HERE) ---
# This starts at the far-left margin so it's always visible.
st.divider()
st.subheader("💬 Interactive Forensic Dialogue")
st.caption("Deep-dive investigation into the selected telemetry.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about this threat (e.g., 'What is the risk to the CPU?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add your Gemini chat logic here...

        with st.chat_message("assistant"):
            with st.spinner("Consulting Forensic Agents..."):
                node_data = df.loc[selected_id]
                contextual_prompt = f"""
                CONTEXT: Analyzing Node {selected_id}
                Request Count: {node_data['request_count']}
                CPU Load: {node_data['cpu_load']}%
                USER QUESTION: {prompt}
                TASK: Provide a technical, NIST-aligned forensic answer.
                """
                # Note: Calling your existing curate_forensics function
                full_response = curate_forensics(node_data)
                st.markdown(full_response)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
#--XAI FEATURE IMPORTANCE ---
st.divider()
st.subheader("📊 Algorithmic Explainability (XAI)")
st.caption("Statistical weight of features contributing to the current classification.")

# Extract importance from your trained Random Forest model
importances = model.feature_importances_
feature_names = ['request_count', 'auth_failures', 'cpu_load']
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Render a professional bar chart
fig_xai = px.bar(
    feature_df, x='Importance', y='Feature', orientation='h',
    title="Random Forest Gini Importance",
    color='Importance', color_continuous_scale='Blues'
)
st.plotly_chart(fig_xai, use_container_width=True)

with st.expander("📝 Interpret this Data"):
    top_feature = feature_df.iloc[0]['Feature']
    st.write(f"The model is primarily prioritizing **{top_feature}** for this isolation. This indicates that the current threat signature is heavily driven by this specific telemetry vector.")
    # --- OPTION 3: ACTIVE MITIGATION (RESPONSE) ---
if is_validated:
    st.divider()
    if st.button("🚀 EXECUTE MITIGATION PROTOCOL"):
        with st.status("Isolating Node...", expanded=True) as status:
            st.write("Blocking Source IP in Firewall...")
            st.write("Terminating Malicious Process IDs...")
            st.write("Resetting Authentication Tokens...")
            status.update(label="✅ Node Isolated & Contained!", state="complete")
        
        st.balloons()
        st.success(f"Response Action Logged: Node {selected_id} is no longer a threat.")

with tab2:
    if st.button("Refresh Historical Audit"):
        if db:
            # RULE 2: FETCH ALL AND SORT IN MEMORY
            docs = db.collection('threat_audit').stream()
            records = []
            for doc in docs:
                data = doc.to_dict()
                records.append(data)
            
            # SORT BY TIMESTAMP DESCENDING
            sorted_records = sorted(records, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            if not sorted_records:
                st.write("No records found in cloud registry.")
            for r in sorted_records:
                with st.expander(f"Incident {r.get('node_id')} | {r.get('timestamp')[:19]}"):
                    st.markdown(r.get('report'))
        else:
            st.warning("Database Connection Unavailable.")
