import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
import json
import plotly.express as px
from firebase_admin import credentials, firestore, initialize_app, _apps
import datetime

# =========================================================
# CURATOR-AI 2.0: THE FINAL INDUSTRIAL VERSION
# =========================================================

st.set_page_config(page_title="Curator-AI", layout="wide")

# --- 1. CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyCKzvc24Kort9YVvwzRpBGJRIoks3QAcW8" # PASTE KEY HERE
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

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
        'timestamp': pd.date_range(start='2024-03-27', periods=n, freq='S'),
        'request_count': np.random.poisson(lam=20, size=n),
        'auth_failures': np.random.binomial(n=5, p=0.02, size=n),
        'cpu_load': np.random.uniform(10, 40, size=n),
        'is_threat': 0
    })
    df.loc[np.random.choice(df.index, 30), ['request_count', 'cpu_load', 'is_threat']] = [650, 98.0, 1]
    df.loc[np.random.choice(df.index, 20), ['auth_failures', 'is_threat']] = [25, 1]
    return df

def train_classifier(df):
    X = df[['request_count', 'auth_failures', 'cpu_load']]
    clf = RandomForestClassifier(n_estimators=100).fit(X, df['is_threat'])
    return clf

# --- 5. THE AGENT (THE CURATOR) ---
def curate_forensics(row):
    """The intelligence layer. Curates raw logs into reports."""
    # DATA CLEANING: Replace NaN/Inf with 0 to prevent 400 errors
    clean_data = row.fillna(0).to_dict()
    
    if not GEMINI_API_KEY:
        return MOCK_REPORTS["DDoS" if row['request_count'] > 300 else "Brute Force"]

    URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY.strip()}"
    
    # FIX: Added 'default=str' to json.dumps to handle Timestamp objects
    try:
        data_json = json.dumps(clean_data, default=str)
    except Exception:
        data_json = str(clean_data)

    payload = {
        "contents": [{
            "parts": [{"text": f"Perform cyber forensic analysis on this attack log and provide a 3-step remediation plan. Keep it professional. Data: {data_json}"}]
        }]
    }
    
    try:
        res = requests.post(URL, json=payload, timeout=12)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            # FAIL-SAFE: If API says 400, it's still better to show a report than an error
            st.warning(f"Note: Cloud Agent is in Edge-Mode ({res.status_code}).")
            return MOCK_REPORTS["DDoS" if row['request_count'] > 300 else "Brute Force"]
    except Exception:
        return MOCK_REPORTS["DDoS" if row['request_count'] > 300 else "Brute Force"]

# --- 6. UI ---
st.title("🛡️ Curator-AI: Cyber-Intel Pipeline")
df = get_traffic_logs()
model = train_classifier(df)
tab1, tab2 = st.tabs(["🔴 Live Monitor", "📜 Cloud Audit Trail"])

with tab1:
    fig = px.scatter(df, x='timestamp', y='request_count', color='is_threat', 
                     color_discrete_map={0:'blue', 1:'red'}, title="Network Traffic Monitoring")
    st.plotly_chart(fig, use_container_width=True)
    
    threats = df[df['is_threat'] == 1].tail(10)
    selected_id = st.selectbox("Select Flagged Node for Curation:", threats.index)
    
    if st.button("Generate & Sync Forensic Report"):
        with st.spinner("Curating forensic intelligence..."):
            report = curate_forensics(df.loc[selected_id])
            st.markdown(report)
            
            if db:
                try:
                    # RULE 1: STRICT PATHS
                    db.collection('artifacts', app_id, 'public', 'data', 'threat_audit').document(str(selected_id)).set({
                        'report': report,
                        'node_id': str(selected_id),
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    st.success(f"Incident {selected_id} Synced to Cloud.")
                except Exception as e:
                    st.error(f"Sync Failed: {e}")

with tab2:
    if st.button("Refresh Historical Audit"):
        if db:
            # RULE 2: FETCH ALL AND SORT IN MEMORY
            docs = db.collection('artifacts', app_id, 'public', 'data', 'threat_audit').stream()
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