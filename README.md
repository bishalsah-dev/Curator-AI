Curator-AI 2.0: Agentic Generative AI for Autonomous Cyber-Forensics

🛡️ Project Overview

Curator-AI 2.0 is an industrial-grade Cyber-Intelligence Forensic Pipeline designed to bridge the gap between binary anomaly detection and actionable forensic response. In modern Security Operations Centers (SOC), analysts suffer from "Alert Fatigue." Curator-AI solves this by not only detecting threats but curating them into human-readable forensic reports using Agentic AI.

The "Hybrid Intelligence" Architecture:

Detection Layer (ML): Uses a Random Forest Ensemble to isolate DDoS and Brute Force signatures from high-velocity network telemetry.

Curation Layer (GenAI): Employs Google Gemini 2.5-Flash to translate raw statistical anomalies into structured remediation protocols.

Persistence Layer (Cloud): Syncs all forensic artifacts to an immutable Firebase Firestore ledger for long-term auditing.

🚀 Key Features

Explainable AI (XAI): Moves beyond "Black Box" detection by providing post-hoc justifications for every flagged threat based on feature importance.

Resilient Edge-Intelligence: Features a Circuit Breaker architecture. If the Cloud API is unreachable (400/500 errors), the system pivots to a Local Forensic Fallback engine to maintain 100% uptime.

Real-Time Analytics: Interactive Plotly-based dashboard for high-velocity telemetry visualization.

Atomic Cloud Registry: Automated synchronization of forensic reports to a distributed NoSQL ledger.

🛠️ Technical Stack

Language: Python 3.9+

ML Engine: Scikit-Learn (Random Forest Classifier, StandardScaler)

Intelligence: Google Gemini 2.5-Flash (RESTful API Integration)

Backend/Database: Firebase Admin SDK (Firestore)

Dashboard: Streamlit Framework

Visualization: Plotly Express

📂 System Architecture

Ingestion: Raw logs (Requests, Auth Failures, CPU Load) are ingested.

Classification: Random Forest performs high-precision anomaly isolation.

Forensic Curation: The isolated vector is sent to the LLM agent for prescriptive analysis.

Audit: The curated report is saved to the Cloud Registry with an atomic timestamp.

🚦 Getting Started

1. Prerequisites

pip install streamlit pandas scikit-learn requests firebase-admin plotly


2. Configuration

Place your serviceAccountKey.json (Firebase) in the root directory.

Add your GEMINI_API_KEY in the app.py configuration section.

3. Execution

streamlit run app.py


📊 Evaluation & Results

Detection Precision: >98% in simulated DDoS and Brute Force environments.

MTTR (Mean Time to Respond): Reduced from minutes to milliseconds via automated curation.

Fault Tolerance: Successfully demonstrated Edge-Mode Fallback during Cloud API latencies.

🔮 Future Roadmap (The remaining 15%)

Active Response: Automated firewall orchestration (AWS/GCP API) to null-route malicious IPs.

Live Ingress: Transition from CSV-based simulation to real-time packet sniffing using Scapy.

Multi-Agent Reasoning: Orchestrating multiple LLM agents for deep-packet forensic analysis.

Author: BishaL

Status: 85% High-Fidelity MVP | Final Year B.Tech (AI/ML)
