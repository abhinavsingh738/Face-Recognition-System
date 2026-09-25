import streamlit as st
import pandas as pd
import os
import time
import shutil
from datetime import datetime, timedelta

# --- 1. UI Styling ---
st.set_page_config(page_title="NEURAL-LINK OS", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #080a0e; border-right: 1px solid #00f2ff33; }
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.05) 0%, rgba(0, 114, 255, 0.05) 100%);
        padding: 20px; border-radius: 15px; border: 1px solid rgba(0, 242, 255, 0.2);
        margin-bottom: 10px; text-align: center;
    }
    .status-active { color: #00ff88; font-weight: bold; text-shadow: 0 0 10px #00ff88; animation: pulse 2s infinite; }
    .live-indicator { 
        background-color: #ff0000; color: white; padding: 2px 10px; 
        border-radius: 5px; font-weight: bold; font-size: 12px; margin-left: 10px;
    }
    .detail-text { color: #00f2ff; font-size: 0.9rem; margin: 4px 0; font-family: monospace; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    div.stButton > button {
        background: linear-gradient(90deg, #00f2ff, #0072ff);
        color: #000; font-weight: bold; border-radius: 8px; width: 100%;
    }
    .section-header { border-left: 4px solid #00f2ff; padding-left: 15px; color: #00f2ff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Engine (Enhanced Real-Time Fetch) ---
LOG_FILE = "attendance/attendance_log.csv"
REGISTRY_FILE = "attendance/student_registry.csv"
DATASET_DIR = "dataset"

def get_data():
    """Refreshes every call to ensure no data is missed."""
    expected_cols = ['Name', 'Enroll No', 'Roll No', 'Email', 'Mobile', 'Photo_Path']
    
    # Reload Logs
    if os.path.exists(LOG_FILE):
        try:
            logs = pd.read_csv(LOG_FILE)
            logs['Timestamp'] = pd.to_datetime(logs['Timestamp'], errors='coerce')
            logs = logs.dropna(subset=['Timestamp'])
        except:
            logs = pd.DataFrame(columns=['Name', 'Timestamp', 'Status'])
    else:
        logs = pd.DataFrame(columns=['Name', 'Timestamp', 'Status'])

    # Reload Registry (Crucial for showing details)
    if os.path.exists(REGISTRY_FILE):
        try:
            reg = pd.read_csv(REGISTRY_FILE)
            reg = reg.drop_duplicates(subset=['Enroll No'], keep='last')
            # Fix column names to be case-insensitive for matching
            reg.columns = [c.strip() for c in reg.columns]
        except:
            reg = pd.DataFrame(columns=expected_cols)
    else:
        reg = pd.DataFrame(columns=expected_cols)
        
    return logs, reg

# --- 3. Sidebar Navigation ---
st.sidebar.markdown("<h1 style='color: #00f2ff; text-align:center;'>NEURAL-LINK</h1>", unsafe_allow_html=True)
menu = st.sidebar.radio("SYSTEM ACCESS", ["LIVE FEED", "ENROLLMENT", "DATABASE", "ARCHIVES"])

# --- MODE: LIVE FEED (Tuned for Fast Sync & Offline) ---
if menu == "LIVE FEED":
    st.markdown("""<h2 class='section-header'>📡 Live Stream <span class='live-indicator'>LIVE</span></h2>""", unsafe_allow_html=True)
    
    # Force reload fresh data
    logs, registry = get_data()
    
    # 30-second window: This ensures "Offline" happens quickly
    active_cutoff = datetime.now() - timedelta(seconds=30)
    current_logs = logs[logs['Timestamp'] >= active_cutoff].sort_values('Timestamp', ascending=False)
    
    if not current_logs.empty:
        st.markdown('<span class="status-active">● SUBJECT DETECTED</span>', unsafe_allow_html=True)
        # Show top 3 unique recent detections
        recent_unique = current_logs.drop_duplicates('Name').head(3)
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(recent_unique.iterrows()):
            # Case-insensitive registry lookup
            person_data = registry[registry['Name'].str.upper() == row['Name'].upper()]
            
            with cols[i % 3]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                if not person_data.empty:
                    p = person_data.iloc[0]
                    img_path = str(p['Photo_Path'])
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.write("👤 [Image Missing]")
                    
                    st.write(f"### {p['Name']}")
                    st.markdown(f"<p class='detail-text'>ID: {p.get('Enroll No', 'N/A')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='detail-text'>Email: {p.get('Email', 'N/A')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='detail-text'>Mob: {p.get('Mobile', 'N/A')}</p>", unsafe_allow_html=True)
                else:
                    st.write(f"### {row['Name']}")
                    st.error("⚠️ Not Enrolled")
                
                st.caption(f"Active: {row['Timestamp'].strftime('%H:%M:%S')}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # This clears the screen almost immediately after you leave
        st.info("🛰️ System Standby: Waiting for detection...")
        st.caption("Cards will automatically clear 30 seconds after a subject leaves the frame.")

# --- MODE: ENROLLMENT (Strict Implementation) ---
elif menu == "ENROLLMENT":
    logs, registry = get_data() # Refresh
    st.markdown("<h2 class='section-header'>👤 Subject Induction</h2>", unsafe_allow_html=True)
    if 'form_id' not in st.session_state: st.session_state.form_id = 0
    fid = st.session_state.form_id
    
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", key=f"n_{fid}")
        enroll = st.text_input("Enrollment ID", key=f"en_{fid}")
    with c2:
        mobile = st.text_input("Mobile Number", key=f"m_{fid}")
        email = st.text_input("Email Address", key=f"em_{fid}")

    cam_on = st.toggle("Enable Induction Camera")
    uploaded_file = st.camera_input("Biometric Scan") if cam_on else st.file_uploader("Upload Profile Image")

    if st.button("🚀 COMPLETE INDUCTION"):
        if name and enroll and uploaded_file:
            path = os.path.join(DATASET_DIR, name)
            os.makedirs(path, exist_ok=True)
            img_path = os.path.join(path, f"{enroll}_profile.jpg")
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            new_entry = pd.DataFrame([{
                'Name': name, 
                'Enroll No': enroll, 
                'Email': email, 
                'Mobile': mobile, 
                'Photo_Path': img_path,
                'Roll No': 'N/A'
            }])
            
            # Save to Registry
            updated_reg = pd.concat([registry, new_entry], ignore_index=True)
            updated_reg.to_csv(REGISTRY_FILE, index=False)
            
            st.success(f"Subject {name} successfully logged into Database.")
            st.session_state.form_id += 1
            time.sleep(1)
            st.rerun()
        else:
            st.error("Missing required fields (Name, ID, or Image).")

# --- MODE: DATABASE ---
elif menu == "DATABASE":
    logs, registry = get_data() # Refresh
    st.markdown("<h2 class='section-header'>⚙️ Registry Management</h2>", unsafe_allow_html=True)
    if registry.empty:
        st.warning("No records found in local database.")
    else:
        for idx, row in registry.iterrows():
            with st.expander(f"👤 {row['Name']} (ID: {row['Enroll No']})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if os.path.exists(str(row['Photo_Path'])):
                        st.image(row['Photo_Path'], width=150)
                with col2:
                    new_email = st.text_input("Edit Email", row['Email'], key=f"edit_e_{idx}")
                    new_mob = st.text_input("Edit Mobile", row['Mobile'], key=f"edit_m_{idx}")
                    
                    b1, b2 = st.columns(2)
                    if b1.button("💾 SAVE CHANGES", key=f"save_{idx}"):
                        registry.at[idx, 'Email'] = new_email
                        registry.at[idx, 'Mobile'] = new_mob
                        registry.to_csv(REGISTRY_FILE, index=False)
                        st.success("Record Updated.")
                        st.rerun()
                    
                    if b2.button("🗑️ ERASE SUBJECT", key=f"del_{idx}"):
                        shutil.rmtree(os.path.join(DATASET_DIR, str(row['Name'])), ignore_errors=True)
                        registry.drop(idx).to_csv(REGISTRY_FILE, index=False)
                        st.warning("Subject Erased from System.")
                        st.rerun()

# --- MODE: ARCHIVES ---
elif menu == "ARCHIVES":
    logs, registry = get_data() # Refresh
    st.markdown("<h2 class='section-header'>🏛️ Timeline History</h2>", unsafe_allow_html=True)
    if logs.empty:
        st.info("Archive is currently empty.")
    else:
        logs['Date'] = logs['Timestamp'].dt.date
        dates = sorted(logs['Date'].unique(), reverse=True)
        sel_date = st.selectbox("📅 Select Date Range", dates)
        day_logs = logs[logs['Date'] == sel_date].sort_values('Timestamp', ascending=False)
        
        for ts in day_logs['Timestamp'].unique():
            with st.expander(f"🕒 Time: {pd.to_datetime(ts).strftime('%H:%M:%S')}"):
                entries = day_logs[day_logs['Timestamp'] == ts]
                for _, entry in entries.iterrows():
                    person = registry[registry['Name'].str.upper() == entry['Name'].upper()]
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        img_p = str(person.iloc[0]['Photo_Path']) if not person.empty else ""
                        if os.path.exists(img_p):
                            st.image(img_p, width=100)
                    with c2:
                        st.write(f"**Name:** {entry['Name']}")
                        if not person.empty:
                            st.caption(f"Enroll ID: {person.iloc[0]['Enroll No']} | Contact: {person.iloc[0]['Email']}")

# --- AUTO REFRESH TRIGGER ---
if menu == "LIVE FEED":
    time.sleep(0.5)
    st.rerun()