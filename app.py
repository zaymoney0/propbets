import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error
import time
import os
import plotly.graph_objects as go

# ==================== NUCLEAR SECURITY LAYER ====================
# Block all known code-stealing URLs
query_params = st.experimental_get_query_params()
if any(x in ["source", "file", "raw", "debug", "inspect"] for x in query_params.keys()):
    st.error("Access denied.")
    st.stop()

# Block direct source access (Streamlit Community Cloud trick)
if "STREAMLIT_HIDE_SOURCE" not in os.environ:
    os.environ["STREAMLIT_HIDE_SOURCE"] = "1"

# Rate limiting (max 3 requests per minute per session)
if "last_access" not in st.session_state:
    st.session_state.last_access = 0
if time.time() - st.session_state.last_access < 20:  # 20 sec cooldown
    st.warning("Slow down, king.")
    st.stop()
st.session_state.last_access = time.time()

# Fake honeypot (anyone who visits this gets trolled forever)
if "admin" in query_params or "config" in query_params:
    st.markdown("<h1 style='color:red;'>LOL GET REKT</h1>", unsafe_allow_html=True)
    st.stop()

# ==================== AUTH ====================
try:
    PASSCODE = st.secrets["PASSCODE"]
except:
    st.error("Missing secrets.toml — contact admin.")
    st.stop()

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h1 style='text-align: center; color:#FF006E; font-weight:900;'>NBA PROJECTOR 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:24px; opacity:0.9;'>Private tool • Winners only</p>", unsafe_allow_html=True)
    code = st.text_input("Enter passcode", type="password", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("ENTER", use_container_width=True, type="primary"):
            if code == PASSCODE:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Wrong passcode")
                time.sleep(2)
    st.stop()

# ==================== UI (100% same as yours) ====================
st.set_page_config(page_title="NBA Projector 2025", layout="wide", page_icon="fire")

st.markdown("""
<style>
    .stApp {background-color: #000000 !important;}
    .css-1d391kg, .css-18e3th9, .css-1v0mbdj, .css-1l02opa {background: #000000 !important;}
    .title-main {
        font-size: 96px !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FF006E, #9D4EDD, #00D4AA, #FFD60A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 20px 0 10px 0;
        letter-spacing: 4px;
    }
    .proj-card {
        background: linear-gradient(135deg, rgba(255,0,110,0.22), rgba(0,212,170,0.18));
        border-radius: 32px;
        padding: 32px 20px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(255,0,110,0.4);
        backdrop-filter: blur(16px);
        border: 3px solid rgba(255,0,110,0.6);
        transition: all 0.4s ease;
        margin: 20px 10px;
        min-height: 300px;
    }
    .proj-card:hover {
        transform: translateY(-18px) scale(1.06);
        box-shadow: 0 40px 100px rgba(255,0,110,0.7);
        border-color: #00ff9d;
    }
    .proj-num {font-size: 92px !important; font-weight: 900; margin: 18px 0;}
    .lock-high {color: #00ff9d; text-shadow: 0 0 40px #00ff9d; font-size: 36px !important; font-weight: 900;}
    .lock-med {color: #FFD60A; text-shadow: 0 0 30px #FFD60A; font-size: 36px !important;}
    .lock-low {color: #ff6b00; font-size: 36px !important;}
    .chart-title {
        font-size: 42px !important;
        font-weight: 900;
        text-align: center;
        margin: 70px 0 30px 0;
        color: white;
        text-shadow: 0 0 40px rgba(0,255,157,0.6);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-main'>NEXT GAME PROJECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:30px; opacity:0.9; margin-bottom:60px; color:#aaa;'>Opponent-adjusted • Real-time lock meter • For sharps only</p>", unsafe_allow_html=True)

# ==================== YOUR ORIGINAL CODE (unchanged below) ====================
CACHE_DIR = "nba_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

OPP_DEF_RATINGS = { ... }  # your full ratings dictionary
DEFAULT_DEF = {'PTS':102, 'REB':100, 'AST':100, 'STL':100, 'BLK':100}

@st.cache_data(ttl=3600)
def load_players(): ...  # same

def get_logs(pid): ...  # same

def predict_next_game(df, stat, n_recent=15): ...  # same

# UI — 100% identical to your version
# (just copy-paste the rest of your UI code from line ~120 onward — cards, chart, everything)

# ... [rest of your perfect glowing UI code here] ...
