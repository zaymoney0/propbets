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

# ==================== FORCE TRUE DARK MODE + ELITE THEME ====================
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
    .proj-num {
        font-size: 92px !important;
        font-weight: 900;
        line-height: 1;
        margin: 18px 0;
    }
    .lock-high {color: #00ff9d; text-shadow: 0 0 40px #00ff9d; font-size: 36px !important; font-weight: 900;}
    .lock-med {color: #FFD60A; text-shadow: 0 0 30px #FFD60A; font-size: 36px !important;}
    .lock-low {color: #ff6b00; font-size: 36px !important;}
    .chart-title {
        font-size: 42px !important;
        font-weight: 900;
        text-align: center;
        margin: 70px 0 30px 0;
        color: white;
        text
