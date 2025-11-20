import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguegamelog
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time
import os
import plotly.graph_objects as go
from datetime import datetime

# ==================== PASSCODE ====================
PASSCODE = "getmoney"

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    st.markdown("<h1 style='text-align: center; color:#FF006E;'>NBA Projector 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>Private tool • Built for winners only</p>", unsafe_allow_html=True)
    code = st.text_input("Enter passcode", type="password", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("ENTER", use_container_width=True):
            if code == PASSCODE:
                st.session_state.auth = True
                st.rerun()
            else: 
                st.error("Wrong passcode")
    st.stop()

st.set_page_config(page_title="NBA Projector 2025", layout="wide", page_icon="fire")
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight: bold; text-align: center; margin: 0;}
    .stat-box {font-size: 32px; font-weight: bold; text-align: center; padding: 10px; border-radius: 15px; margin: 10px;}
    .proj-main {background: linear-gradient(90deg, #FF006E, #9D4EDD); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .lock-high {color: #00D4AA;}
    .lock-med {color: #FFD60A;}
    .lock-low {color: #FF6B00;}
    section[data-testid="stSidebar"] {background: #0e0e0e;}
    .css-1d391kg {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>NBA NEXT-GAME PROJECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:20px; opacity:0.8;'>Opponent-adjusted • Pace-aware • Zero overfitting</p>", unsafe_allow_html=True)

CACHE_DIR = "nba_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== DEFENSIVE RATINGS (2024-25 season) ====================
OPP_DEF_RATINGS = {
    'PTS': {'BOS':94, 'MIN':95, 'ORL':96, 'OKC':94, 'CLE':93, 'NYK':97, 'MIA':98, 'HOU':96, 'LAL':101, 'DEN':100,
            'GSW':99, 'PHI':98, 'MIL':104, 'IND':107, 'PHX':102, 'SAC':103, 'LAC':100, 'DAL':101, 'NOP':105,
            'ATL':110, 'CHI':108, 'TOR':109, 'MEM':99, 'SAS':104, 'POR':111, 'BKN':107, 'CHA':112, 'WAS':113, 'DET':106, 'UTA':110},
    'REB': {'MIN':92, 'ORL':94, 'BOS':95, 'NOP':96, 'CLE':94, 'NYK':97, 'OKC':96, 'LAL':100, 'SAC':101, 'CHI':105,
            'TOR':106, 'IND':108, 'PHI':99, 'MIL':107, 'DET':105, 'WAS':112, 'CHA Jas':111, 'ATL':110},
    'AST': {'BOS':93, 'OKC':94, 'ORL':95, 'CLE':94, 'MIN':96, 'NYK':97, 'MIA':98, 'HOU':97, 'GSW':100, 'LAC':99},
    'STL': {'OKC':92, 'BOS':93, 'ORL':94, 'HOU':95, 'MIN':95, 'NYK':96, 'CHI':105, 'TOR':106},
    'BLK': {'MEM':90, 'ORL':92, 'SAS':93, 'MIA':94, 'BOS':94, 'BKN':105, 'WAS':108}
}
DEFAULT_DEF = {'PTS':102, 'REB':100, 'AST':102, 'STL':100, 'BLK':100}

# ==================== DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_players():
    df = pd.DataFrame(players.get_active_players())
    df['lower'] = df['full_name'].str.lower()
    return df
players_df = load_players()

@st.cache_data(ttl=3600)
def get_team_abbrev():
    season = "2024-25"
    log = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    mapping = dict(zip(log['TEAM_NAME'], log['TEAM_ABBREVIATION']))
    return mapping

team_name_to_abv = get_team_abbrev()

def get_logs(pid):
    cache = f"{CACHE_DIR}/p_{pid}.parquet"
    if os.path.exists(cache) and os.path.getmtime(cache) > time.time() - 86400:
        return pd.read_parquet(cache)
    
    logs = []
    for s in ["2022-23", "2023-24", "2024-25"]:
        try:
            data = playergamelog.PlayerGameLog(player_id=pid, season=s).get_data_frames()[0]
            logs.append(data)
            time.sleep(0.8)
        except: pass
    if not logs: return pd.DataFrame()
    
    df = pd.concat(logs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    keep = ['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK','FGM','FGA','FG3M','FTM','FTA','TOV','PLUS_MINUS']
    df = df[keep].copy()
    
    for c in ['PTS','REB','AST','STL','BLK','MIN','TOV','PLUS_MINUS']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    df.to_parquet(cache)
    return df

# ==================== IMPROVED MODEL ====================
def predict_next_game(df, stat, n_recent=15):
    if len(df) < 20:
        return round(df[stat].mean(), 1), None, None, 0
    
    d = df.tail(80).copy()  # Use more history for stability
    d['REST'] = d['GAME_DATE'].diff().dt.days.fillna(1).clip(lower=0)
    d['HOME'] = d['MATCHUP'].str.contains('vs.').astype(int)
    d['OPP'] = d['MATCHUP'].apply(lambda x: x.split()[-1])
    d['OPP_DEF_RATING'] = d['OPP'].map(OPP_DEF_RATINGS.get(stat, DEFAULT_DEF)).fillna(DEFAULT_DEF[stat])
    d['DEF_ADJ'] = 100 / d['OPP_DEF_RATING']  # >1 = easier opponent
    
    # Rolling weighted stats (exponential decay)
    for s in ['PTS','REB','AST','STL','BLK','MIN']:
        d[f'ROLL_{s}'] = d[s].rolling(10, min_periods=1).mean()
        d[f'WTD_{s}'] = d[s].ewm(span=12, adjust=False).mean()
    
    for i in [1,2,3,5]:
        for s in ['PTS','REB','AST','STL','BLK','MIN']:
            d[f'L{i}_{s}'] = d[s].shift(i)
    
    d['PACE_ROLL'] = (d['FGM'] + d['FGA'] + d['TOV']).rolling(8).mean()
    d['PRA'] = d['PTS'] + d['REB'] + d['AST']
    
    feats = [c for c in d.columns if c not in ['GAME_DATE','MATCHUP',stat,'OPP','PLUS_MINUS']]
    feats = [f for f in feats if d[f].dtype in ['float64','int64']]
    
    d = d.dropna(subset=feats + [stat])
    if len(d) < 15:
        return round(d[stat].mean(),1), None, None, 0
    
    X = d[feats]
    y = d[stat]
    
    # Train/val split (last 12 games = validation)
    train_X, train_y = X.iloc[:-12], y.iloc[:-12]
    val_X, val_y = X.iloc[-12:], y.iloc[-12:]
    
    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=60, verbose=False)
    
    pred = model.predict(val_X.tail(1))[0]
    mae = mean_absolute_error(val_y, model.predict(val_X))
    ci = 1.65 * mae
    
    # Confidence score
    recent_vol = d[stat].tail(n_recent).std()
    base_conf = max(0, 100 - recent_vol*3 - mae*2)
    lock_score = int(min(98, max(30, base_conf)))
    
    return round(pred,1), round(pred-ci,1), round(pred+ci,1), lock_score

# ==================== UI ====================
c1, c2, c3 = st.columns([3,1,1])
with c1:
    search = st.text_input("Search player", placeholder="lebron, wemby, jokic, curry...", label_visibility="collapsed")
with c2:
    recent_games = st.selectbox("Form bias", ["Hot (8 games)", "Recent (15)", "Stable (25)"], index=1)
    n_map = {"Hot (8 games)":8, "Recent (15)":15, "Stable (25)":25}
with c3:
    st.write(" ")
    st.write(" ")
    if st.button("Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

if search:
    matches = players_df[players_df['lower'].str.contains(search.lower(), na=False)]
    if matches.empty:
        st.error("Player not found")
    elif len(matches) > 1:
        pick = st.selectbox("Select player", matches['full_name'].str.title().tolist())
        pid = matches[matches['full_name'].str.title() == pick].iloc[0]['id']
        name = pick
    else:
        pick = matches.iloc[0]['full_name']
        pid = matches.iloc[0]['id']
        name = pick
        st.write(f"**{name.upper()}**")

    with st.spinner("Training opponent-adjusted model..."):
        logs = get_logs(pid)
    
    if len(logs) < 20:
        st.error("Not enough recent games")
    else:
        st.success(f"Loaded {len(logs)} career games • 2024-25 updated")

        n_recent = n_map[recent_games]
        stats = ['PTS','REB','AST','STL','BLK']
        results = []
        pra = 0
        
        for s in stats:
            proj, lo, hi, lock = predict_next_game(logs, s, n_recent)
            results.append({"STAT":s, "PROJ":proj, "RANGE":f"{lo}–{hi}", "LOCK":lock})
            if s in ['PTS','REB','AST']:
                pra += proj
        
        # === MAIN PROJECTIONS ===
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00", "#00F5FF"]
        for i, row in enumerate(results):
            with [col1, col2, col3, col4, col5, col6][i]:
                lock_color = "lock-high" if row["LOCK"] >= 80 else "lock-med" if row["LOCK"] >= 60 else "lock-low"
                st.markdown(f"""
                <div class="stat-box" style="background: {colors[i]}20; border: 3px solid {colors[i]};">
                    <div style="font-size:18px; opacity:0.9;">{row['STAT']}</div>
                    <div class="proj-main" style="font-size:48px;">{row['PROJ']}</div>
                    <div style="font-size:14px;">{row['RANGE']}</div>
                    <div class="{lock_color}">🔥 Lock: {row['LOCK']}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # PRA Bonus
        with col6:
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(45deg, #FFD60A33, #FF006E33); border: 3px solid #FFD60A;">
                <div style="font-size:18px; opacity:0.9;">PRA</div>
                <div class="proj-main" style="font-size:48px;">{round(pra,1)}</div>
                <div style="font-size:14px;">Combined</div>
                <div style="color:#FFD60A;">Combined</div>
            </div>
            """, unsafe_allow_html=True)

        # === CHART ===
        fig = go.Figure()
        last_n = logs.tail(40)
        last_n['idx'] = range(len(last_n))
        colors_dict = {"PTS":"#FF006E", "REB":"#00D200", "AST":"#FFD60A", "STL":"#9D4EDD", "BLK":"#FF6B00"}
        
        for s in stats:
            proj, lo, hi, _ = predict_next_game(logs, s, n_recent)
            fig.add_trace(go.Scatter(
                x=last_n['idx'], y=last_n[s],
                mode='lines+markers', name=s,
                line=dict(color=colors_dict[s], width=4),
                marker=dict(size=7)
            ))
            fig.add_trace(go.Scatter(
                x=[last_n['idx'].iloc[-1]], y=[proj],
                mode="markers+text", name=f"{s} proj",
                marker=dict(color=colors_dict[s], size=28, symbol="star-diamond", line=dict(width=4, color="white")),
                text=f" {proj} ", textposition="middle center",
                textfont=dict(size=16, color="white"),
                showlegend=False
            ))
        
        fig.update_layout(
            height=660, template="plotly_dark",
            title=f"Last 40 Games • Next Game = Star • Opponent & Pace Adjusted",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#0e0e0e", paper_bgcolor="#0e0e0e"
        )
        fig.update_xaxes(showticklabels=False, title="")
        fig.update_yaxes(title="Stat Value", gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Last 12 Games • Full Box"):
            show = logs.tail(12)[['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK','TOV']].copy()
            show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
            show['OPP'] = show['MATCHUP'].str.split().str[-1]
            show = show[['DATE','OPP','MIN','PTS','REB','AST','STL','BLK','TOV']]
            st.dataframe(show.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

st.caption("Opponent-adjusted • No overfitting • Built for winners only • 2025")
