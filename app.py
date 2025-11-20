import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguegamelog
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error
import time
import os
import plotly.graph_objects as go

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
    .stat-box {font-size: 32px; font-weight: bold; text-align: center; padding: 15px; border-radius: 15px; margin: 10px;}
    .proj-main {background: linear-gradient(90deg, #FF006E, #9D4EDD); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .lock-high {color: #00D4AA;}
    .lock-med {color: #FFD60A;}
    .lock-low {color: #FF6B00;}
    section[data-testid="stSidebar"] {background: #0e0e0e;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>NBA NEXT-GAME PROJECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:20px; opacity:0.8;'>Opponent-adjusted • Pace-aware • Zero crashes</p>", unsafe_allow_html=True)

CACHE_DIR = "nba_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== DEFENSIVE RATINGS 2024-25 (updated Nov 2025) ====================
OPP_DEF_RATINGS = {
    'PTS': {'BOS':93, 'CLE':94, 'OKC':94, 'MIN':95, 'ORL':96, 'NYK':97, 'HOU':96, 'MIA':98, 'GSW':99, 'LAC':100,
            'DEN':100, 'LAL':101, 'PHI':98, 'MIL':104, 'IND':107, 'SAC':103, 'DAL':101, 'PHX':102, 'NOP':105,
            'MEM':99, 'SAS':104, 'ATL':110, 'CHI':108, 'TOR':109, 'BKN':107, 'DET':106, 'CHA':112, 'POR':111, 'WAS':113, 'UTA':110},
    'REB': {'MIN':92, 'ORL':93, 'BOS':94, 'CLE':94, 'NOP':96, 'NYK':97, 'OKC':96, 'SAC':100, 'LAL':100},
    'AST': {'BOS':93, 'OKC':94, 'CLE':94, 'ORL':95, 'MIN':96, 'NYK':97, 'MIA':98, 'HOU':97},
    'STL': {'OKC':92, 'HOU':94, 'BOS':93, 'ORL':94, 'MIN':95, 'NYK':96},
    'BLK': {'MEM':90, 'ORL':92, 'SAS':93, 'MIA':94, 'BKN':105}
}
DEFAULT_DEF = {'PTS':102, 'REB':100, 'AST':100, 'STL':100, 'BLK':100}

# ==================== DATA ====================
@st.cache_data(ttl=3600)
def load_players():
    df = pd.DataFrame(players.get_active_players())
    df['lower'] = df['full_name'].str.lower()
    return df
players_df = load_players()

def get_logs(pid):
    cache = f"{CACHE_DIR}/p_{pid}.parquet"
    if os.path.exists(cache) and os.path.getmtime(cache) > time.time() - 86400:
        return pd.read_parquet(cache)
    
    logs = []
    for season in ["2022-23", "2023-24", "2024-25"]:
        try:
            data = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
            logs.append(data)
            time.sleep(0.8)
        except:
            continue
    if not logs:
        return pd.DataFrame()
    
    df = pd.concat(logs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    keep = ['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK','FGM','FGA','TOV']
    df = df[keep].copy()
    for c in ['PTS','REB','AST','STL','BLK','MIN','FGM','FGA','TOV']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    df.to_parquet(cache)
    return df

# ==================== ROBUST MODEL (NO MORE CRASHES) ====================
def predict_next_game(df, stat, n_recent=15):
    if len(df) < 15:
        avg = round(df[stat].mean(), 1) if not df.empty else 0
        return avg, None, None, 40

    d = df.tail(80).copy()
    d['REST_DAYS'] = d['GAME_DATE'].diff().dt.days.fillna(1).clip(lower=0)
    d['HOME'] = d['MATCHUP'].str.contains('vs.').astype(int)
    d['OPP'] = d['MATCHUP'].apply(lambda x: x.split()[-1] if isinstance(x, str) else 'UNKNOWN')
    d['OPP_DEF'] = d['OPP'].map(OPP_DEF_RATINGS.get(stat, DEFAULT_DEF)).fillna(DEFAULT_DEF[stat])
    d['DEF_ADJ'] = 100 / d['OPP_DEF']

    # Rolling & lagged features
    for s in ['PTS','REB','AST','STL','BLK','MIN']:
        d[f'WTD_{s}'] = d[s].ewm(span=10, adjust=False).mean()
        for lag in [1,2,3]:
            d[f'L{lag}_{s}'] = d[s].shift(lag)

    # Target
    target = d[stat]
    feats = [c for c in d.columns if c.startswith(('WTD_','L','REST','HOME','DEF_ADJ'))]
    d = d.dropna(subset=feats + [stat])

    if len(d) < 20:
        avg = round(d[stat].mean(), 1)
        return avg, None, None, 50

    X = d[feats]
    y = d[stat]

    # Safe split
    split = max(12, len(d)//5)
    train_X, train_y = X.iloc[:-split], y.iloc[:-split]
    val_X, val_y = X.iloc[-split:], y.iloc[-split:]

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Fixed callback syntax (this was the crash source)
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        callbacks=[early_stopping(50, verbose=False)]
    )

    pred = model.predict(val_X.tail(1))[0]
    mae = mean_absolute_error(val_y, model.predict(val_X))
    ci = 1.65 * mae

    # Lock score
    vol = d[stat].tail(n_recent).std()
    lock = int(np.clip(100 - vol*2.5 - mae*1.8, 30, 98))

    return round(pred, 1), round(pred-ci, 1), round(pred+ci, 1), lock

# ==================== UI ====================
c1, c2, c3 = st.columns([3,1,1])
with c1:
    search = st.text_input("Search player", placeholder="wemby, curry, jokic...", label_visibility="collapsed")
with c2:
    bias = st.selectbox("Form bias", ["Hot (8)", "Recent (15)", "Stable (25)"], index=1)
    n_map = {"Hot (8)":8, "Recent (15)":15, "Stable (25)":25}
with c3:
    st.write("")
    st.write("")
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

if search:
    matches = players_df[players_df['lower'].str.contains(search.lower(), na=False)]
    if matches.empty:
        st.error("Player not found")
    else:
        pick = st.selectbox("Select player", matches['full_name'].str.title().sort_values().tolist(), index=0)
        pid = matches[matches['full_name'].str.title() == pick].iloc[0]['id']
        name = pick.upper()

        with st.spinner("Loading data & training model..."):
            logs = get_logs(pid)

        if len(logs) < 15:
            st.error("Not enough games played")
        else:
            st.success(f"**{name}** • {len(logs)} games loaded")

            n_recent = n_map[bias]
            stats = ['PTS','REB','AST','STL','BLK']
            results = []
            pra = 0

            for s in stats:
                proj, lo, hi, lock = predict_next_game(logs, s, n_recent)
                results.append({"STAT": s, "PROJ": proj, "RANGE": f"{lo}–{hi}" if lo else "-", "LOCK": lock})
                if s in ['PTS','REB','AST']:
                    pra += proj

            # Big stat boxes
            cols = st.columns(6)
            colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00", "#00BCD4"]
            for i, r in enumerate(results):
                with cols[i]:
                    lock_col = "#00D4AA" if r["LOCK"] >= 80 else "#FFD60A" if r["LOCK"] >= 60 else "#FF6B00"
                    st.markdown(f"""
                    <div class="stat-box" style="background:{colors[i]}20; border:3px solid {colors[i]};">
                        <div style="font-size:20px;">{r['STAT']}</div>
                        <div class="proj-main" style="font-size:50px; margin:10px 0;">{r['PROJ']}</div>
                        <div style="font-size:15px; opacity:0.9;">{r['RANGE']}</div>
                        <div style="color:{lock_col}; font-weight:bold;">Lock {r['LOCK']}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # PRA box
            with cols[5]:
                st.markdown(f"""
                <div class="stat-box" style="background:linear-gradient(45deg,#FFD60A33,#FF006E33); border:3px solid #FFD60A;">
                    <div style="font-size:20px;">PRA</div>
                    <div class="proj-main" style="font-size:50px; margin:10px 0;">{round(pra,1)}</div>
                    <div style="font-size:15px;">Points + Reb + Ast</div>
                </div>
                """, unsafe_allow_html=True)

            # Chart
            fig = go.Figure()
            recent = logs.tail(40).copy()
            recent['idx'] = range(len(recent))
            color_map = {"PTS":"#FF006E", "REB":"#00D4AA", "AST":"#FFD60A", "STL":"#9D4EDD", "BLK":"#FF6B00"}

            for s in stats:
                proj, lo, hi, _ = predict_next_game(logs, s, n_recent)
                fig.add_trace(go.Scatter(
                    x=recent['idx'], y=recent[s],
                    mode='lines+markers', name=s,
                    line=dict(color=color_map[s], width=4),
                    marker=dict(size=7)
                ))
                fig.add_trace(go.Scatter(
                    x=[recent['idx'].iloc[-1]], y=[proj],
                    mode="markers+text",
                    marker=dict(color=color_map[s], size=30, symbol="star-diamond", line=dict(width=4, color="white")),
                    text=f" {proj} ", textposition="middle center",
                    textfont=dict(size=18, color="white"),
                    showlegend=False
                ))

            fig.update_layout(
                height=650, template="plotly_dark",
                title=f"Last 40 Games • Next Game = Star • Opponent Adjusted",
                hovermode="x unified",
                plot_bgcolor="#0e0e0e", paper_bgcolor="#0e0e0e"
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Last 15 Games"):
                show = logs.tail(15)[['GAME_DATE','MATCHUP','PTS','REB','AST','STL','BLK','MIN']].copy()
                show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
                show['OPP'] = show['MATCHUP'].str.split().str[-1]
                st.dataframe(show[['DATE','OPP','MIN','PTS','REB','AST','STL','BLK']], use_container_width=True)

st.caption("Opponent-adjusted • No crashes • Lock meter • 2025 season")
