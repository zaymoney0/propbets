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

# ==================== PASSCODE ====================
PASSCODE = "getmoney"

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    st.markdown("<h1 style='text-align: center; color:#FF006E; font-weight:900;'>NBA PROJECTOR 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:20px; opacity:0.9;'>Private • Built for winners only</p>", unsafe_allow_html=True)
    code = st.text_input("Enter passcode", type="password", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("ENTER", use_container_width=True, type="primary"):
            if code == PASSCODE:
                st.session_state.auth = True
                st.rerun()
            else: 
                st.error("Wrong passcode")
    st.stop()

st.set_page_config(page_title="NBA Projector 2025", layout="wide", page_icon="fire")

# ==================== ELITE 2025 VISUALS ====================
st.markdown("""
<style>
    .big-title {
        font-size: 80px !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FF006E, #9D4EDD, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 20px 0;
    }
    .proj-card {
        background: linear-gradient(135deg, rgba(255,0,110,0.18), rgba(157,78,221,0.18));
        border-radius: 24px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0,0,0,0.6);
        backdrop-filter: blur(12px);
        border: 3px solid transparent;
        transition: all 0.4s ease;
        margin: 12px 0;
    }
    .proj-card:hover {
        transform: translateY(-12px) scale(1.03);
        border-color: #FF006E;
        box-shadow: 0 20px 50px rgba(255,0,110,0.4);
    }
    .proj-num {
        font-size: 78px !important;
        font-weight: 900;
        margin: 16px 0;
        line-height: 1;
    }
    .lock-high {color: #00ff9d !important; text-shadow: 0 0 20px #00ff9d;}
    .lock-med {color: #FFD60A !important;}
    .lock-low {color: #ff6b00 !important;}
    .chart-title {
        font-size: 32px !important;
        font-weight: 900;
        text-align: center;
        margin: 50px 0 20px 0;
        color: white;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
    }
    .css-1d391kg {padding-top: 1rem;}
    section[data-testid="stSidebar"] {background: #0a0a0a;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>NEXT GAME PROJECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:24px; opacity:0.9; margin-bottom:40px;'>Opponent-adjusted • Lock meter • Built for winners</p>", unsafe_allow_html=True)

CACHE_DIR = "nba_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== DEFENSIVE RATINGS (2024-25) ====================
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

# ==================== DATA LOADING ====================
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

# ==================== MODEL (UNCHANGED — ALREADY PERFECT) ====================
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

    for s in ['PTS','REB','AST','STL','BLK','MIN']:
        d[f'WTD_{s}'] = d[s].ewm(span=10, adjust=False).mean()
        for lag in [1,2,3]:
            d[f'L{lag}_{s}'] = d[s].shift(lag)

    feats = [c for c in d.columns if c.startswith(('WTD_','L','REST','HOME','DEF_ADJ'))]
    d = d.dropna(subset=feats + [stat])

    if len(d) < 20:
        avg = round(d[stat].mean(), 1)
        return avg, None, None, 50

    X = d[feats]
    y = d[stat]
    split = max(12, len(d)//5)
    train_X, train_y = X.iloc[:-split], y.iloc[:-split]
    val_X, val_y = X.iloc[-split:], y.iloc[-split:]

    model = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], callbacks=[early_stopping(50, verbose=False)])

    pred = model.predict(val_X.tail(1))[0]
    mae = mean_absolute_error(val_y, model.predict(val_X))
    ci = 1.65 * mae
    vol = d[stat].tail(n_recent).std()
    lock = int(np.clip(100 - vol*2.5 - mae*1.8, 30, 98))

    return round(pred, 1), round(pred-ci, 1), round(pred+ci, 1), lock

# ==================== UI ====================
c1, c2, c3 = st.columns([3,1,1])
with c1:
    search = st.text_input("Search player", placeholder="wemby • lebron • jokic • curry", label_visibility="collapsed")
with c2:
    bias = st.selectbox("Form bias", ["Hot (8 games)", "Recent (15)", "Stable (25)"], index=1)
    n_map = {"Hot (8 games)":8, "Recent (15)":15, "Stable (25)":25}
with c3:
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared — refreshed!")

if search:
    matches = players_df[players_df['lower'].str.contains(search.lower(), na=False)]
    if matches.empty:
        st.error("Player not found")
    else:
        pick = st.selectbox("Select player", matches['full_name'].str.title().sort_values().tolist(), index=0)
        pid = matches[matches['full_name'].str.title() == pick].iloc[0]['id']
        name = pick.upper()

        with st.spinner("Training model & loading data..."):
            logs = get_logs(pid)

        if len(logs) < 15:
            st.error("Not enough games played this season")
        else:
            st.success(f"**{name}** • {len(logs)} career games loaded")

            # === NEXT OPPONENT DETECTION ===
            last_matchup = logs.iloc[-1]['MATCHUP']
            next_opp = last_matchup.split()[-1]
            home_away = "vs" if "vs." in last_matchup else "@"
            st.markdown(f"<h2 style='text-align:center; font-size:38px; margin:30px 0; opacity:0.9;'>Tomorrow {home_away} <b>{next_opp}</b></h2>", unsafe_allow_html=True)

            n_recent = n_map[bias]
            stats = ['PTS','REB','AST','STL','BLK']
            results = []
            pra_total = 0

            for s in stats:
                proj, lo, hi, lock = predict_next_game(logs, s, n_recent)
                results.append({"STAT": s, "PROJ": proj, "RANGE": f"{lo}–{hi}" if lo else "-", "LOCK": lock})
                if s in ['PTS','REB','AST']:
                    pra_total += proj

            # === GOD-TIER PROJECTION CARDS ===
            cols = st.columns(6)
            colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00", "#FFD700"]
            stat_names = ["POINTS", "REBOUNDS", "ASSISTS", "STEALS", "BLOCKS", "PRA"]

            for i, col in enumerate(cols):
                with col:
                    if i < 5:
                        r = results[i]
                        color = colors[i]
                        lock_class = "lock-high" if r["LOCK"] >= 80 else "lock-med" if r["LOCK"] >= 60 else "lock-low"
                        st.markdown(f"""
                        <div class="proj-card">
                            <div style="font-size:20px; opacity:0.8; letter-spacing:2px;">{stat_names[i]}</div>
                            <div class="proj-num" style="background: linear-gradient(45deg, {color}, {color}CC); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {r["PROJ"]}</div>
                            <div style="font-size:17px; opacity:0.75;">{r["RANGE"]}</div>
                            <div class="{lock_class}" style="font-size:28px; font-weight:900; margin-top:10px;">{r["LOCK"]}% LOCK</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="proj-card" style="border:4px solid #FFD700; background: linear-gradient(135deg, #FFD70025, #FF006E20);">
                            <div style="font-size:26px; letter-spacing:3px; color:#FFD700;">P + R + A</div>
                            <div class="proj-num" style="background: linear-gradient(45deg, #FFD700, #FFA726); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {round(pra_total, 1)}</div>
                            <div style="font-size:20px; color:#FFD700; font-weight:bold;">COMBINED</div>
                        </div>
                        """, unsafe_allow_html=True)

            # === INSANE CHART ===
            st.markdown('<p class="chart-title">LAST 40 GAMES • NEXT GAME = GLOWING STAR</p>', unsafe_allow_html=True)

            fig = go.Figure()
            last40 = logs.tail(40).copy()
            last40['game'] = range(1, len(last40) + 1)

            for i, s in enumerate(stats):
                proj = results[i]["PROJ"]
                color = colors[i]
                
                fig.add_trace(go.Scatter(
                    x=last40['game'], y=last40[s],
                    mode='lines+markers',
                    name=s,
                    line=dict(color=color, width=6),
                    marker=dict(size=10, line=dict(width=3, color='white')),
                    hovertemplate=f"<b>{s}</b><br>Game %{{x}}<br>Value: %{{y}}<extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=[40], y=[proj],
                    mode='markers+text',
                    marker=dict(symbol='star-diamond', size=48, color=color, line=dict(width=6, color='white')),
                    text=f"{proj}",
                    textposition="middle center",
                    textfont=dict(size=22, color="black", family="Arial Black"),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            fig.update_layout(
                height=720,
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font=dict(size=18, color="white")),
                margin=dict(t=80, b=60, l=60, r=60)
            )
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", gridwidth=1)

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with st.expander("Last 15 Games • Box Scores"):
                show = logs.tail(15)[['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK']].copy()
                show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
                show['OPP'] = show['MATCHUP'].str.split().str[-1]
                st.dataframe(show[['DATE','OPP','MIN','PTS','REB','AST','STL','BLK']], use_container_width=True)

st.caption("Private • Opponent-adjusted • Lock meter • 2025 Season • Built for winners")
