import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguegamelog
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time
import os
import plotly.graph_objects as go
from datetime import datetime
import pytz

# ==================== DARK THEME + GLOW ====================
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

# ==================== PASSCODE ====================
PASSCODE = "getmoney"

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h1 style='text-align:center; color:#FF006E; font-weight:900;'>NBA PROJECTOR 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:24px; opacity:0.9;'>Private tool • Winners only</p>", unsafe_allow_html=True)
    code = st.text_input("Enter passcode", type="password", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("ENTER", use_container_width=True, type="primary"):
            if code == PASSCODE:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Wrong")
    st.stop()

st.markdown("<h1 class='title-main'>NEXT GAME PROJECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:30px; opacity:0.9; margin-bottom:60px; color:#aaa;'>Opponent-adjusted • Real-time lock meter • For sharps only</p>", unsafe_allow_html=True)

CACHE_DIR = "nba_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== DEFENSIVE RATINGS ====================
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

@st.cache_data(ttl=3600)
def load_players():
    df = pd.DataFrame(players.get_active_players())
    df['lower'] = df['full_name'].str.lower()
    return df
players_df = load_players()

@st.cache_data(ttl=1800)  # Refresh schedule every 30 min
def get_league_schedule(season="2025-26"):
    try:
        game_log = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
        game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
        return game_log
    except:
        return pd.DataFrame()

def get_player_team(pid):
    try:
        player_info = players.find_player_by_id(pid)
        return player_info.get('team_abbreviation', 'TBD')
    except:
        return 'TBD'

def get_next_opponent(player_team, game_log, current_date):
    if player_team == 'TBD' or game_log.empty:
        return "NEXT GAME TBD", "#FFD60A", "0 0 40px #FFD60A"

    future_games = game_log[game_log['GAME_DATE'] >= current_date]
    if future_games.empty:
        return "NEXT GAME TBD", "#FFD60A", "0 0 40px #FFD60A"

    player_games = future_games[(future_games['TEAM_ABBREVIATION'] == player_team) | (future_games['OPPONENT_ABBREVIATION'] == player_team)]
    if player_games.empty:
        return "NEXT GAME TBD", "#FFD60A", "0 0 40px #FFD60A"

    next_game = player_games.iloc[0]
    game_date = next_game['GAME_DATE'].normalize()
    today = current_date.normalize()

    if game_date == today:
        day_text = "TODAY"
        color = "#FF006E"
        shadow = "0 0 60px #FF006E"
    else:
        day_text = "TOMORROW"
        color = "#00ff9d"
        shadow = "0 0 40px #00ff9d"

    if next_game['TEAM_ABBREVIATION'] == player_team:
        home_away = "vs"
        opponent = next_game['OPPONENT_ABBREVIATION']
    else:
        home_away = "@"
        opponent = next_game['TEAM_ABBREVIATION']

    return f"{day_text} {home_away} {opponent}", color, shadow

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

def predict_next_game(df, stat, n_recent=15, next_opp_def=None):
    if len(df) < 15:
        avg = round(df[stat].mean(), 1) if not df.empty else 0
        return avg, None, None, 40

    d = df.tail(80).copy()
    d['REST_DAYS'] = d['GAME_DATE'].diff().dt.days.fillna(1).clip(lower=0)
    d['HOME'] = d['MATCHUP'].str.contains('vs.').astype(int)
    d['OPP'] = d['MATCHUP'].apply(lambda x: x.split()[-1] if isinstance(x, str) else 'UNKNOWN')
    d['OPP_DEF'] = d['OPP'].map(OPP_DEF_RATINGS.get(stat, DEFAULT_DEF)).fillna(DEFAULT_DEF[stat])
    
    # DYNAMIC NEXT OPPONENT ADJUSTMENT
    if next_opp_def:
        d['OPP_DEF'] = next_opp_def  # Override with next game's defense rating
        d['DEF_ADJ'] = 100 / d['OPP_DEF']  # Re-calculate adjustment for next game

    for s in ['PTS','REB','AST','STL','BLK','MIN']:
        d[f'WTD_{s}'] = d[s].ewm(span=10, adjust=False).mean()
        for lag in [1,2,3]:
            d[f'L{lag}_{s}'] = d[s].shift(lag)

    feats = [c for c in d.columns if c.startswith(('WTD_','L','REST','HOME','DEF_ADJ'))]
    d = d.dropna(subset=feats + [stat])
    if len(d) < 20:
        return round(d[stat].mean(), 1), None, None, 50

    X = d[feats]
    y = d[stat]
    split = max(12, len(d)//5)
    train_X, train_y = X.iloc[:-split], y.iloc[:-split]
    val_X, val_y = X.iloc[-split:], y.iloc[-split:]

    model = LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)])

    pred = model.predict(val_X.tail(1))[0]
    mae = mean_absolute_error(val_y, model.predict(val_X))
    ci = 1.65 * mae
    vol = d[stat].tail(n_recent).std()
    lock = int(np.clip(100 - vol*2.5 - mae*1.8, 30, 98))

    return round(pred, 1), round(pred-ci, 1), round(pred+ci, 1), lock

# ==================== UI ====================
c1, c2, c3 = st.columns([3,1,1])
with c1:
    search = st.text_input("Search player", placeholder="wemby • curry • jokic • westbrook", label_visibility="collapsed")
with c2:
    bias = st.selectbox("Form bias", ["Hot (8)", "Recent (15)", "Stable (25)"], index=1)
    n_map = {"Hot (8)":8, "Recent (15)":15, "Stable (25)":25}
with c3:
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared")

if search:
    matches = players_df[players_df['lower'].str.contains(search.lower(), na=False)]
    if matches.empty:
        st.error("Player not found")
    else:
        pick = st.selectbox("Select player", matches['full_name'].str.title().sort_values().tolist(), index=0)
        pid = matches[matches['full_name'].str.title() == pick].iloc[0]['id']
        name = pick.upper()

        with st.spinner("Training model..."):
            logs = get_logs(pid)

        if len(logs) < 15:
            st.error("Not enough games")
        else:
            st.success(f"**{name}** • {len(logs)} games loaded")

            # LIVE SCHEDULE + OPPONENT LOOKUP
            game_log = get_league_schedule()
            player_team = get_player_team(pid)
            current_date = pd.Timestamp.now(tz='US/Eastern')

            game_text, color, shadow = get_next_opponent(player_team, game_log, current_date)
            next_opp_abbr = game_text.split()[-1] if "TBD" not in game_text else None  # Extract opponent abbr

            # Get next opponent's defense ratings for adjustment
            next_opp_def = {}
            if next_opp_abbr:
                for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
                    next_opp_def[stat] = OPP_DEF_RATINGS[stat].get(next_opp_abbr, DEFAULT_DEF[stat])

            st.markdown(f"""
            <h2 style='text-align:center; font-size:62px; margin:70px 0 40px 0; color:{color}; 
                       text-shadow: {shadow}; font-weight:900; letter-spacing: 3px;'>
                {game_text}
            </h2>
            """, unsafe_allow_html=True)

            n_recent = n_map[bias]
            stats = ['PTS','REB','AST','STL','BLK']
            results = []
            pra_total = 0

            for s in stats:
                proj, lo, hi, lock = predict_next_game(logs, s, n_recent, next_opp_def.get(s))
                results.append({"PROJ": proj, "RANGE": f"{lo}–{hi}" if lo else "-", "LOCK": lock})
                if s in ['PTS','REB','AST']:
                    pra_total += proj

            cols = st.columns(6)
            colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00", "#00ff9d"]
            names = ["POINTS", "REBOUNDS", "ASSISTS", "STEALS", "BLOCKS", "P+R+A"]

            for i, col in enumerate(cols):
                with col:
                    if i < 5:
                        r = results[i]
                        c = colors[i]
                        lock_class = "lock-high" if r["LOCK"] >= 80 else "lock-med" if r["LOCK"] >= 60 else "lock-low"
                        st.markdown(f"""
                        <div class="proj-card">
                            <div style="font-size:24px; opacity:0.9; letter-spacing:3px;">{names[i]}</div>
                            <div class="proj-num" style="background: linear-gradient(45deg, {c}, {c}CC); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {r["PROJ"]}</div>
                            <div style="font-size:20px; opacity:0.75; margin:15px 0;">{r["RANGE"]}</div>
                            <div class="{lock_class}" style="font-weight:900;">{r["LOCK"]}% LOCK</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="proj-card" style="border:6px solid #00ff9d; background: linear-gradient(135deg, #00ff9d30, #FF006E20);">
                            <div style="font-size:32px; color:#00ff9d; letter-spacing:4px; text-shadow: 0 0 30px #00ff9d;">P + R + A</div>
                            <div class="proj-num" style="background: linear-gradient(45deg, #00ff9d, #00D4AA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {round(pra_total, 1)}</div>
                            <div style="font-size:26px; color:#00ff9d; font-weight:bold; text-shadow: 0 0 20px #00ff9d;">COMBINED</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown('<p class="chart-title">LAST 40 GAMES • NEXT = NEON STAR</p>', unsafe_allow_html=True)
            fig = go.Figure()
            last40 = logs.tail(40).copy()
            last40['game'] = range(1, 41)
            glow_colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00"]

            for i, s in enumerate(stats):
                proj = results[i]["PROJ"]
                color = glow_colors[i]
                fig.add_trace(go.Scatter(x=last40['game'], y=last40[s], mode='lines+markers', name=s,
                                        line=dict(color=color, width=9), marker=dict(size=14, line=dict(width=4, color='white'))))
                fig.add_trace(go.Scatter(x=[40], y=[proj], mode='markers+text',
                                        marker=dict(symbol='star-diamond', size=80, color=color, line=dict(width=10, color='white')),
                                        text=f"{proj}", textposition="middle center",
                                        textfont=dict(size=32, color="black", family="Arial Black"), showlegend=False))
                fig.add_trace(go.Scatter(x=[40], y=[proj], mode='markers',
                                        marker=dict(size=130, color=color, opacity=0.25), showlegend=False))

            fig.update_layout(height=850, template="plotly_dark", plot_bgcolor="#000000", paper_bgcolor="#000000",
                              hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5, font=dict(size=22)),
                              margin=dict(t=140, b=80, l=80, r=80))
            fig.update_xaxes(showgrid=False, showticklabels=False)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with st.expander("Last 15 Games"):
                show = logs.tail(15)[['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK']].copy()
                show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
                show['OPP'] = show['MATCHUP'].str.split().str[-1]
                st.dataframe(show[['DATE','OPP','MIN','PTS','REB','AST','STL','BLK']], use_container_width=True)

st.caption("Private • Live Schedule • Opponent-Adjusted • Nuclear Glow • 2025 Season • Winners Only")
