import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time
import os
import plotly.graph_objects as go

# ==================== PASSCODE ====================
PASSCODE = "hi"   # change anytime

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    st.markdown("<h1 style='text-align: center;'>NBA Projector 2025</h1>", unsafe_allow_html=True)
    code = st.text_input("Passcode", type="password")
    if st.button("Enter"):
        if code == PASSCODE:
            st.session_state.auth = True
            st.rerun()
        else: st.error("Wrong")
    st.stop()

st.set_page_config(page_title="NBA Projector", layout="wide", page_icon="fire")
st.markdown("<h1 style='text-align: center;'>NBA Next-Game Projector</h1>", unsafe_allow_html=True)

CACHE_DIR = "nba_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data
def load_players():
    df = pd.DataFrame(players.get_active_players())
    df['lower'] = df['full_name'].str.lower()
    return df
players_df = load_players()

def get_logs(pid):
    cache = f"{CACHE_DIR}/p_{pid}.parquet"
    if os.path.exists(cache): return pd.read_parquet(cache)
    logs = []
    for s in [2022,2023,2024,2025]:
        try:
            data = playergamelog.PlayerGameLog(player_id=pid, season=s).get_data_frames()[0]
            logs.append(data)
            time.sleep(0.7)
        except: pass
    if not logs: return pd.DataFrame()
    df = pd.concat(logs)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    keep = ['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK','FGM','FGA','FTM','FTA','TOV']
    df = df[keep].copy()
    for c in ['PTS','REB','AST','STL','BLK','FGM','FGA','FTM','FTA','TOV']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['MIN'] = df['MIN'].apply(lambda x: int(x.split(':')[0]) if isinstance(x,str) and ':' in x else float(x) if pd.notna(x) else np.nan)
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    df.to_parquet(cache)
    return df

def predict_recent(df, stat, n_games):
    recent = df.tail(n_games)
    if len(recent) < 10:
        return round(df[stat].tail(10).mean(),1), None, None
    d = recent.copy()
    for s in ['PTS','REB','AST','STL','BLK','MIN']:
        for i in range(1,6): d[f'L{i}_{s}'] = d[s].shift(i)
    d['REST'] = d['GAME_DATE'].diff().dt.days.clip(lower=0).fillna(1)
    d['HOME'] = d['MATCHUP'].str.contains('vs.').astype(int)
    feats = [c for c in d.columns if c not in ['GAME_DATE','MATCHUP',stat]]
    d = d.dropna()
    X = d[feats].select_dtypes('number')
    y = d[stat]
    model = LGBMRegressor(n_estimators=400, learning_rate=0.06, max_depth=6, random_state=42, verbose=-1)
    model.fit(X,y)
    pred = model.predict(X.tail(1))[0]
    mae = mean_absolute_error(y.tail(12), model.predict(X.tail(12)))
    ci = 1.6 * mae
    return round(pred,1), round(pred-ci,1), round(pred+ci,1)

# ==================== UI ====================
c1, c2 = st.columns([2,1])
with c1:
    search = st.text_input("Search player", placeholder="wembanyama, lebron, curry...")
with c2:
    games = st.slider("Games used for model", 5, 20, 12, help="5 = hot streak | 20 = stable form")

if search:
    matches = players_df[players_df['lower'].str.contains(search.lower())]
    if matches.empty:
        st.error("Player not found")
    else:
        pick = st.selectbox("Select", matches['full_name'].str.title().tolist())
        pid = matches[matches['full_name'].str.title() == pick].iloc[0]['id']
        name = pick

        with st.spinner("Loading & training..."):
            logs = get_logs(pid)

        if len(logs) < 10:
            st.error("Not enough games")
        else:
            st.success(f"{name.upper()} • Using last {games} games")

            # Projections table
            res = []
            stats = ['PTS','REB','AST','STL','BLK']
            for s in stats:
                p, lo, hi = predict_recent(logs, s, games)
                res.append({"STAT":s, "PROJ":p, "95% RANGE":f"{lo}–{hi}"})
            st.subheader("NEXT GAME PROJECTIONS")
            st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)

            # ONE CLEAN CHART
            fig = go.Figure()
            colors = {"PTS":"#FF006E", "REB":"#00D4AA", "AST":"#FFD60A", "STL":"#9D4EDD", "BLK":"#FF6B00"}
            last_40 = logs.tail(40).copy()
            last_40['game_num'] = range(len(last_40))

            for stat in stats:
                p, lo, hi = predict_recent(logs, stat, games)
                fig.add_trace(go.Scatter(
                    x=last_40['game_num'], y=last_40[stat],
                    mode='lines+markers', name=stat,
                    line=dict(color=colors[stat], width=4),
                    marker=dict(size=6)
                ))
                # Projection star
                fig.add_trace(go.Scatter(
                    x=[len(last_40)-1], y=[p],
                    mode="markers+text", name=f"{stat} proj",
                    marker=dict(color=colors[stat], size=20, symbol="star", line=dict(width=3, color="white")),
                    text=[f" {p} "], textposition="middle center",
                    showlegend=False
                ))
                # Confidence band
                fig.add_trace(go.Scatter(
                    x=[len(last_40)-1, len(last_40)-1],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(width=8, color=colors[stat]),
                    opacity=0.3,
                    showlegend=False
                ))

            fig.update_layout(
                height=620,
                template="plotly_dark",
                title=f"Last 40 Games • Next-Game Projection = Star • Using last {games} games",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showticklabels=False, title="")
            fig.update_yaxes(title="Stat Value")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Last 15 Games"):
                show = logs.tail(15)[['GAME_DATE','MATCHUP','PTS','REB','AST','STL','BLK']].copy()
                show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
                show['OPP'] = show['MATCHUP'].str.split().str[-1]
                st.dataframe(show[['DATE','OPP','PTS','REB','AST','STL','BLK']], use_container_width=True)


st.caption("Private • Clean charts • Built for winners")
