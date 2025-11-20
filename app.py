# === SAME IMPORTS + DARK MODE CSS (only change is the CHART section below) ===

# ... [everything up to the chart section stays exactly the same as the last version] ...

            # === NUCLEAR GLOW CHART ===
            st.markdown('<p class="chart-title">LAST 40 GAMES • NEXT = NEON STAR</p>', unsafe_allow_html=True)

            fig = go.Figure()
            last40 = logs.tail(40).copy()
            last40['game'] = range(1, 41)

            # Glow colors (exact match to cards)
            glow_colors = ["#FF006E", "#00D4AA", "#FFD60A", "#9D4EDD", "#FF6B00"]
            stat_names = ['PTS','REB','AST','STL','BLK']

            for i, s in enumerate(stat_names):
                proj = results[i]["PROJ"]
                color = glow_colors[i]

                # Main glowing line
                fig.add_trace(go.Scatter(
                    x=last40['game'],
                    y=last40[s],
                    mode='lines+markers',
                    name=s,
                    line=dict(color=color, width=9),
                    marker=dict(
                        size=14,
                        line=dict(width=4, color='white'),
                        shadow=dict(
                            blur=15,
                            color=color + '88',  # 50% opacity glow
                            x=0, y=0
                        )
                    ),
                    hovertemplate=f"<b>{s}</b>: %{{y}}<extra></extra>"
                ))

                # MASSIVE GLOWING DIAMOND STAR
                fig.add_trace(go.Scatter(
                    x=[40], y=[proj],
                    mode='markers+text',
                    marker=dict(
                        symbol='star-diamond',
                        size=80,
                        color=color,
                        line=dict(width=10, color='white'),
                        shadow=dict(
                            blur=40,
                            color=color + 'CC',  # intense glow
                            x=0, y=0
                        )
                    ),
                    text=f"{proj}",
                    textposition="middle center",
                    textfont=dict(size=32, color="black", family="Arial Black", weight="bold"),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Extra outer glow ring (insane effect)
                fig.add_trace(go.Scatter(
                    x=[40], y=[proj],
                    mode='markers',
                    marker=dict(
                        size=120,
                        color=color,
                        opacity=0.3,
                        line=dict(width=0)
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Layout - pure darkness + glow
            fig.update_layout(
                height=850,
                template="plotly_dark",
                plot_bgcolor="#000000",
                paper_bgcolor="#000000",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.12,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=22, color="white", family="Arial Black"),
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="#333",
                    borderwidth=2
                ),
                margin=dict(t=140, b=80, l=80, r=80),
                transition_duration=500
            )

            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.03)",
                gridwidth=1,
                zeroline=False
            )

            # This makes the glow POP even more
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={'displayModeBar': False, 'scrollZoom': False}
            )

            with st.expander("Last 15 Games • Box Scores", expanded=False):
                show = logs.tail(15)[['GAME_DATE','MATCHUP','MIN','PTS','REB','AST','STL','BLK']].copy()
                show['DATE'] = show['GAME_DATE'].dt.strftime('%m/%d')
                show['OPP'] = show['MATCHUP'].str.split().str[-1]
                st.dataframe(
                    show[['DATE','OPP','MIN','PTS','REB','AST','STL','BLK']],
                    use_container_width=True,
                    hide_index=True
                )

st.caption("Private • Opponent-adjusted • Nuclear glow • 2025 Season • For winners only")
