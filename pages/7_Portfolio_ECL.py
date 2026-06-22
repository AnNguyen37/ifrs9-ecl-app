"""
Page 7: Portfolio ECL Analysis
File: ifrs9_app/pages/7_💼_Portfolio_ECL.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Portfolio ECL",
    layout="wide"
)

st.title("Portfolio ECL Analysis")
st.markdown("**The CFO view — IFRS 9 provisions across the entire portfolio**")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_ecl_data():
    summary = pd.read_csv('data/ecl_portfolio_summary.csv').iloc[0].to_dict()
    timeline = pd.read_csv('data/ecl_timeline.csv')
    by_grade = pd.read_csv('data/ecl_by_grade.csv')
    by_term = pd.read_csv('data/ecl_by_term.csv')
    return summary, timeline, by_grade, by_term

summary, timeline, by_grade, by_term = load_ecl_data()

# ============================================================
# SECTION 1: KEY FINDINGS
# ============================================================
st.markdown("---")
st.header("1. Key Findings")

# Format large numbers nicely
def fmt_money(value, scale='B'):
    if scale == 'B':
        return f"${value/1e9:.2f}B"
    elif scale == 'M':
        return f"${value/1e6:,.0f}M"
    else:
        return f"${value:,.0f}"

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Portfolio Exposure",
    fmt_money(summary['total_exposure'], 'B'),
    delta=f"{int(summary['n_loans']):,} loans",
    delta_color="off"
)

col2.metric(
    "Stage 1 ECL (12m)",
    fmt_money(summary['total_ecl_stage1'], 'M'),
    delta=f"{summary['coverage_stage1']*100:.2f}% coverage",
    delta_color="off"
)

col3.metric(
    "Stage 2 ECL (lifetime)",
    fmt_money(summary['total_ecl_stage2'], 'M'),
    delta=f"{summary['coverage_stage2']*100:.2f}% coverage",
    delta_color="off"
)

stage2_multiplier = summary['total_ecl_stage2'] / summary['total_ecl_stage1']
col4.metric(
    "Stage 2 Multiplier",
    f"{stage2_multiplier:.2f}×",
    delta="Lifetime / 12-month",
    delta_color="off"
)

st.success(f"""
**Across {int(summary['n_loans']):,} performing loans totalling {fmt_money(summary['total_exposure'], 'B')}, 
the IFRS 9 model produces:**

- **Stage 1 provision** of {fmt_money(summary['total_ecl_stage1'], 'M')} ({summary['coverage_stage1']*100:.2f}% coverage ratio)
- **Stage 2 lifetime provision** of {fmt_money(summary['total_ecl_stage2'], 'M')} ({summary['coverage_stage2']*100:.2f}% coverage ratio)

Stage 2 is {stage2_multiplier:.2f}× higher than Stage 1, reflecting the additional months of
default risk captured under lifetime ECL. Stage 2 ECL sums default risk beyond month 12 —
the full remaining loan lifetime rather than just the next 12 months.
""")

# ============================================================
# SECTION 2: ECL TIMELINE
# ============================================================
st.markdown("---")
st.header("2. ECL Timeline — When Losses Materialise")

st.markdown("""
The chart below shows the **monthly expected loss** across the entire portfolio. 
Blue bars (months 1–12) represent the Stage 1 horizon; orange bars (months 13+) 
show the additional lifetime exposure captured under Stage 2.
""")

# Identify Stage 1 vs Stage 2
timeline['stage'] = timeline['future_month'].apply(
    lambda m: 'Stage 1 (months 1-12)' if m <= 12 else 'Stage 2 (months 13+)'
)

# Convert ECL to millions for chart readability
timeline['total_ecl_m'] = timeline['total_ecl'] / 1e6

fig = px.bar(
    timeline,
    x='future_month',
    y='total_ecl_m',
    color='stage',
    color_discrete_map={
        'Stage 1 (months 1-12)': '#0f3460',
        'Stage 2 (months 13+)': '#e94560'
    },
    labels={
        'future_month': 'Future Month',
        'total_ecl_m': 'Monthly ECL ($M)',
        'stage': 'IFRS 9 Stage'
    },
    height=450
)
fig.update_layout(
    margin=dict(l=40, r=40, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Three insights
peak_month = timeline.loc[timeline['total_ecl'].idxmax(), 'future_month']
peak_ecl = timeline['total_ecl'].max()
stage1_total = timeline[timeline['future_month'] <= 12]['total_ecl'].sum()
stage2_extra = timeline[timeline['future_month'] > 12]['total_ecl'].sum()
stage1_share = stage1_total / (stage1_total + stage2_extra)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Peak loss month",
    f"Month {int(peak_month)}",
    delta=fmt_money(peak_ecl, 'M'),
    delta_color="off"
)
col2.metric(
    "Stage 1 share of lifetime",
    f"{stage1_share*100:.1f}%",
    delta="First 12 months",
    delta_color="off"
)
col3.metric(
    "Months 13–24",
    fmt_money(timeline[(timeline['future_month'] > 12) & 
                        (timeline['future_month'] <= 24)]['total_ecl'].sum(), 'M'),
    delta="Stage 2 main contributor",
    delta_color="off"
)

# ============================================================
# SECTION 3: ECL BY GRADE
# ============================================================
st.markdown("---")
st.header("3. ECL by Grade — Concentration Analysis")

st.markdown("""
Portfolio risk concentrates by borrower grade. The table shows how exposure, 
provisions, and coverage ratios vary across the seven grades.
""")

# Build the display table
grade_display = by_grade.copy()
grade_display['exposure_b'] = grade_display['total_exposure'] / 1e9
grade_display['stage1_m'] = grade_display['ecl_stage1'] / 1e6
grade_display['stage2_m'] = grade_display['ecl_stage2'] / 1e6
grade_display['cov_stage1_pct'] = grade_display['coverage_stage1'] * 100
grade_display['pct_total_ecl'] = (grade_display['ecl_stage2'] / 
                                    grade_display['ecl_stage2'].sum() * 100)

grade_table = pd.DataFrame({
    'Grade': grade_display['grade'],
    'n Loans': grade_display['n_loans'].apply(lambda x: f"{x:,}"),
    'Exposure ($B)': grade_display['exposure_b'].apply(lambda x: f"${x:.2f}B"),
    'Stage 1 ECL ($M)': grade_display['stage1_m'].apply(lambda x: f"${x:,.0f}M"),
    'Stage 2 ECL ($M)': grade_display['stage2_m'].apply(lambda x: f"${x:,.0f}M"),
    'Coverage S1': grade_display['cov_stage1_pct'].apply(lambda x: f"{x:.2f}%"),
    '% of Lifetime ECL': grade_display['pct_total_ecl'].apply(lambda x: f"{x:.1f}%")
})

st.dataframe(grade_table, use_container_width=True, hide_index=True)

# Two charts side by side
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Exposure by Grade")
    fig = px.bar(
        by_grade,
        x='grade',
        y='total_exposure',
        labels={'grade': 'Grade', 'total_exposure': 'Exposure ($)'},
        color='grade',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        text=by_grade['total_exposure'].apply(lambda x: f"${x/1e9:.2f}B")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("##### Coverage Ratio by Grade")
    fig = px.bar(
        by_grade,
        x='grade',
        y='coverage_stage1',
        labels={'grade': 'Grade', 'coverage_stage1': 'Stage 1 Coverage'},
        color='grade',
        color_discrete_sequence=px.colors.sequential.Reds,
        text=by_grade['coverage_stage1'].apply(lambda x: f"{x*100:.2f}%")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False,
        yaxis=dict(tickformat='.0%')
    )
    st.plotly_chart(fig, use_container_width=True)

# Insight
highest_volume_grade = by_grade.loc[by_grade['total_exposure'].idxmax(), 'grade']
highest_cov_grade = by_grade.loc[by_grade['coverage_stage1'].idxmax(), 'grade']

st.info(f"""
**Two patterns to note:**

- **Exposure concentrates in mid-grades** (B, C, D) — these grades account for 
  the largest absolute provisioning despite individually lower coverage ratios. 
  Grade {highest_volume_grade} alone represents {by_grade.loc[by_grade['total_exposure'].idxmax(), 'total_exposure']/summary['total_exposure']*100:.1f}% 
  of total portfolio exposure.

- **Coverage ratios validate the scorecard** — grade A coverage 
  ({by_grade.loc[by_grade['grade']=='A', 'coverage_stage1'].iloc[0]*100:.2f}%) is meaningfully lower than grade G 
  ({by_grade.loc[by_grade['grade']=='G', 'coverage_stage1'].iloc[0]*100:.2f}%), 
  confirming the PD model discriminates correctly across risk segments.
""")

# ============================================================
# SECTION 4: ECL BY TERM
# ============================================================
st.markdown("---")
st.header("4. ECL by Term — Stage 2 Sensitivity")

st.markdown("""
Loan term determines how much lifetime exposure remains beyond the 12-month 
Stage 1 horizon. 60-month loans face proportionally larger Stage 2 provisioning 
because more months remain at risk.
""")

# Build comparison table
term_display = by_term.copy()
term_display['stage2_multiplier'] = term_display['ecl_stage2'] / term_display['ecl_stage1']

term_table = pd.DataFrame({
    'Term': term_display['term'].apply(lambda x: f"{x} months"),
    'n Loans': term_display['n_loans'].apply(lambda x: f"{x:,}"),
    'Exposure': term_display['total_exposure'].apply(lambda x: f"${x/1e9:.2f}B"),
    'Stage 1 ECL': term_display['ecl_stage1'].apply(lambda x: f"${x/1e6:,.0f}M"),
    'Stage 2 ECL': term_display['ecl_stage2'].apply(lambda x: f"${x/1e6:,.0f}M"),
    'Stage 1 Coverage': term_display['coverage_stage1'].apply(lambda x: f"{x*100:.2f}%"),
    'Stage 2 Coverage': term_display['coverage_stage2'].apply(lambda x: f"{x*100:.2f}%"),
    'Stage 2 / Stage 1': term_display['stage2_multiplier'].apply(lambda x: f"{x:.2f}×")
})

st.dataframe(term_table, use_container_width=True, hide_index=True)

# Side-by-side bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Stage 1 ECL',
    x=[f"{t} months" for t in by_term['term']],
    y=by_term['ecl_stage1'] / 1e6,
    marker_color='#0f3460',
    text=[f"${v/1e6:,.0f}M" for v in by_term['ecl_stage1']],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Stage 2 ECL (lifetime)',
    x=[f"{t} months" for t in by_term['term']],
    y=by_term['ecl_stage2'] / 1e6,
    marker_color='#e94560',
    text=[f"${v/1e6:,.0f}M" for v in by_term['ecl_stage2']],
    textposition='outside'
))

fig.update_layout(
    barmode='group',
    yaxis_title='ECL ($M)',
    height=400,
    margin=dict(l=40, r=40, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

term36_multiplier = by_term[by_term['term']==36]['ecl_stage2'].iloc[0] / by_term[by_term['term']==36]['ecl_stage1'].iloc[0]
term60_multiplier = by_term[by_term['term']==60]['ecl_stage2'].iloc[0] / by_term[by_term['term']==60]['ecl_stage1'].iloc[0]

st.info(f"""
**The Stage 2 multiplier differs by term:**

- **36-month loans:** Stage 2 is {term36_multiplier:.2f}× Stage 1
- **60-month loans:** Stage 2 is {term60_multiplier:.2f}× Stage 1

When a 60-month loan is evaluated on a lifetime basis (Stage 2), the provision increase is
proportionally larger because more remaining months are in scope. This is why longer-term
loans carry a higher Stage 2 multiplier — a critical consideration for forward-looking
IFRS 9 risk management.
""")

# ============================================================
# SECTION 5: STAGE MIGRATION SLIDER
# ============================================================
st.markdown("---")
st.header("5. Interactive Horizon — Stage 1 vs Stage 2")

st.markdown("""
The slider below shows how cumulative portfolio ECL changes as the horizon 
extends from 12 months (Stage 1) toward the full loan lifetime (Stage 2). 
Drag to see exactly where the provision falls between the two stages.
""")

horizon = st.slider(
    "ECL horizon (months)",
    min_value=1,
    max_value=int(timeline['future_month'].max()),
    value=12,
    step=1,
    help="12 = Stage 1, beyond 12 = Stage 2 territory"
)

# Calculate cumulative ECL up to selected horizon
cumulative_ecl = timeline[timeline['future_month'] <= horizon]['total_ecl'].sum()
total_lifetime = timeline['total_ecl'].sum()
proportion = cumulative_ecl / total_lifetime

col1, col2, col3 = st.columns(3)
col1.metric(
    "Cumulative ECL",
    fmt_money(cumulative_ecl, 'M'),
    delta=f"At month {horizon}",
    delta_color="off"
)
col2.metric(
    "% of Lifetime ECL",
    f"{proportion*100:.1f}%",
    delta=f"Of {fmt_money(total_lifetime, 'M')} total",
    delta_color="off"
)
col3.metric(
    "Coverage Ratio",
    f"{cumulative_ecl/summary['total_exposure']*100:.2f}%",
    delta="ECL / Exposure",
    delta_color="off"
)

# Stage indicator
if horizon == 12:
    st.success(f"**Stage 1 boundary — 12-month ECL** = {fmt_money(cumulative_ecl, 'M')}")
elif horizon < 12:
    st.info(f"**Partial Stage 1** — {horizon} months of provisioning")
else:
    st.warning(f"**Stage 2 territory** — {horizon}-month horizon, beyond the 12-month Stage 1 boundary")

# Cumulative ECL curve
cumul_df = timeline.copy()
cumul_df['cumulative_ecl_m'] = cumul_df['total_ecl'].cumsum() / 1e6

fig = go.Figure()

# Full curve in light grey
fig.add_trace(go.Scatter(
    x=cumul_df['future_month'],
    y=cumul_df['cumulative_ecl_m'],
    mode='lines',
    line=dict(color='lightgrey', width=2),
    name='Full lifetime',
    showlegend=False
))

# Highlighted region up to selected horizon
highlighted = cumul_df[cumul_df['future_month'] <= horizon]
fig.add_trace(go.Scatter(
    x=highlighted['future_month'],
    y=highlighted['cumulative_ecl_m'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='#0f3460', width=3),
    fillcolor='rgba(15, 52, 96, 0.2)',
    name=f'Selected ({horizon} months)',
    showlegend=False
))

# Vertical lines
fig.add_vline(
    x=horizon,
    line_dash="dash",
    line_color="#e94560",
    annotation_text=f"Month {horizon}",
    annotation_position="top right"
)

fig.add_vline(
    x=12,
    line_dash="dot",
    line_color="green",
    annotation_text="Stage 1 boundary",
    annotation_position="bottom right"
)

fig.update_layout(
    xaxis_title='Future Month',
    yaxis_title='Cumulative ECL ($M)',
    height=400,
    margin=dict(l=40, r=40, t=20, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 6: STRESS SCENARIO SIMULATOR
# ============================================================
st.markdown("---")
st.header("6. Stress Scenario Simulator")

st.markdown("""
Apply economic stress assumptions to the portfolio. This is the type of analysis 
EBA and ECB stress tests require — and what banks must submit annually for 
regulatory review.
""")

# Two-column layout
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.subheader("Stress Assumptions")
    
    scenario = st.selectbox(
        "Pre-set scenarios:",
        ["Custom", "Base Case", "Mild Recession", "Severe Recession", "Adverse (Regulatory)"]
    )
    
    # Set defaults based on scenario
    if scenario == "Base Case":
        default_pd_mult = 1.0
        default_lgd = 0.8555
    elif scenario == "Mild Recession":
        default_pd_mult = 1.5
        default_lgd = 0.90
    elif scenario == "Severe Recession":
        default_pd_mult = 2.0
        default_lgd = 0.95
    elif scenario == "Adverse (Regulatory)":
        default_pd_mult = 2.5
        default_lgd = 0.97
    else:
        default_pd_mult = 1.0
        default_lgd = 0.8555
    
    pd_multiplier = st.slider(
        "PD Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=default_pd_mult,
        step=0.1,
        help="Multiplier on baseline PD. 1.5 = mild recession, 2.0 = severe"
    )
    
    lgd_stress = st.slider(
        "LGD Assumption",
        min_value=0.70,
        max_value=1.00,
        value=default_lgd,
        step=0.01,
        help="Stressed LGD. Base is 0.8555; downturn scenarios use higher values."
    )
    
    st.caption(f"**Scenario:** {scenario}")
    st.caption(f"**PD ×** {pd_multiplier:.1f}, **LGD =** {lgd_stress:.4f}")

with col_results:
    st.subheader("Portfolio Impact")
    
    # Calculate stress ratio
    base_lgd = 0.8555
    stress_ratio = pd_multiplier * (lgd_stress / base_lgd)
    
    # Stressed values
    base_stage1 = summary['total_ecl_stage1']
    base_stage2 = summary['total_ecl_stage2']
    stressed_stage1 = base_stage1 * stress_ratio
    stressed_stage2 = base_stage2 * stress_ratio
    
    # Metrics
    mcol1, mcol2 = st.columns(2)
    
    with mcol1:
        st.metric(
            "Stressed Stage 1 ECL",
            fmt_money(stressed_stage1, 'M'),
            delta=f"+{fmt_money(stressed_stage1 - base_stage1, 'M')}",
            delta_color="off"
        )
        st.metric(
            "Stressed Coverage (S1)",
            f"{stressed_stage1/summary['total_exposure']*100:.2f}%",
            delta=f"+{(stressed_stage1-base_stage1)/summary['total_exposure']*100:.2f} pp",
            delta_color="off"
        )
    
    with mcol2:
        st.metric(
            "Stressed Stage 2 ECL",
            fmt_money(stressed_stage2, 'M'),
            delta=f"+{fmt_money(stressed_stage2 - base_stage2, 'M')}",
            delta_color="off"
        )
        st.metric(
            "Stressed Coverage (S2)",
            f"{stressed_stage2/summary['total_exposure']*100:.2f}%",
            delta=f"+{(stressed_stage2-base_stage2)/summary['total_exposure']*100:.2f} pp",
            delta_color="off"
        )
    
    # Comparison bar chart
    comparison_df = pd.DataFrame({
        'Scenario': ['Base', 'Stressed', 'Base', 'Stressed'],
        'Stage': ['Stage 1', 'Stage 1', 'Stage 2', 'Stage 2'],
        'ECL_M': [base_stage1/1e6, stressed_stage1/1e6, 
                   base_stage2/1e6, stressed_stage2/1e6]
    })
    
    fig = px.bar(
        comparison_df,
        x='Stage',
        y='ECL_M',
        color='Scenario',
        barmode='group',
        color_discrete_map={'Base': '#0f3460', 'Stressed': '#e94560'},
        labels={'ECL_M': 'ECL ($M)'},
        height=300
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Stressed timeline comparison
st.markdown("##### Stressed Timeline Overlay")

timeline_stress = timeline.copy()
timeline_stress['stressed_ecl_m'] = (timeline_stress['total_ecl'] * stress_ratio) / 1e6

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=timeline['future_month'],
    y=timeline['total_ecl_m'],
    mode='lines',
    fill='tozeroy',
    name='Base',
    line=dict(color='#0f3460', width=2),
    fillcolor='rgba(15, 52, 96, 0.2)'
))

fig.add_trace(go.Scatter(
    x=timeline_stress['future_month'],
    y=timeline_stress['stressed_ecl_m'],
    mode='lines',
    name='Stressed',
    line=dict(color='#e94560', width=2, dash='dash')
))

fig.update_layout(
    xaxis_title='Future Month',
    yaxis_title='Monthly ECL ($M)',
    height=350,
    margin=dict(l=40, r=40, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Stress impact summary
pct_change = (stress_ratio - 1) * 100
if pct_change > 0:
    impact_text = f"+{pct_change:.1f}% (provisions increase)"
else:
    impact_text = f"{pct_change:.1f}% (provisions decrease)"

st.info(f"""
**Stress scenario impact:** {impact_text}

Under {scenario.lower() if scenario != 'Custom' else 'these assumptions'}, portfolio 
ECL changes by **{pct_change:+.1f}%**. The stress ratio of **{stress_ratio:.3f}** 
applies multiplicatively because ECL = PD × LGD × EAD — each component change 
propagates linearly to provisions.

**Regulatory context:** This type of sensitivity analysis is required under EBA 
stress testing exercises and ECB SREP reviews. Banks must demonstrate that 
provisions remain adequate under adverse macroeconomic conditions.
""")

# ============================================================
# METHODOLOGY EXPANDER
# ============================================================
st.markdown("---")

with st.expander("📘 ECL Methodology Summary"):
    st.markdown(r"""
    ### The ECL formula
    """)
    
    st.latex(r"ECL = \sum_{t=1}^{T} PD_t \times LGD \times EAD_t")
    
    st.markdown(r"""
    Where:
    - $PD_t$ = monthly default probability (Page 2 scorecard × Page 6 timing curve)
    - $LGD$ = 0.8555 (Page 5 portfolio constant)
    - $EAD_t$ = outstanding balance at month $t$ (Page 4 amortisation schedule)
    
    ### Staging logic

    | Stage | Definition in this model | Horizon |
    |---|---|---|
    | Stage 1 | ECL summed over months 1–12 | 12 months |
    | Stage 2 | ECL summed over all remaining months (month 13 to loan maturity) | Lifetime |
    
    ### Stress test interpretation
    
    The stress ratio is computed as:
    
    `stress_ratio = PD_multiplier × (LGD_stress / LGD_base)`
    
    Applied uniformly because PD, LGD, and EAD enter ECL multiplicatively — the 
    stress impact scales linearly with each factor. This is a simplifying 
    assumption; in production, banks may apply differentiated stress factors 
    across segments (e.g., higher PD stress on higher-risk grades).
    
    ### Key assumptions
    
    - **No prepayment**: Loans are assumed to follow scheduled amortisation
    - **No discounting**: ECL is presented in undiscounted form for clarity
    - **Static portfolio**: No new originations or runoff assumed
    - **Homogeneous LGD**: Single constant across all segments (validated in Page 5)
    
    Production IFRS 9 models would typically add:
    - Discounting at the loan's effective interest rate
    - Prepayment behavioural model
    - Forward-looking macroeconomic scenarios with probability weighting
    - Three-scenario weighted average (base, upside, downside)
    """)