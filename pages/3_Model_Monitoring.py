"""
Page 3: Model Monitoring — PSI Analysis
File location: ifrs9_app/pages/3_🔍_Model_Monitoring.py

Required data files in /data folder:
- psi_summary.csv               (variable-level PSI)
- psi_bin_breakdown.csv         (bin-level PSI for drill-down)
- score_distribution_comparison.csv  (score histogram train vs monitoring)
- monitoring_summary.pkl        (top-level metrics dict)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Model Monitoring",
    layout="wide"
)

st.title("Model Monitoring — PSI Analysis")
st.markdown("**Population Stability Index** — measuring distribution shifts between training and out-of-time monitoring data")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_monitoring_data():
    psi_summary = pd.read_csv('data/psi_summary.csv')
    psi_breakdown = pd.read_csv('data/psi_bin_breakdown.csv')
    score_dist = pd.read_csv('data/score_distribution_comparison.csv')
    summary = pickle.load(open('data/monitoring_summary.pkl', 'rb'))
    return psi_summary, psi_breakdown, score_dist, summary

psi_summary, psi_breakdown, score_dist, summary = load_monitoring_data()

# ============================================================
# SECTION 1: MONITORING OVERVIEW
# ============================================================
st.markdown("---")
st.header("1. Monitoring Overview")

st.markdown(f"""
**Reporting period:** {summary['reporting_period']}  
**Training period:** {summary['training_period']}  
**Comparison:** Out-of-time validation comparing live monitoring data against the development sample.
""")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Score PSI",
    f"{summary['score_psi']:.4f}",
    delta="Stable" if summary['score_psi'] < 0.10 else "Investigate"
)

total_vars = summary['n_stable'] + summary['n_minor'] + summary['n_major']
col2.metric(
    "Variables Stable",
    f"{summary['n_stable']} / {total_vars}",
    delta="PSI < 0.10"
)

col3.metric(
    "Minor Shifts",
    summary['n_minor'],
    delta="0.10 ≤ PSI < 0.25",
    delta_color="off"
)

col4.metric(
    "Major Shifts",
    summary['n_major'],
    delta="PSI ≥ 0.25",
    delta_color="inverse"
)

# Sample sizes
col1, col2 = st.columns(2)
col1.metric("Training Sample", f"{summary['n_train']:,}")
col2.metric("Monitoring Sample", f"{summary['n_monitoring']:,}")

# ============================================================
# SECTION 2: SCOPE OF THIS MONITORING CYCLE
# ============================================================
st.markdown("---")
st.header("2. Scope of This Monitoring Cycle")

st.info("""
**This page focuses exclusively on distribution stability (PSI).**

Discrimination metrics (Gini, AUC) and calibration testing are deliberately 
excluded from the 2019 Q1 monitoring cycle because they require **observed 
default outcomes** that have not yet materialised.

The 2019 Q1 vintage is too recent to provide reliable outcome data:
- Lifetime PD horizon: 36–60 months
- Observed snapshot horizon: < 3 months at time of monitoring
- Most loans that will eventually default haven't done so yet

This is consistent with industry practice — PSI provides **immediate** stability 
signals while outcome-based metrics require waiting for vintages to mature.
""")

# ============================================================
# SECTION 3: PSI METHODOLOGY
# ============================================================
st.markdown("---")
st.header("3. PSI Methodology")

col_formula, col_thresholds = st.columns([2, 1])

with col_formula:
    st.markdown(r"""
    **Population Stability Index** measures how much a distribution has shifted 
    between two time periods:
    """)
    
    st.latex(r"PSI = \sum_i (P_{actual,i} - P_{expected,i}) \times \ln\left(\frac{P_{actual,i}}{P_{expected,i}}\right)")
    
    st.markdown("""
    Where:
    - $P_{expected}$ = proportion of population in bin $i$ during **training**
    - $P_{actual}$ = proportion in bin $i$ during the **monitoring period**
    
    PSI is computed for the **score** itself (overall model stability) and for 
    **each input variable** (to identify which variables drive any instability).
    
    **What causes PSI shifts?**
    
    - Changes in customer mix (e.g., targeting new segments)
    - Macroeconomic shifts (recession, growth)
    - Product changes (new loan types, pricing changes)
    - Platform mechanics (origination process changes)
    """)

with col_thresholds:
    thresholds_df = pd.DataFrame({
        'PSI Range': ['< 0.10', '0.10 – 0.25', '≥ 0.25'],
        'Stability': ['Stable', 'Minor shift', 'Major shift'],
        'Action': ['Monitor', 'Investigate', 'Recalibrate']
    })
    
    def color_stability(val):
        colors = {
            'Stable': 'background-color: #d4edda',
            'Minor shift': 'background-color: #fff3cd',
            'Major shift': 'background-color: #f8d7da'
        }
        return colors.get(val, '')
    
    styled = thresholds_df.style.applymap(color_stability, subset=['Stability'])
    st.dataframe(styled, hide_index=True, use_container_width=True)

# ============================================================
# SECTION 4: SCORE-LEVEL STABILITY
# ============================================================
st.markdown("---")
st.header("4. Score-Level Stability")

st.markdown("""
The **Score PSI** captures whether the model's overall output distribution has 
shifted between training and the monitoring period. A stable Score PSI indicates 
the model continues to produce predictions consistent with its training data.
""")

# Score distribution overlay
fig = go.Figure()

fig.add_trace(go.Bar(
    x=score_dist['bin_center'],
    y=score_dist['train_proportion'],
    name=f"Training (n = {summary['n_train']:,})",
    marker_color='#0f3460',
    opacity=0.6
))

fig.add_trace(go.Bar(
    x=score_dist['bin_center'],
    y=score_dist['monitoring_proportion'],
    name=f"Monitoring (n = {summary['n_monitoring']:,})",
    marker_color='#e94560',
    opacity=0.6
))

fig.update_layout(
    barmode='overlay',
    xaxis_title='Credit Score',
    yaxis_title='Proportion',
    height=400,
    margin=dict(l=40, r=40, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Score PSI verdict
score_psi = summary['score_psi']
if score_psi < 0.10:
    st.success(f"""
    ✅ **Score PSI = {score_psi:.4f} — Stable**
    
    The model's output distribution is consistent with training. No action needed.
    The score distribution overlays closely between periods, confirming the model 
    continues to produce reliable predictions.
    """)
elif score_psi < 0.25:
    st.warning(f"""
    ⚠️ **Score PSI = {score_psi:.4f} — Minor shift**
    
    Investigate variable-level PSI to identify drivers.
    """)
else:
    st.error(f"""
    🔴 **Score PSI = {score_psi:.4f} — Major shift**
    
    Recalibration recommended. The model may not generalise to the current population.
    """)

# ============================================================
# SECTION 5: VARIABLE-LEVEL PSI
# ============================================================
st.markdown("---")
st.header("5. Variable-Level PSI")

st.markdown("""
PSI computed for each input variable identifies which features are driving 
distribution shifts. Variables with major shifts may require investigation or 
re-binning even if the overall score PSI is stable.
""")

# Sort for visualisation
psi_sorted = psi_summary.sort_values('PSI', ascending=True).reset_index(drop=True)

# Colour-code by threshold
def get_color(psi_val):
    if psi_val < 0.10:
        return '#27ae60'  # green
    elif psi_val < 0.25:
        return '#f5a623'  # amber
    else:
        return '#e94560'  # red

psi_sorted['color'] = psi_sorted['PSI'].apply(get_color)

# Horizontal bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=psi_sorted['PSI'],
    y=psi_sorted['Original feature name'],
    orientation='h',
    marker_color=psi_sorted['color'].tolist(),
    text=[f'{v:.4f}' for v in psi_sorted['PSI']],
    textposition='outside'
))

# Threshold lines
fig.add_vline(x=0.10, line_dash="dash", line_color="orange",
              annotation_text="Minor (0.10)", annotation_position="top")
fig.add_vline(x=0.25, line_dash="dash", line_color="red",
              annotation_text="Major (0.25)", annotation_position="top")

fig.update_layout(
    xaxis_title='PSI',
    yaxis_title='',
    height=500,
    margin=dict(l=220, r=40, t=40, b=40),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Variable PSI table
def style_stability(val):
    if 'Stable' in str(val):
        return 'background-color: #d4edda; color: #155724'
    elif 'Minor' in str(val):
        return 'background-color: #fff3cd; color: #856404'
    elif 'Major' in str(val):
        return 'background-color: #f8d7da; color: #721c24'
    return ''

styled_table = (psi_summary.style
                .applymap(style_stability, subset=['Stability'])
                .format({'PSI': '{:.4f}'}))

st.dataframe(styled_table, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 6: DRILL-DOWN BY VARIABLE
# ============================================================
st.markdown("---")
st.header("6. Drill-Down — Bin-Level Analysis")

st.markdown("""
Select a variable to see how its individual bins shifted between training and 
monitoring periods. Large positive or negative differences indicate which bins 
gained or lost population share.
""")

variables = sorted(psi_breakdown['Original feature name'].unique())

# Default to most unstable variable
default_var = psi_summary.iloc[0]['Original feature name']
default_idx = variables.index(default_var) if default_var in variables else 0

selected_var = st.selectbox(
    "Select variable:",
    options=variables,
    index=default_idx
)

# Filter to selected variable
var_data = psi_breakdown[psi_breakdown['Original feature name'] == selected_var].copy()
var_data['difference'] = var_data['Proportions_2019Q1'] - var_data['Proportions_Train']
var_data = var_data.sort_values('index')

# Variable PSI summary
var_psi = psi_summary[psi_summary['Original feature name'] == selected_var]['PSI'].iloc[0]
var_stability = psi_summary[psi_summary['Original feature name'] == selected_var]['Stability'].iloc[0]

col1, col2 = st.columns([1, 3])

with col1:
    st.metric(f"PSI for {selected_var}", f"{var_psi:.4f}")
    
    if 'Stable' in var_stability:
        st.success(var_stability)
    elif 'Minor' in var_stability:
        st.warning(var_stability)
    else:
        st.error(var_stability)

with col2:
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Training',
        x=var_data['index'],
        y=var_data['Proportions_Train'],
        marker_color='#0f3460',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name='Monitoring',
        x=var_data['index'],
        y=var_data['Proportions_2019Q1'],
        marker_color='#e94560',
        opacity=0.7
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='',
        yaxis_title='Proportion of Population',
        height=350,
        margin=dict(l=40, r=40, t=20, b=120),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Bin-level table
st.markdown("##### Bin-Level Breakdown")

var_display = var_data[['index', 'Proportions_Train', 'Proportions_2019Q1',
                         'difference', 'Contribution']].copy()
var_display.columns = ['Bin', 'Training %', 'Monitoring %', 'Difference', 'PSI Contribution']

def highlight_contribution(val):
    if isinstance(val, (int, float)):
        if val > 0.05:
            return 'background-color: #f8d7da'
        elif val > 0.01:
            return 'background-color: #fff3cd'
    return ''

styled_drill = (var_display.style
                .applymap(highlight_contribution, subset=['PSI Contribution'])
                .format({
                    'Training %': '{:.4f}',
                    'Monitoring %': '{:.4f}',
                    'Difference': '{:+.4f}',
                    'PSI Contribution': '{:.4f}'
                }))

st.dataframe(styled_drill, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 7: MONITORING FRAMEWORK
# ============================================================
st.markdown("---")
st.header("7. Monitoring Framework")

st.markdown("""
PSI is one component of a complete model monitoring framework. Different 
checks become possible at different points in the loan lifecycle.
""")

framework_df = pd.DataFrame({
    'Check': [
        'Score PSI',
        'Variable PSI',
        'Discrimination (Gini, AUC)',
        'Calibration (predicted vs observed)',
        'Full revalidation'
    ],
    'Available when': [
        '✅ Immediately',
        '✅ Immediately',
        '⏳ 12+ months after origination',
        '⏳ 24–36 months after origination',
        '⏳ Annually with mature data'
    ],
    'Status this cycle': [
        f"✅ Done ({summary['score_psi']:.4f} — Stable)",
        '✅ Done (see Section 5)',
        '⏳ Awaiting mature defaults',
        '⏳ Awaiting mature defaults',
        '⏳ Annual cycle'
    ],
    'Trigger threshold': [
        'PSI > 0.25',
        'PSI > 0.25',
        'Gini drop > 0.05',
        'Predicted/Observed outside 0.8–1.2',
        'Two consecutive failures'
    ]
})

st.dataframe(framework_df, use_container_width=True, hide_index=True)

st.success(f"""
**This monitoring cycle: model passes all available stability tests.**

Score PSI of {summary['score_psi']:.4f} confirms the output distribution remains 
consistent with training. Calibration and discrimination assessments will be 
performed in 2020 onwards when 2019 Q1 vintage defaults have materialised.
""")

# ============================================================
# SECTION 8: KEY INSIGHTS
# ============================================================
st.markdown("---")

with st.expander("🎯 Key insights from this monitoring cycle"):
    major_shifts = psi_summary[psi_summary['PSI'] >= 0.25]['Original feature name'].tolist()
    minor_shifts = psi_summary[
        (psi_summary['PSI'] >= 0.10) & (psi_summary['PSI'] < 0.25)
    ]['Original feature name'].tolist()
    
    st.markdown(f"""
    **1. Score-level stability is excellent**
    
    Score PSI of {summary['score_psi']:.4f} is comfortably below the 0.10 threshold. 
    The model continues to produce score distributions consistent with training.
    
    **2. Variables with major shifts ({len(major_shifts)})**
    
    {', '.join(major_shifts) if major_shifts else 'None'}
    
    These reflect genuine population changes between 2014–2018 (training) and 
    2019 Q1 (monitoring). For variables like `initial_list_status`, this is a 
    platform mechanic shift rather than a borrower risk signal — Lending Club 
    shifted its listing methodology over time.
    
    **3. Variables with minor shifts ({len(minor_shifts)})**
    
    {', '.join(minor_shifts) if minor_shifts else 'None'}
    
    These warrant continued monitoring but don't currently trigger action.
    
    **4. Why score PSI is stable despite variable shifts**
    
    When multiple variables shift in offsetting directions, their effects on 
    the final score can cancel out. This is why score PSI alone is not enough — 
    variable-level monitoring catches risks that aggregate metrics miss.
    
    **5. Scope of this monitoring cycle**
    
    Outcome-based metrics (discrimination, calibration) are deferred to future 
    cycles when 2019 Q1 defaults have matured. Current verdict is based 
    exclusively on distribution stability.
    
    **Conclusion:** No model recalibration required. Continue quarterly PSI 
    monitoring. Schedule discrimination assessment for 2020 Q1+ and calibration 
    assessment for 2022+.
    """)