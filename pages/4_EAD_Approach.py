"""
Page 4: EAD Approach
File: ifrs9_app/pages/4_EAD_Approach.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EAD Approach",
    layout="wide"
)

st.title("EAD Approach")
st.markdown("**Exposure at Default for the IFRS 9 ECL calculation**")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_ead_data():
    summary = pd.read_csv('data/ead_summary.csv')
    recalibration = pd.read_csv('data/ead_recalibration_results.csv')
    coefficients = pd.read_csv('data/ead_ols_coefficients.csv')
    
    # Extract train/test for easy access
    train_metrics = summary[summary['split'] == 'train'].iloc[0]
    test_metrics = summary[summary['split'] == 'test'].iloc[0]
    
    metrics = {
        'train_mae': float(train_metrics['mae']),
        'train_rmse': float(train_metrics['rmse']),
        'train_r2': float(train_metrics['r2']),
        'train_correlation': float(train_metrics['correlation']),
        'train_bias': float(train_metrics['bias']),
        'train_mean_actual': float(train_metrics['mean_actual']),
        'train_mean_pred': float(train_metrics['mean_pred']),
        'n_train': int(train_metrics['n']),
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
        'test_correlation': float(test_metrics['correlation']),
        'test_bias': float(test_metrics['bias']),
        'test_mean_actual': float(test_metrics['mean_actual']),
        'test_mean_pred': float(test_metrics['mean_pred']),
        'n_test': int(test_metrics['n']),
    }
    
    return metrics, recalibration, coefficients

metrics, recalibration, coefficients = load_ead_data()

# ============================================================
# SECTION 1: THE VERDICT
# ============================================================
st.markdown("---")
st.header("1. Key findings")

col1, col2, col3 = st.columns(3)

col1.metric(
    "EAD approach",
    "Amortisation schedule",
    delta="No statistical model",
    delta_color="off"
)

col2.metric(
    "OLS Test R²",
    f"{metrics['test_r2']:.2f}",
    delta="Severe calibration failure",
    delta_color="off"
)

col3.metric(
    "Decision basis",
    "Statistical + Regulatory",
    delta="CRR3 Article 182",
    delta_color="off"
)

st.success(f"""
**For term loans, EAD equals the outstanding balance projected via the amortisation schedule.**

An OLS regression was tested as the alternative — it achieved correlation of 
{metrics['test_correlation']:.3f} on test but suffered severe calibration failure 
(R² of {metrics['test_r2']:.2f}, bias of {metrics['test_bias']:+.3f}). The amortisation 
approach is the methodologically correct choice for term loans, supported by 
CRR3 guidance that restricts EAD modelling to revolving exposures.
""")

# ============================================================
# SECTION 2: WHAT I TESTED AND WHAT HAPPENED
# ============================================================
st.markdown("---")
st.header("2. What I Tested and What Happened")

st.markdown(f"""
**The model:** OLS regression predicting EAD ratio (outstanding balance ÷ original 
loan amount) at default, using {len(coefficients)-1} borrower and loan characteristics.
""")

st.markdown("**The results:**")

results_table = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'Correlation', 'R²', 'Bias'],
    'Train': [
        f"{metrics['train_mae']:.4f}",
        f"{metrics['train_rmse']:.4f}",
        f"{metrics['train_correlation']:.4f}",
        f"{metrics['train_r2']:.4f}",
        f"{metrics['train_bias']:+.4f}"
    ],
    'Test': [
        f"{metrics['test_mae']:.4f}",
        f"{metrics['test_rmse']:.4f}",
        f"{metrics['test_correlation']:.4f}",
        f"{metrics['test_r2']:.4f}",
        f"{metrics['test_bias']:+.4f}"
    ],
    'Verdict': [
        'Similar',
        'Similar',
        'Stable ranking',
        'Severe degradation',
        'Major level shift'
    ]
})

st.dataframe(results_table, use_container_width=True, hide_index=True)

# Two-column diagnosis
col_diagnosis, col_visual = st.columns([1, 1])

with col_diagnosis:
    st.markdown(f"""
    ##### The diagnosis
    
    Correlation actually **improved** on test ({metrics['test_correlation']:.3f} 
    vs {metrics['train_correlation']:.3f} train) — meaning the model still ranks 
    borrowers correctly.
    
    But the bias of **{metrics['test_bias']:+.3f}** reveals a systematic level shift. 
    The test population has a mean EAD ratio of **{metrics['test_mean_actual']:.4f}** 
    versus training mean of **{metrics['train_mean_actual']:.4f}** — a 
    {(metrics['test_mean_actual'] - metrics['train_mean_actual'])*100:.1f} percentage 
    point gap.
    
    Same coefficients, completely different population.
    """)

with col_visual:
    # Visualisation: train vs test mean comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Train', 'Test'],
        y=[metrics['train_mean_actual'], metrics['test_mean_actual']],
        name='Actual',
        marker_color='#0f3460',
        text=[f"{metrics['train_mean_actual']:.4f}", f"{metrics['test_mean_actual']:.4f}"],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=['Train', 'Test'],
        y=[metrics['train_mean_pred'], metrics['test_mean_pred']],
        name='Predicted',
        marker_color='#e94560',
        text=[f"{metrics['train_mean_pred']:.4f}", f"{metrics['test_mean_pred']:.4f}"],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Mean EAD Ratio — Actual vs Predicted',
        yaxis_title='EAD Ratio',
        yaxis=dict(range=[0, 1]),
        barmode='group',
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 3: RECALIBRATION ATTEMPT
# ============================================================
st.markdown("---")
st.header("3. Recalibration Couldn't Fix It")

st.markdown("""
Before abandoning the model, recalibration was tested. A constant intercept 
shift was applied to align mean predictions with actual values.
""")

# Build a clean recalibration comparison
recal_display = pd.DataFrame({
    'Model': ['Original', 'Recalibrated'],
    'Test R²': [
        f"{recalibration[(recalibration['model']=='Original') & (recalibration['split']=='test')]['r2'].iloc[0]:.4f}",
        f"{recalibration[(recalibration['model']=='Recalibrated') & (recalibration['split']=='test')]['r2'].iloc[0]:.4f}"
    ],
    'Test Bias': [
        f"{recalibration[(recalibration['model']=='Original') & (recalibration['split']=='test')]['bias'].iloc[0]:+.4f}",
        f"{recalibration[(recalibration['model']=='Recalibrated') & (recalibration['split']=='test')]['bias'].iloc[0]:+.4f}"
    ],
    'Test MAE': [
        f"{recalibration[(recalibration['model']=='Original') & (recalibration['split']=='test')]['mae'].iloc[0]:.4f}",
        f"{recalibration[(recalibration['model']=='Recalibrated') & (recalibration['split']=='test')]['mae'].iloc[0]:.4f}"
    ],
    'Verdict': [
        'Severe failure',
        'Partial improvement'
    ]
})

st.dataframe(recal_display, use_container_width=True, hide_index=True)

# Extract values for the diagnosis text
recal_test_r2 = recalibration[(recalibration['model']=='Recalibrated') & (recalibration['split']=='test')]['r2'].iloc[0]
recal_test_bias = recalibration[(recalibration['model']=='Recalibrated') & (recalibration['split']=='test')]['bias'].iloc[0]
orig_test_r2 = recalibration[(recalibration['model']=='Original') & (recalibration['split']=='test')]['r2'].iloc[0]

st.info(f"""
**Diagnosis:** Constant offset recalibration improved R² from {orig_test_r2:.2f} 
to {recal_test_r2:.2f}, but the bias only narrowed from {metrics['test_bias']:+.3f} 
to {recal_test_bias:+.3f}. The test population isn't just shifted by a constant — 
it has fundamentally different EAD characteristics driven by vintage effects 
(newer-vintage defaulters had less time to amortise before defaulting, producing 
systematically higher EAD ratios).
""")

# ============================================================
# SECTION 4: THE STRUCTURAL TRUTH
# ============================================================
st.markdown("---")
st.header("4. Why the Failure Confirms the Right Approach")

st.markdown("""
The OLS failure isn't a methodology problem — it's confirmation that EAD for 
term loans should not be modelled statistically.
""")

reasons_df = pd.DataFrame({
    'Reason': [
        '1. Amortisation is deterministic',
        '2. Vintage drift breaks calibration',
        '3. Regulators agree'
    ],
    'Why': [
        'Outstanding balance at any future month is mathematically calculable from loan terms',
        'Even a working model needs constant recalibration as vintage mix changes — the OLS test confirmed this empirically',
        'CRR3 Article 182 restricts CCF/EAD estimation to revolving exposures, not term loans'
    ]
})

st.dataframe(reasons_df, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 5: AMORTISATION CALCULATOR
# ============================================================
st.markdown("---")
st.header("5. The Amortisation Approach (Interactive)")

st.markdown("""
For term loans, EAD at any future month is determined by the amortisation 
schedule. Adjust the inputs below to see how outstanding balance evolves over time.
""")

# Inputs
col_input, col_chart = st.columns([1, 2])

with col_input:
    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000, max_value=40000, value=15000, step=1000
    )
    annual_rate = st.slider(
        "Interest Rate (%)",
        min_value=5.0, max_value=30.0, value=12.0, step=0.5
    )
    term_months = st.radio(
        "Term",
        options=[36, 60],
        horizontal=True
    )

with col_chart:
    # Calculate amortisation schedule
    monthly_rate = (annual_rate / 100) / 12
    payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
    
    schedule_data = []
    balance = loan_amount
    for month in range(1, term_months + 1):
        interest = balance * monthly_rate
        principal = payment - interest
        balance = max(balance - principal, 0)
        schedule_data.append({
            'Month': month,
            'Balance': balance,
            'Principal': principal,
            'Interest': interest
        })
    
    schedule = pd.DataFrame(schedule_data)
    
    # Outstanding balance chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=schedule['Month'],
        y=schedule['Balance'],
        fill='tozeroy',
        name='Outstanding Balance (EAD)',
        line=dict(color='#0f3460', width=2.5),
        fillcolor='rgba(15, 52, 96, 0.15)'
    ))
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Outstanding Balance ($)',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Summary metrics
st.markdown("##### Key amortisation values")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Monthly payment", f"${payment:,.2f}")
col2.metric("Total interest", f"${schedule['Interest'].sum():,.2f}")
col3.metric("EAD at month 12", f"${schedule.loc[11, 'Balance']:,.0f}")

if term_months >= 24:
    col4.metric("EAD at month 24", f"${schedule.loc[23, 'Balance']:,.0f}")
else:
    col4.metric("EAD at maturity", f"${schedule.iloc[-1]['Balance']:,.0f}")

if term_months >= 36:
    col5.metric("EAD at month 36", f"${schedule.loc[35, 'Balance']:,.0f}")
else:
    col5.metric("Term", f"{term_months} months")

# Formula display
with st.expander("📐 The amortisation formula"):
    st.latex(r"EAD_t = P \times \frac{(1+r)^N - (1+r)^t}{(1+r)^N - 1}")
    st.markdown(r"""
    Where:
    - $P$ = original loan amount
    - $r$ = monthly interest rate (annual rate ÷ 12)
    - $N$ = total months in the loan term
    - $t$ = months elapsed since origination
    
    This formula gives the **exact** outstanding balance at any future month — 
    no estimation required.
    """)

# ============================================================
# SECTION 6: TERM LOANS VS REVOLVING CREDIT
# ============================================================
st.markdown("---")
st.header("6. Term Loans vs Revolving Credit — Regulatory Context")

st.markdown("""
CCF (Credit Conversion Factor) modelling exists for products where future 
exposure is unpredictable. For fully-disbursed term loans, this concept 
doesn't apply.
""")

comparison_df = pd.DataFrame({
    'Aspect': [
        'EAD predictability',
        'Borrower drawdown risk',
        'Off-balance sheet exposure',
        'Regulatory treatment',
        'Model required?',
        'CRR3 Article 182'
    ],
    'Term Loans (Lending Club)': [
        'Deterministic',
        'Zero (fully drawn at origination)',
        'None',
        'EAD = outstanding balance',
        'No',
        'Not applicable'
    ],
    'Revolving Credit (cards, overdrafts)': [
        'Stochastic',
        'High (distressed borrowers reach for credit)',
        'Undrawn commitment',
        'EAD = Drawn + CCF × Undrawn',
        'Yes (CCF model)',
        'Applies — own estimates permitted'
    ]
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 7: ECL IMPLICATION
# ============================================================
st.markdown("---")
st.header("7. ECL Implication")

st.markdown("""
The EAD decision feeds into the ECL calculation as a deterministic, 
month-by-month projected balance:
""")

st.latex(r"ECL = \sum_{t=1}^{N} PD_t \times LGD \times EAD_t")

st.markdown("""
Where:
- $PD_t$ comes from the PD scorecard × timing curve (Page 2)
- $LGD$ = 0.8555 (Page 5)
- $EAD_t$ comes from the amortisation schedule (this page)

For **Stage 1** (12-month horizon): sum across months 1–12 with EAD decreasing 
each month per the amortisation schedule.

For **Stage 2** (lifetime): sum from month 1 to maturity, with EAD eventually 
reaching zero at the loan's final payment.
""")

# ============================================================
# SECTION 8: METHODOLOGY EXPANDER
# ============================================================
st.markdown("---")

with st.expander("📘 For technical readers: OLS coefficient table and diagnostics"):
    
    st.markdown("### OLS Coefficients")
    st.markdown(f"""
    The OLS model used {len(coefficients)-1} predictors (excluding intercept). 
    Below shows all coefficients with significance flags.
    """)
    
    # Format the coefficients table
    def style_significance(val):
        if val == True:
            return 'background-color: #d4edda; color: #155724'
        elif val == False:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    coef_display = coefficients.copy()
    
    styled_coef = (coef_display.style
                   .map(style_significance, subset=['significant'])
                   .format({
                       'coefficient': '{:.6f}',
                       'p_value': '{:.4f}'
                   }))
    
    st.dataframe(styled_coef, use_container_width=True, hide_index=True, height=500)
    
    # Summary stats
    n_significant = coefficients['significant'].sum()
    n_total = len(coefficients)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total coefficients", n_total)
    col2.metric("Significant (p < 0.05)", n_significant)
    col3.metric("Non-significant", n_total - n_significant)
    
    st.markdown("### Strongest predictors by coefficient magnitude")
    
    # Top 10 by absolute coefficient (excluding intercept)
    top_coefs = (coefficients[coefficients['feature'] != 'const']
                 .copy())
    top_coefs['abs_coef'] = top_coefs['coefficient'].abs()
    top_coefs = top_coefs.nlargest(10, 'abs_coef')
    
    fig = px.bar(
        top_coefs.sort_values('coefficient'),
        x='coefficient',
        y='feature',
        orientation='h',
        color='significant',
        color_discrete_map={True: '#0f3460', False: '#e94560'},
        labels={'coefficient': 'Coefficient', 'feature': '', 'significant': 'p < 0.05'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        height=400,
        margin=dict(l=180, r=40, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    ### Why the OLS model fails out-of-time
    
    The OLS achieves Train R² of {metrics['train_r2']:.4f} — modest but positive. 
    However, on test the R² drops to {metrics['test_r2']:.4f}, indicating the 
    model performs **worse than predicting the mean**.
    
    Three drivers of failure:
    
    **1. Vintage shift in EAD ratio**
    
    Test mean EAD ratio is {metrics['test_mean_actual']:.4f} vs train 
    {metrics['train_mean_actual']:.4f} — a 
    {(metrics['test_mean_actual'] - metrics['train_mean_actual'])*100:.1f} percentage 
    point gap. The model predicts at the training-mean level, producing systematic 
    underestimation on test.
    
    **2. Newer vintages default earlier**
    
    Test loans were originated more recently and defaulted sooner after origination — 
    less time to amortise, higher EAD ratio. The model's coefficients were trained 
    on older vintages with longer amortisation periods.
    
    **3. The structural truth**
    
    Even if the model could perfectly capture vintage effects, it would still be 
    approximating what the amortisation formula gives exactly. Statistical models 
    can never improve on deterministic relationships.
    """)

# ============================================================
# SECTION 9: WHEN EAD MODELLING IS REQUIRED
# ============================================================
st.markdown("---")

with st.expander("🔍 When would EAD modelling be required?"):
    st.markdown("""
    The amortisation approach is **specific to fully-disbursed term loans**. 
    Other product types require EAD modelling because future exposure is uncertain.
    """)
    
    product_comparison = pd.DataFrame({
        'Portfolio type': [
            'Term loans (this project)',
            'Mortgages (fixed)',
            'Mortgages with offset accounts',
            'Credit cards',
            'Overdrafts',
            'Committed revolving lines',
            'Corporate term loans with covenants'
        ],
        'EAD approach': [
            'Amortisation schedule',
            'Amortisation schedule',
            'Adjusted amortisation + drawdown model',
            'CCF model',
            'CCF model',
            'CCF model',
            'Stochastic / behavioural model'
        ],
        'Why': [
            'Contractually deterministic',
            'Contractually deterministic',
            'Borrower can pause/resume drawdowns',
            'High drawdown risk before default',
            'Variable utilisation',
            'Undrawn portion at risk',
            'Complex pre-default behaviour'
        ]
    })
    
    st.dataframe(product_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    The common thread across products requiring EAD modelling: **future exposure 
    is uncertain at the point of analysis**. CCF and behavioural models exist to 
    estimate this uncertainty.
    
    For Lending Club term loans, future exposure is **certain** (assuming scheduled 
    payments). The amortisation formula gives the exact answer. No model can 
    outperform an exact calculation.
    """)