"""
Page 4: LGD Estimation
File: ifrs9_app/pages/4_LGD_Estimation.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LGD Estimation",
    layout="wide"
)

st.title("LGD Estimation")
st.markdown("**Loss Given Default for the IFRS 9 ECL calculation**")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_lgd_data():
    # ─── Load summary ───────────────────────────────────────
    summary_df = pd.read_csv('data/lgd_summary.csv')
    model1 = summary_df[summary_df['model'].str.contains('2-Stage')].iloc[0]
    model2 = summary_df[summary_df['model'].str.contains('Direct Beta')].iloc[0]
    
    summary = {
        'portfolio_lgd': 0.8555,
        'twostage_mae': model1['mae'],
        'twostage_rmse': model1['rmse'],
        'twostage_bias': model1['bias'],
        'twostage_correlation': model1['correlation'],
        'twostage_r2': model1['r2'],
        'beta_mae': model2['mae'],
        'beta_rmse': model2['rmse'],
        'beta_bias': model2['bias'],
        'beta_correlation': model2['correlation'],
        'beta_r2': model2['r2'],
        'n_defaults_train': int(model1['n_train']),
        'n_defaults_test': int(model1['n_test']),
    }
    
    # ─── Load head-to-head with benchmark ───────────────────
    h2h_raw = pd.read_csv('data/lgd_head_to_head.csv')
    benchmark_mae = h2h_raw[h2h_raw['model'].str.contains('Benchmark')]['mae'].iloc[0]
    
    head_to_head_rows = []
    for _, row in h2h_raw.iterrows():
        model_name = row['model']
        mae = row['mae']
        
        if 'Benchmark' in model_name:
            improvement = 'baseline'
            decision = '✅ Yes — production approach'
            display_name = 'Portfolio constant (0.8555)'
        else:
            pct = (1 - mae / benchmark_mae) * 100
            improvement = f"{pct:+.1f}%"
            if '2-Stage' in model_name:
                decision = '❌ Worse than constant'
                display_name = '2-stage model (logistic + beta)'
            else:
                decision = '❌ Failed'
                display_name = 'Direct beta regression'
        
        head_to_head_rows.append({
            'Approach': display_name,
            'MAE': mae,
            'Bias': row['bias'],
            'Correlation': row['correlation'],
            'Improvement vs baseline': improvement,
            'Selected?': decision
        })
    
    head_to_head = pd.DataFrame(head_to_head_rows)
    
    # Calculate consistent values
    summary['benchmark_mae'] = float(benchmark_mae)
    summary['twostage_improvement_pct'] = round(
        (1 - model1['mae'] / benchmark_mae) * 100, 1
    )
    summary['beta_improvement_pct'] = round(
        (1 - model2['mae'] / benchmark_mae) * 100, 1
    )
    
    # ─── Load segments ──────────────────────────────────────
    segments_raw = pd.read_csv('data/lgd_segments.csv')
    
    segment_type_map = {
        'grade': 'Grade',
        'term': 'Term',
        'purpose': 'Purpose',
        'home_ownership': 'Home Ownership'
    }
    
    segments = pd.DataFrame({
        'segment_type': segments_raw['segment_type'].map(segment_type_map),
        'segment': segments_raw['bin'],
        'actual_lgd': segments_raw['actual_lgd_mean'],
        'n': segments_raw['n']
    })
    
    return summary, segments, head_to_head

summary, segments, head_to_head = load_lgd_data()

# ============================================================
# SECTION 1: THE VERDICT
# ============================================================
st.markdown("---")
st.header("1. Results Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Final LGD",
    f"{summary['portfolio_lgd']:.4f}",
    delta="Portfolio constant",
    delta_color="off"
)

col2.metric(
    "Constant MAE (baseline)",
    f"{summary['benchmark_mae']:.4f}",
    delta="Best approach",
    delta_color="off"
)

col3.metric(
    "2-stage model MAE",
    f"{summary['twostage_mae']:.4f}",
    delta=f"{summary['twostage_improvement_pct']:.1f}% vs baseline",
    delta_color="off"
)

st.success(f"""
**LGD = {summary['portfolio_lgd']:.4f} applied to every loan in the portfolio.**

After testing two modelling approaches, the portfolio constant **outperforms** 
both individual-level models on out-of-time data. The 2-stage model is 
{abs(summary['twostage_improvement_pct']):.1f}% worse, and direct beta regression 
is {abs(summary['beta_improvement_pct']):.1f}% worse. Modelling LGD adds noise, 
not signal — the structural truth is that unsecured retail recovery is 
essentially constant across all observable borrower characteristics.
""")

# ============================================================
# SECTION 2: WHAT DROVE THE DECISION
# ============================================================
st.markdown("---")
st.header("2. What Drove the Decision")

st.markdown("Three findings, in order of importance:")

# Finding 1: Recovery is constant across segments
col_finding, col_evidence = st.columns([1, 2])

with col_finding:
    st.markdown("##### 1. Recovery is essentially constant")
    st.metric(
        "LGD range across all segments", 
        f"{segments['actual_lgd'].min():.4f} – {segments['actual_lgd'].max():.4f}",
        delta=f"{(segments['actual_lgd'].max() - segments['actual_lgd'].min())*100:.2f} pp spread",
        delta_color="off"
    )

with col_evidence:
    st.markdown(f"""
    Across **grade, term, purpose, and home ownership** — every observable 
    segmentation produces LGD within a {(segments['actual_lgd'].max() - segments['actual_lgd'].min())*100:.2f} 
    percentage point band. There is essentially nothing for a model to discriminate.
    """)

st.markdown("---")

# Finding 2: Models perform WORSE than the constant
col_finding, col_evidence = st.columns([1, 2])

with col_finding:
    st.markdown("##### 2. Models perform WORSE than the constant")
    st.metric(
        "2-stage MAE vs baseline", 
        f"{summary['twostage_improvement_pct']:.1f}%",
        delta=f"{summary['twostage_mae']:.4f} vs {summary['benchmark_mae']:.4f}",
        delta_color="off"
    )

with col_evidence:
    st.markdown(f"""
    The 2-stage model produces **higher MAE** ({summary['twostage_mae']:.4f}) than 
    the portfolio constant ({summary['benchmark_mae']:.4f}). Correlation of 
    {summary['twostage_correlation']:.4f} confirms the model has **no ranking ability**. 
    Individual-level modelling actively degrades accuracy for unsecured retail.
    """)

st.markdown("---")

# Finding 3: Industry practice
col_finding, col_evidence = st.columns([1, 2])

with col_finding:
    st.markdown("##### 3. Industry uses portfolio averages")
    st.metric("Source", "GARP whitepaper")
    st.caption("Modeling LGD for CCAR, IFRS 9 and CECL")

with col_evidence:
    st.markdown("""
    > "The current state of the industry for retail portfolio particularly 
    > unsecured is to use simplistic long run average for LGD estimation."
    
    Major banks use the same approach for unsecured retail. This is 
    GARP-supported and accepted under IFRS 9 governance.
    """)

# ============================================================
# SECTION 3: MODELS TESTED
# ============================================================
st.markdown("---")
st.header("3. Models Tested")

st.markdown("Three approaches compared on the same test set:")

# Color-code the selected row
def highlight_selected(row):
    selected_val = str(row['Selected?'])
    if 'Yes' in selected_val:
        return ['background-color: #d4edda'] * len(row)
    elif 'Failed' in selected_val:
        return ['background-color: #f8d7da'] * len(row)
    elif 'Worse' in selected_val:
        return ['background-color: #fff3cd'] * len(row)
    return [''] * len(row)

styled_h2h = head_to_head.style.apply(highlight_selected, axis=1).format({
    'MAE': '{:.4f}',
    'Bias': '{:+.4f}',
    'Correlation': '{:+.4f}'
})

st.dataframe(
    styled_h2h,
    use_container_width=True,
    hide_index=True
)

st.info(f"""
Both individual-level models perform worse than simply using the portfolio 
constant of {summary['portfolio_lgd']:.4f}. The 2-stage model is 
{abs(summary['twostage_improvement_pct']):.1f}% worse on MAE; direct beta 
regression is {abs(summary['beta_improvement_pct']):.1f}% worse. The added 
complexity of maintaining two regression models in production cannot be 
justified when the simpler approach is also the more accurate one.
""")

# ============================================================
# SECTION 4: LGD BY SEGMENT
# ============================================================
st.markdown("---")
st.header("4. LGD by Segment — Visual Proof of Uniformity")

st.markdown("""
Every observable segment produces an LGD between 0.84 and 0.86. The model 
has nothing meaningful to discriminate.
""")

# Horizontal dot plot
fig = go.Figure()

colors_map = {
    'Grade': '#0f3460',
    'Term': '#e94560',
    'Purpose': '#16a085',
    'Home Ownership': '#f39c12'
}

for seg_type in segments['segment_type'].unique():
    df_filter = segments[segments['segment_type'] == seg_type]
    fig.add_trace(go.Scatter(
        x=df_filter['actual_lgd'],
        y=df_filter['segment'],
        mode='markers',
        name=seg_type,
        marker=dict(
            size=14,
            color=colors_map.get(seg_type, '#888'),
            line=dict(color='white', width=2)
        ),
        text=[f"n={n:,}" for n in df_filter['n']],
        hovertemplate='%{y}<br>LGD: %{x:.4f}<br>%{text}<extra></extra>'
    ))

fig.add_vline(
    x=summary['portfolio_lgd'],
    line_dash="dash",
    line_color="grey",
    annotation_text=f"Portfolio LGD = {summary['portfolio_lgd']:.4f}",
    annotation_position="top"
)

fig.update_layout(
    xaxis=dict(title='Actual LGD', range=[0.83, 0.87]),
    yaxis=dict(title='', autorange='reversed'),
    height=500,
    margin=dict(l=200, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.caption(f"""
Every dot represents the actual mean LGD for that segment. The clustering 
between {segments['actual_lgd'].min():.4f} and {segments['actual_lgd'].max():.4f} 
confirms LGD is essentially uniform across all observable borrower characteristics.
""")

# Segment statistics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Min LGD", f"{segments['actual_lgd'].min():.4f}")
col2.metric("Max LGD", f"{segments['actual_lgd'].max():.4f}")
col3.metric("Spread", f"{(segments['actual_lgd'].max() - segments['actual_lgd'].min())*100:.2f} pp")
col4.metric("Mean LGD", f"{segments['actual_lgd'].mean():.4f}")

# ============================================================
# SECTION 5: ECL IMPLICATION
# ============================================================
st.markdown("---")
st.header("5. ECL Implication")

st.markdown("How the LGD decision flows into the ECL calculation:")

st.latex(rf"ECL = PD \times {summary['portfolio_lgd']:.4f} \times EAD")

total_modelled = summary['n_defaults_train'] + summary['n_defaults_test']

st.info(f"""
**Every performing loan in the portfolio uses LGD = {summary['portfolio_lgd']:.4f} 
in the ECL calculation**, regardless of grade, term, or purpose.

The ECL Calculator page applies this constant to all {total_modelled:,} 
loans modelled. The total provision is driven by:

- **PD** (from the scorecard, Page 2) — varies by borrower
- **LGD** = {summary['portfolio_lgd']:.4f} — constant across the portfolio
- **EAD** = outstanding balance — varies by loan and time
""")

# ============================================================
# SECTION 6: METHODOLOGY (EXPANDER)
# ============================================================
st.markdown("---")

with st.expander("📘 For technical readers: model methodology and detailed results"):
    
    st.markdown("### Recovery rate distribution")
    st.markdown("""
    The recovery rate distribution has two distinct components:
    
    - **Spike at zero**: total losses (no recovery)
    - **Continuous distribution between 0 and 1**: partial recoveries
    
    This bimodal structure motivates the **two-stage modelling approach**.
    """)
    
    st.markdown("### Two-stage approach")
    
    st.markdown("**Stage 1 — Logistic regression:** Predicts whether any recovery will occur.")
    st.latex(r"P(recovery > 0 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta'X)}}")
    
    st.markdown("**Stage 2 — Beta regression:** Conditional on recovery > 0, predicts the recovery amount.")
    st.latex(r"E[recovery \mid recovery > 0, X] \sim Beta(\mu(X), \phi)")
    
    st.markdown("**Combined prediction:**")
    st.latex(r"E[recovery] = P(recovery > 0) \times E[recovery \mid recovery > 0]")
    st.latex(r"LGD = 1 - E[recovery]")
    
    st.markdown("### Why direct beta regression fails")
    st.markdown(f"""
    Beta regression with Smithson-Verkuilen transformation tries to fit one smooth 
    curve to the bimodal recovery distribution. It cannot capture the spike at zero 
    properly, leading to:
    
    - Systematic underestimation of recovery (bias {summary['beta_bias']:+.4f})
    - MAE three times worse than the 2-stage approach ({summary['beta_mae']:.4f} vs {summary['twostage_mae']:.4f})
    - R² of {summary['beta_r2']:.4f} on test
    """)
    
    st.markdown("### Performance metrics — full comparison")
    
    detailed_metrics = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'Correlation', 'Bias'],
        '2-Stage Model': [
            f"{summary['twostage_mae']:.4f}",
            f"{summary['twostage_rmse']:.4f}",
            f"{summary['twostage_r2']:.4f}",
            f"{summary['twostage_correlation']:.4f}",
            f"{summary['twostage_bias']:+.4f}"
        ],
        'Direct Beta': [
            f"{summary['beta_mae']:.4f}",
            f"{summary['beta_rmse']:.4f}",
            f"{summary['beta_r2']:.4f}",
            f"{summary['beta_correlation']:.4f}",
            f"{summary['beta_bias']:+.4f}"
        ],
        'Interpretation': [
            'Mean absolute error of LGD predictions',
            'Root mean squared error',
            'Negative = worse than mean baseline',
            'Linear correlation with actual LGD',
            'Average prediction error'
        ]
    })
    
    st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("### Why correlation is the most damning metric")
    
    st.markdown(f"""
    The 2-stage model achieves correlation of **{summary['twostage_correlation']:.4f}** 
    with actual LGD on test data — essentially zero.
    
    This means the model **cannot rank borrowers** by recovery likelihood. The MAE 
    improvement comes entirely from better handling of the zero-recovery spike 
    (Stage 1 correctly identifying total losses), not from genuine discrimination 
    on partial recoveries.
    
    A model that predicts the same value for everyone would achieve identical 
    correlation. The 2-stage model is essentially a sophisticated way to predict 
    the portfolio mean.
    """)
    
    st.markdown("### Sample data")
    st.markdown(f"""
    - **Training defaults**: {summary['n_defaults_train']:,}
    - **Test defaults**: {summary['n_defaults_test']:,}
    - **Recovery measurement**: post-default cash flows / outstanding balance at default
    - **Time window**: matches the temporal train/test split of the PD model
    """)

# ============================================================
# SECTION 7: WHEN INDIVIDUAL-LEVEL LGD WORKS
# ============================================================
st.markdown("---")

with st.expander("🔍 When would individual-level LGD modelling work?"):
    st.markdown("""
    The decision to use a portfolio constant is **specific to unsecured retail 
    lending**. Other portfolios benefit from individual-level LGD models:
    """)
    
    portfolio_comparison = pd.DataFrame({
        'Portfolio type': [
            'Unsecured retail (this project)',
            'Secured retail (mortgages)',
            'Auto loans',
            'Wholesale / Corporate',
            'Project finance'
        ],
        'LGD approach': [
            'Portfolio constant',
            'Individual-level model',
            'Individual-level model',
            'Individual-level model',
            'Deal-specific assessment'
        ],
        'Why': [
            'Recovery is uniform; no observable predictors',
            'LTV at default predicts recovery strongly',
            'Vehicle value and depreciation observable',
            'Counterparty-specific workout outcomes',
            'Each deal has unique collateral structure'
        ]
    })
    
    st.dataframe(portfolio_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    For unsecured retail specifically:
    
    - **No collateral** → recovery depends entirely on collections
    - **Borrower information is limited post-default** → no observable predictors
    - **Recovery processes are uniform** → little segment variation
    
    These structural features make portfolio-level LGD the correct approach for 
    this product type, supported by industry practice (GARP) and accepted under 
    IFRS 9 governance.
    """)