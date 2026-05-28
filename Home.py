"""
Home.py — Landing page for the IFRS 9 Streamlit app
File location: ifrs9_app/Home.py
"""

import streamlit as st
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IFRS 9 ECL Engine",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.title("IFRS 9 Expected Credit Loss Engine")
st.markdown("**End-to-end credit risk modelling — from variable selection to portfolio provisions**")

st.markdown("---")

# ============================================================
# LOAD HEADLINE METRICS
# ============================================================
@st.cache_data
def load_headline_metrics():
    """Load the top-level numbers shown on the homepage."""
    summary = pd.read_csv('data/ecl_portfolio_summary.csv').iloc[0].to_dict()
    return summary

summary = load_headline_metrics()

def fmt_money(value, scale='B'):
    if scale == 'B':
        return f"${value/1e9:.2f}B"
    elif scale == 'M':
        return f"${value/1e6:,.0f}M"
    else:
        return f"${value:,.0f}"

# ============================================================
# SECTION 1: THE HEADLINE
# ============================================================
st.header("Headline Results")

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
**A fully functional IFRS 9 ECL engine** built on Lending Club's unsecured 
retail lending portfolio. The model produces {fmt_money(summary['total_ecl_stage1'], 'M')} 
in Stage 1 provisions and {fmt_money(summary['total_ecl_stage2'], 'M')} in Stage 2 
lifetime provisions across {int(summary['n_loans']):,} loans totalling 
{fmt_money(summary['total_exposure'], 'B')} in exposure.

Each component (PD, LGD, EAD, timing) was tested, justified, and documented 
following industry practice (GARP) and regulatory guidance (IFRS 9, CRR3).
""")

# ============================================================
# SECTION 2: WHAT THIS PROJECT DELIVERS
# ============================================================
st.markdown("---")
st.header("What This Project Delivers")

st.markdown("""
Three concrete outputs that demonstrate end-to-end credit risk modelling competence:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📊 A working PD scorecard")
    st.markdown("""
    - 13 variables across all 5C dimensions
    - Logistic regression with WoE binning
    - **Test Gini: 0.40** (out-of-time)
    - Train-test gap: 0.024 (no overfitting)
    - PSI Stable: 0.046
    """)

with col2:
    st.markdown("##### 💰 Defensible LGD and EAD")
    st.markdown("""
    - LGD: portfolio constant **0.8555**
    - EAD: amortisation schedule
    - Both supported by industry practice
    - CRR3-compliant approach
    - Empirically validated decisions
    """)

with col3:
    st.markdown("##### 🎯 Loan-level ECL simulator")
    st.markdown("""
    - Interactive borrower input
    - Real-time PD + LGD + EAD + timing
    - Stage 1 and Stage 2 provisions
    - Stress scenario testing
    - Reproducible production output
    """)

# ============================================================
# SECTION 3: METHODOLOGY DECISIONS AT A GLANCE
# ============================================================
st.markdown("---")
st.header("Methodology Decisions")

st.markdown("""
Three pragmatic decisions distinguish this project from textbook implementations:
""")

decisions_df = pd.DataFrame({
    'Component': [
        'PD',
        'LGD',
        'EAD',
        'Timing'
    ],
    'Approach': [
        'Logistic regression with WoE',
        'Portfolio constant (0.8555)',
        'Amortisation schedule',
        'Kaplan-Meier timing curves'
    ],
    'Reasoning': [
        'Standard for retail credit scoring; explainable; regulatory-friendly',
        'Tested 2-stage and beta models — both performed worse than the constant. LGD has no within-portfolio variation worth modelling.',
        'Tested OLS — failed out-of-time due to vintage drift. Amortisation gives the exact answer; modelling can only approximate it.',
        '10 segments (term × grade) capture distinct default timing patterns. Reuses logistic scorecard rather than building Cox PH.'
    ],
    'Industry Support': [
        'Standard practice',
        'GARP whitepaper',
        'CRR3 Article 182',
        'Most retail banks'
    ]
})

st.dataframe(decisions_df, use_container_width=True, hide_index=True)

st.info("""
**The unified narrative:** I built a sophisticated PD model where complexity adds 
value, and chose simpler approaches for LGD and EAD where complexity hurts. 
Each decision is supported by industry practice and regulatory guidance. This 
demonstrates analytical maturity — knowing **when not to model**.
""")

# ============================================================
# SECTION 4: PROJECT STRUCTURE
# ============================================================
st.markdown("---")
st.header("How to Navigate This App")

st.markdown("""
The seven pages tell a complete IFRS 9 story — from variable selection through 
to portfolio provisions. Use the sidebar to navigate, or follow this recommended order:
""")

pages_df = pd.DataFrame({
    'Page': [
        '1. Theoretical Framework',
        '2. PD Model',
        '3. Model Monitoring',
        '4. EAD Approach',
        '5. LGD Estimation',
        '6. Default Timing & ECL Simulator',
        '7. Portfolio ECL'
    ],
    'What You\'ll See': [
        'Variable selection across the 5C framework',
        'Scorecard estimation, performance, and PD simulator',
        'PSI analysis on 2019 Q1 monitoring data',
        'Why amortisation beats OLS for term loans',
        'Why a portfolio constant beats individual-level models',
        'Loan-level ECL simulator combining all components',
        'Aggregated portfolio provisions and stress scenarios'
    ],
    'Best For': [
        'Methodology reviewers',
        'Quantitative modellers',
        'Validation teams',
        'Regulatory contacts',
        'Senior credit managers',
        'Everyone — the integration showcase',
        'CFOs and risk committees'
    ]
})

st.dataframe(pages_df, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 5: TECHNICAL SUMMARY
# ============================================================
st.markdown("---")

with st.expander("📘 Technical summary"):
    st.markdown("""
    ### Dataset
    
    - Source: Lending Club public loan data (2007–2019)
    - Scope: Unsecured personal loans, 36-month and 60-month terms
    - Training sample: 1.08 million completed loans (Jan 2014 – Dec 2017)
    - Test sample: 264,000 loans (Jan 2018 – Dec 2018, temporal out-of-time)
    - Monitoring sample: 14,000 loans (2019 Q1, vintage-fresh)
    
    ### Modelling pipeline
    
    1. **Variable selection** using IV thresholds with 5C framework justification
    2. **Coarse classing** with monotonic WoE-based binning
    3. **Logistic regression** with WoE-transformed predictors
    4. **Temporal train/test split** to validate out-of-time generalisation
    5. **Standard scorecard transformation** (PDO = 50, base score = 600)
    6. **PSI monitoring** on the score distribution and individual variables
    7. **LGD model exploration** (2-stage logistic + beta, direct beta)
    8. **EAD model exploration** (OLS with recalibration)
    9. **Default timing analysis** via Kaplan-Meier survival curves
    10. **ECL aggregation** at loan and portfolio level
    
    ### Technology stack
    
    - **Python** for modelling (statsmodels, scikit-learn, lifelines)
    - **Streamlit** for the dashboard
    - **Plotly** for interactive visualisations
    - **Pandas / NumPy** for data manipulation
    
    ### Regulatory framework
    
    - **IFRS 9** — staging logic, 12-month vs lifetime ECL
    - **CRR3 Article 182** — EAD modelling restrictions
    - **GARP whitepaper** — industry LGD practice
    - **SR 11-7 principles** — model governance framework
    """)

# ============================================================
# SECTION 6: ABOUT
# ============================================================
st.markdown("---")

with st.expander("👤 About this project"):
    st.markdown("""
    **Purpose:** Demonstrate end-to-end credit risk modelling competence for 
    quantitative roles in IRB model development, IFRS 9 implementation, and 
    credit risk consulting.
    
    **What this project shows:**
    
    - Practical understanding of the IFRS 9 ECL framework
    - Ability to make and document pragmatic modelling decisions
    - Regulatory awareness (CRR3, IFRS 9, Basel)
    - Production-grade visualisation and dashboarding skill
    - End-to-end pipeline thinking from data to provisions
    
    **What this project is not:**
    
    - A claim to have built a deployment-ready bank model
    - A substitute for full IRB or IFRS 9 validation
    - Production data — Lending Club is public retail data, not a real bank portfolio
    
    **Code and documentation:** https://ifrs9-ecl-app-zxjnxemkyxcm7magyrhyvs.streamlit.app/
    
    **Contact:** An Nguyen (ngth.hoai.an@gmail.com)
    """)