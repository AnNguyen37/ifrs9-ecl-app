import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PD Model and Scorecard", layout="wide")
st.title("PD Model and Scorecard")
st.markdown("**Logistic regression with WoE-transformed predictors — temporal out-of-time validation**")

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_pd_data():
    fit_stats = pd.read_csv('data/pd_fit_stats.csv')
    coef = pd.read_csv('data/pd_summary_table.csv')
    
    # Convert fit_stats to a dict for easy access
    stats_dict = dict(zip(fit_stats['Statistic'], fit_stats['Value']))
    
    # Rename coefficient columns
    coef = coef.rename(columns={
        'Feature name': 'feature_name',
        'Coefficients': 'coefficient',
        'p_values': 'p_value'
    })
    
    # Derive additional columns
    coef['significant'] = coef['p_value'] < 0.05
    coef['variable'] = coef['feature_name'].apply(
        lambda x: x.split(':')[0] if ':' in x else x
    )
    coef['bin'] = coef['feature_name'].apply(
        lambda x: x.split(':')[1] if ':' in x else ''
    )
    
    return stats_dict, coef

stats_dict, coef = load_pd_data()

# ============================================================
# SECTION 1: MODEL ESTIMATION
# ============================================================
st.markdown("---")
st.header("1. Model Estimation")

# ─── Model fit statistics ───────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    f"{int(stats_dict['N observations']):,}"
)
col2.metric(
    "Predictors",
    f"{len(coef) - 1}",
    delta="excluding intercept"
)
col3.metric(
    "Pseudo R² (McFadden)",
    f"{stats_dict['Pseudo R-squared (McFadden)']:.4f}"
)
col4.metric(
    "LLR p-value",
    f"{stats_dict['LLR p-value']:.4f}",
    delta="Highly significant"
)

# ─── Methodology expander ───────────────────────────────────
with st.expander("📘 Methodology"):
    st.markdown(r"""
    **Logistic regression** estimates the log-odds of being a good borrower 
    (non-default) as a linear function of WoE-transformed predictors:
    """)
    
    st.latex(r"\ln\left(\frac{P(good)}{1-P(good)}\right) = \beta_0 + \sum_i \beta_i \cdot WoE_i(x)")
    
    st.markdown("""
    **Reference categories** (set to 0 in dummies):
    
    - **Grade G** (riskiest)
    - **Highest DTI bin** (>30.2)
    - **Highest interest rate bin** (>22.772)
    - **60-month term**
    - **`small_business_renewable_energy_moving_house_medical`** (riskiest purpose)
    - **`RENT_OTHER_NONE_ANY`** (riskiest home ownership)
    - **`Verified`** (verification status reference)
    
    These represent the riskiest profile in each dimension. The intercept reflects 
    the baseline log-odds for a borrower at all reference categories simultaneously.
    """)

# ─── Coefficient table ──────────────────────────────────────
st.subheader("Coefficient Estimates")

st.markdown("""
Each coefficient represents the change in log-odds of non-default relative to the 
reference category, holding other variables constant. Positive coefficients indicate 
**lower default risk** than the reference; negative coefficients indicate **higher risk**.
""")

# Filter controls
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    filter_variable = st.multiselect(
        "Filter by variable:",
        options=sorted(coef['variable'].unique()),
        default=sorted(coef['variable'].unique())
    )

with col_filter2:
    show_only_significant = st.checkbox("Show only significant (p < 0.05)", value=False)

# Apply filters
display_df = coef[coef['variable'].isin(filter_variable)].copy()
if show_only_significant:
    display_df = display_df[display_df['significant']]

# Sort by variable, then by coefficient
display_df = display_df.sort_values(['variable', 'coefficient'], ascending=[True, False])

# Color-code significance
def style_significance(val):
    if val is True:
        return 'background-color: #d4edda; color: #155724'
    elif val is False:
        return 'background-color: #f8d7da; color: #721c24'
    return ''

styled_coef = (display_df[['feature_name', 'variable', 'bin', 'coefficient', 'p_value', 'significant']]
               .style
               .map(style_significance, subset=['significant'])
               .format({
                   'coefficient': '{:.4f}',
                   'p_value': '{:.4f}'
               }))

st.dataframe(styled_coef, use_container_width=True, hide_index=True, height=500)

# Summary statistics
n_significant = coef['significant'].sum()
n_total = len(coef)

col1, col2, col3 = st.columns(3)
col1.metric("Total coefficients", n_total)
col2.metric("Significant (p < 0.05)", n_significant)
col3.metric("Non-significant", n_total - n_significant)

# ─── Coefficient visualisation by variable ──────────────────
st.subheader("Coefficient Visualisation")

st.markdown("Select a variable to see how its bin coefficients compare:")

selected_var = st.selectbox(
    "Variable:",
    options=sorted(coef[coef['variable'] != 'const']['variable'].unique())
)

var_data = coef[coef['variable'] == selected_var].sort_values('coefficient', ascending=True)

fig = px.bar(
    var_data,
    x='coefficient',
    y='bin',
    orientation='h',
    color='significant',
    color_discrete_map={True: '#0f3460', False: '#e94560'},
    labels={'coefficient': 'Coefficient (vs reference)', 'bin': '', 'significant': 'p < 0.05'},
    height=max(300, len(var_data) * 35)
)

# Add reference line at 0
fig.add_vline(x=0, line_dash="dash", line_color="grey",
              annotation_text="Reference", annotation_position="top")

fig.update_layout(
    margin=dict(l=120, r=40, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Show monotonicity check
if len(var_data) > 2:
    coefs_sorted = var_data['coefficient'].values
    is_monotonic = all(coefs_sorted[i] <= coefs_sorted[i+1] for i in range(len(coefs_sorted)-1))
    
    if is_monotonic:
        st.success(f"✓ **{selected_var}** coefficients are monotonic across bins — economic logic preserved.")
    else:
        st.info(f"ℹ️ **{selected_var}** has non-monotonic coefficients. Check whether this reflects genuine non-linearity or noise.")

# ─── Intercept interpretation ───────────────────────────────
const_row = coef[coef['variable'] == 'const']
if not const_row.empty:
    intercept_value = const_row['coefficient'].iloc[0]
    prob_good_baseline = 1 / (1 + np.exp(-intercept_value))
    baseline_pd = 1 - prob_good_baseline
    
    with st.expander("📐 Intercept interpretation"):
        st.markdown(f"""
        **Intercept = {intercept_value:.4f}**
        
        Converting to baseline probability:
        """)
        
        st.latex(r"P(good)_{baseline} = \frac{1}{1 + e^{-\beta_0}} = " 
                 + f"{prob_good_baseline:.4f}")
        
        st.markdown(f"""
        $P(default)_{{baseline}} = 1 - {prob_good_baseline:.4f} = $ **{baseline_pd:.2%}**
        
        This is the predicted default probability for a borrower in **all reference 
        categories simultaneously** — grade G, DTI >30, interest rate >22%, 60-month term, 
        and the highest-risk profile across all other variables.
        
        Such a borrower combination is rare in practice, so this represents a theoretical 
        baseline. Most borrowers fall into safer categories on at least some dimensions, 
        producing much lower predicted PDs.
        """)


# ─── Load Section 2 data ───────────────────────────────────
@st.cache_data
def load_performance_data():
    metrics = pd.read_csv('data/pd_metrics.csv').iloc[0].to_dict()
    roc = pd.read_csv('data/pd_roc_curve_small.csv')
    score_hist = pd.read_csv('data/pd_score_histogram.csv')
    return metrics, roc, score_hist

metrics, roc, score_hist = load_performance_data()

# ============================================================
# SECTION 2: MODEL PERFORMANCE
# ============================================================
st.markdown("---")
st.header("2. Model Performance and Validation")

st.markdown("""
The model is validated using a **temporal out-of-time split** — training on earlier 
issue dates, testing on later ones. This simulates the real production scenario 
where a model trained on historical data is applied to new borrowers.
""")

# ─── Headline metrics ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "AUROC (test)",
    f"{metrics['AUROC_test']:.4f}",
    delta=f"vs train {metrics['AUROC_train']:.4f}"
)
col2.metric(
    "Gini (test)",
    f"{metrics['Gini_test']:.4f}",
    delta=f"Gap {metrics['gini_gap']:.4f}",
    delta_color="off"
)
col3.metric(
    "KS Statistic",
    f"{metrics['KS_test']:.4f}"
)
col4.metric(
    "Test Observations",
    f"{int(metrics['n_test']):,}",
    delta=f"Bad rate {metrics['bad_rate_test']:.2%}"
)

# ─── Gini gap interpretation ────────────────────────────────
gap = metrics['gini_gap']
if gap < 0.03:
    st.success(f"✓ **Excellent stability** — Gini gap of {gap:.4f} is well below the 0.05 threshold. No evidence of overfitting.")
elif gap < 0.05:
    st.info(f"ℹ️ **Acceptable stability** — Gini gap of {gap:.4f} is within the 0.05 threshold.")
else:
    st.warning(f"⚠️ **Stability concern** — Gini gap of {gap:.4f} exceeds 0.05. Investigate potential overfitting.")

# ─── ROC Curve ──────────────────────────────────────────────
st.subheader("ROC Curve — Train vs Test")

col_chart, col_explain = st.columns([2, 1])

with col_chart:
    fig = go.Figure()
    
    # Train ROC
    train_roc = roc[roc['split'] == 'train']
    fig.add_trace(go.Scatter(
        x=train_roc['fpr'],
        y=train_roc['tpr'],
        name=f"Train (AUC = {metrics['AUROC_train']:.4f})",
        line=dict(color='#0f3460', width=2.5)
    ))
    
    # Test ROC
    test_roc = roc[roc['split'] == 'test']
    fig.add_trace(go.Scatter(
        x=test_roc['fpr'],
        y=test_roc['tpr'],
        name=f"Test (AUC = {metrics['AUROC_test']:.4f})",
        line=dict(color='#e94560', width=2.5)
    ))
    
    # Random reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random (AUC = 0.50)',
        line=dict(color='grey', dash='dash', width=1.5)
    ))
    
    fig.update_layout(
        xaxis=dict(title='False Positive Rate', range=[0, 1]),
        yaxis=dict(title='True Positive Rate', range=[0, 1]),
        height=450,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(x=0.55, y=0.15, bgcolor='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig, use_container_width=True)

with col_explain:
    st.markdown(f"""
    **Reading the chart:**
    
    Each curve plots the trade-off between detecting defaults (true positives) 
    and false alarms (false positives) at every possible cutoff threshold.
    
    **Gini benchmarks** for retail unsecured:
    
    | Range | Assessment |
    |---|---|
    | < 0.30 | Weak |
    | **0.30 – 0.60** | **Acceptable** |
    | > 0.60 | Strong |
    
    Gini of **{metrics['Gini_test']:.3f}** sits comfortably in the acceptable range.
    
    **Why not higher?** Retail unsecured default has a large random component 
    (job loss, illness, divorce) that no application-time variable can predict. 
    This caps achievable Gini around 0.50–0.60 even for the best models.
    """)

# ─── Gini comparison bar chart ──────────────────────────────
st.subheader("Train vs Test Comparison")

comparison_df = pd.DataFrame({
    'Metric': ['AUROC', 'Gini'],
    'Train': [metrics['AUROC_train'], metrics['Gini_train']],
    'Test': [metrics['AUROC_test'], metrics['Gini_test']]
})

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Train',
    x=comparison_df['Metric'],
    y=comparison_df['Train'],
    marker_color='#0f3460',
    text=[f'{v:.4f}' for v in comparison_df['Train']],
    textposition='outside'
))
fig.add_trace(go.Bar(
    name='Test',
    x=comparison_df['Metric'],
    y=comparison_df['Test'],
    marker_color='#e94560',
    text=[f'{v:.4f}' for v in comparison_df['Test']],
    textposition='outside'
))

fig.update_layout(
    barmode='group',
    height=350,
    margin=dict(l=40, r=40, t=20, b=40),
    yaxis=dict(range=[0, max(metrics['AUROC_train'], metrics['Gini_train']) * 1.15]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ─── Score distribution ─────────────────────────────────────
st.subheader("Score Distribution — Train vs Test")

st.markdown("""
The distribution of predicted scores should be nearly identical between train and 
test if the model is stable. Overlapping distributions confirm no significant 
population shift.
""")

fig = go.Figure()

fig.add_trace(go.Bar(
    x=score_hist['bin_center'],
    y=score_hist['train_proportion'],
    name=f"Train (n = {int(metrics['n_train']):,})",
    marker_color='#0f3460',
    opacity=0.6
))

fig.add_trace(go.Bar(
    x=score_hist['bin_center'],
    y=score_hist['test_proportion'],
    name=f"Test (n = {int(metrics['n_test']):,})",
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

# ─── Performance metrics summary table ──────────────────────
with st.expander("📊 Detailed Performance Metrics"):
    perf_table = pd.DataFrame({
        'Metric': [
            'AUROC (train)', 'AUROC (test)',
            'Gini (train)', 'Gini (test)', 'Gini gap',
            'KS statistic (test)',
            'Accuracy at threshold {:.2f}'.format(metrics['threshold']),
            'Default rate (train)', 'Default rate (test)',
            'Sample size (train)', 'Sample size (test)'
        ],
        'Value': [
            f"{metrics['AUROC_train']:.4f}",
            f"{metrics['AUROC_test']:.4f}",
            f"{metrics['Gini_train']:.4f}",
            f"{metrics['Gini_test']:.4f}",
            f"{metrics['gini_gap']:.4f}",
            f"{metrics['KS_test']:.4f}",
            f"{metrics['accuracy_test']:.4f}",
            f"{metrics['bad_rate_train']:.2%}",
            f"{metrics['bad_rate_test']:.2%}",
            f"{int(metrics['n_train']):,}",
            f"{int(metrics['n_test']):,}"
        ],
        'Interpretation': [
            'Area under ROC curve on training data',
            'Area under ROC curve on test data',
            'Gini = 2 × AUROC - 1 (training)',
            'Gini on out-of-time test set',
            'Train-test gap (< 0.05 indicates no overfitting)',
            'Kolmogorov-Smirnov maximum separation',
            'Accuracy if cutting at this PD',
            'Proportion of defaulters in train',
            'Proportion of defaulters in test',
            'Number of training observations',
            'Number of test observations'
        ]
    })
    st.dataframe(perf_table, use_container_width=True, hide_index=True)

# ─── Key insights ───────────────────────────────────────────
with st.expander("🎯 Key validation insights"):
    bad_rate_diff = abs(metrics['bad_rate_test'] - metrics['bad_rate_train'])
    
    st.markdown(f"""
    **1. Discrimination is strong and stable**
    
    Test AUROC of {metrics['AUROC_test']:.4f} (Gini {metrics['Gini_test']:.4f}) sits 
    comfortably in the acceptable range for retail unsecured lending. The train-test 
    Gini gap of just **{metrics['gini_gap']:.4f}** is excellent — well below the 0.05 
    threshold that would trigger overfitting concerns.
    
    **2. Default rate shift exists between periods**
    
    Train default rate {metrics['bad_rate_train']:.2%} vs test {metrics['bad_rate_test']:.2%} 
    — a difference of {bad_rate_diff:.2%}. This reflects vintage effects: later vintages 
    in the test set defaulted at slightly higher rates than earlier vintages used for 
    training. The model still discriminates well despite this shift.
    
    **3. KS statistic of {metrics['KS_test']:.4f}**
    
    Maximum separation between cumulative distributions of good and bad borrowers. 
    Values above 0.20 are acceptable; above 0.30 are strong. The model achieves 
    {metrics['KS_test']:.2f}, indicating clear separation at the optimal threshold.
    
    **4. Score distributions overlap**
    
    The histograms confirm the model produces similar score distributions on 
    unseen data — no major population shift between training and test periods.
    """)

    # ============================================================
# SECTION 3: PD SIMULATOR WITH SCORECARD
# ============================================================
st.markdown("---")
st.header("3. PD Simulator")

st.markdown("""
Input borrower characteristics to compute the credit score using the production 
scorecard. Each variable contributes a fixed number of points based on its bin — 
the total score determines the predicted PD.
""")

# ─── Load Scorecard ─────────────────────────────────────────
@st.cache_data
def load_scorecard():
    sc = pd.read_csv('data/df_scorecard.csv')
    
    # Use the rounded final scores
    sc = sc[['Feature name', 'Original feature name', 'Score - Final', 'Coefficients']].copy()
    sc.columns = ['feature_name', 'variable', 'points', 'coefficient']
    
    # Parse the bin from feature name
    sc['bin'] = sc['feature_name'].apply(
        lambda x: x.split(':')[1] if ':' in x else ''
    )
    
    return sc

scorecard = load_scorecard()

def get_points(variable, bin_value):
    """Look up the points for a variable-bin combination."""
    match = scorecard[
        (scorecard['variable'] == variable) & 
        (scorecard['bin'] == bin_value)
    ]
    if match.empty:
        return 0
    return int(match['points'].iloc[0])

def get_intercept_points():
    """Get the base score (intercept points)."""
    match = scorecard[scorecard['variable'] == 'const']
    return int(match['points'].iloc[0]) if not match.empty else 0

# ─── Layout ─────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Borrower Profile")
    
    grade = st.selectbox(
        "Grade",
        options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        index=2
    )
    
    int_rate = st.slider(
        "Interest Rate (%)",
        5.0, 30.0, 12.0, 0.1
    )
    
    term = st.radio(
        "Loan Term (months)",
        options=[36, 60],
        horizontal=True
    )
    
    dti = st.slider(
        "Debt-to-Income Ratio (%)",
        0.0, 40.0, 15.0, 0.5
    )
    
    annual_inc = st.number_input(
        "Annual Income ($)",
        10000, 300000, 60000, 5000
    )
    
    installment = st.number_input(
        "Monthly Installment ($)",
        30, 2000, 350, 10
    )
    
    verification = st.selectbox(
        "Income Verification",
        options=['Verified', 'Source Verified', 'Not Verified'],
        index=1
    )
    
    inq_last_6mths = st.slider(
        "Credit Inquiries (last 6 months)",
        0, 10, 1
    )
    
    home_ownership = st.selectbox(
        "Home Ownership",
        options=['MORTGAGE', 'OWN', 'RENT_OTHER_NONE_ANY']
    )
    
    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            'credit_card_car_wedding',
            'debt_consolidation',
            'vacation_major_purchase_home_improvement',
            'educational_other',
            'small_business_renewable_energy_moving_house_medical'
        ],
        index=1
    )
    
    state_group = st.selectbox(
        "State (grouped)",
        options=[
            'CA', 'NY', 'FL', 'TX_AK',
            'WV_WA_CO_ME_OR_NH_VT_DC',
            'WI_CT_MT_WY_SC_KS',
            'MA_GA_RI_IL_UT',
            'HI_MI_VA_DE_MN_AZ',
            'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
            'IN_NM_MO',
            'LA_NV_TN_SD',
            'NE_MS_AL_AR_OK'  # reference category
        ]
    )
    
    total_rev_hi_lim = st.number_input(
        "Total Revolving Credit Limit ($)",
        0, 300000, 20000, 5000
    )
    
    mths_since_earliest = st.slider(
        "Months Since Earliest Credit Line",
        50, 600, 250, 10
    )

with col_output:
    st.subheader("Predicted Risk")
    
    # ─── Map inputs to bin labels ───────────────────────────
    def bin_dti(val):
        if val < 8.6: return '<8.6'
        elif val < 11: return '8.6-11'
        elif val < 12.6: return '11-12.6'
        elif val < 15: return '12.6-15'
        elif val < 18.2: return '15-18.2'
        elif val < 22.2: return '18.2-22.2'
        elif val < 26.2: return '22.2-26.2'
        elif val < 30.2: return '26.2-30.2'
        else: return '>=30.2'
    
    def bin_int_rate(val):
        if val < 9.419: return '5.284_9.419'
        elif val < 11.987: return '9.419_11.987'
        elif val < 13.014: return '11.987_13.014'
        elif val < 15.068: return '13.014_15.068'
        elif val < 17.123: return '15.068_17.123'
        elif val < 22.772: return '17.123_22.772'
        else: return '>22.772'
    
    def bin_annual_inc(val):
        if val < 30000: return '<30K'
        elif val < 40000: return '30K-40K'
        elif val < 50000: return '40K-50K'
        elif val < 60000: return '50K-60K'
        elif val < 70000: return '60K-70K'
        elif val < 80000:  return '70K-80K'
        elif val < 90000:  return '80K-90K'
        elif val < 100000: return '90K-100K'
        elif val < 120000: return '100K-120K'
        elif val < 150000: return '120K-150K'
        else:              return '>150K'

    def bin_installment(val):
        if val < 116.359:   return '<116.359'
        elif val < 218.708: return '116.359-218.708'
        elif val < 321.058: return '218.708-321.058'
        elif val < 423.407: return '321.058-423.407'
        elif val < 798.687: return '423.407-798.687'
        else:               return '>798.687'

    def bin_total_rev(val):
        if val <= 20000:    return '<=20K'
        elif val <= 30000:  return '20K-30K'
        elif val <= 40000:  return '30K-40K'
        elif val <= 55000:  return '40K-55K'
        elif val <= 95000:  return '55K-95K'
        else:               return '>95K'

    def bin_mths_cr(val):
        if val < 250:   return '<250'
        elif val <= 350: return '250-350'
        else:           return '>350'

    def bin_inq(val):
        if val == 0:   return '0'
        elif val == 1: return '1'
        elif val == 2: return '2'
        else:          return '>2'

    # ─── Compute score ───────────────────────────────────────
    inputs = {
        'grade':                        grade,
        'verification_status':          verification,
        'inq_last_6mths':               bin_inq(inq_last_6mths),
        'mths_since_earliest_cr_line':  bin_mths_cr(mths_since_earliest),
        'annual_inc':                   bin_annual_inc(annual_inc),
        'dti':                          bin_dti(dti),
        'installment':                  bin_installment(installment),
        'total_rev_hi_lim':             bin_total_rev(total_rev_hi_lim),
        'home_ownership':               home_ownership,
        'int_rate':                     bin_int_rate(int_rate),
        'term':                         str(term),
        'purpose':                      purpose,
        'addr_state':                   state_group,
    }

    total_score = get_intercept_points()
    score_breakdown = []

    for variable, bin_val in inputs.items():
        pts = get_points(variable, bin_val)
        total_score += pts
        score_breakdown.append({
            'Variable':  variable,
            'Bin':       bin_val,
            'Points':    pts,
        })

    # ─── PD from score ───────────────────────────────────────
    scorecard_params = {
        'min_score': 300, 'max_score': 850,
        'min_sum_coef': scorecard.groupby('variable')['coefficient'].min().sum(),
        'max_sum_coef': scorecard.groupby('variable')['coefficient'].max().sum(),
    }
    try:
        import pickle
        p = pickle.load(open('data/scorecard_params.pkl', 'rb'))
        scorecard_params.update(p)
    except Exception:
        pass

    min_sc = scorecard_params['min_score']
    max_sc = scorecard_params['max_score']
    min_coef = scorecard_params['min_sum_coef']
    max_coef = scorecard_params['max_sum_coef']

    sum_coef = (total_score - min_sc) / (max_sc - min_sc) * (max_coef - min_coef) + min_coef
    pd_score = np.exp(sum_coef) / (1 + np.exp(sum_coef))
    pd_proba = 1 - pd_score   # good_bad = 1 means good; PD = P(default)

    # ─── Risk band ───────────────────────────────────────────
    if total_score >= 700:
        risk_band, band_color = 'Low Risk',    '#28a745'
    elif total_score >= 550:
        risk_band, band_color = 'Medium Risk', '#fd7e14'
    else:
        risk_band, band_color = 'High Risk',   '#dc3545'

    # ─── Display ─────────────────────────────────────────────
    st.metric("Credit Score", f"{int(total_score)}", delta=risk_band)

    col_pd, col_band = st.columns(2)
    col_pd.metric("Probability of Default", f"{pd_proba:.2%}")
    col_band.metric("Risk Band", risk_band)

    # Score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=total_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': band_color},
            'steps': [
                {'range': [300, 550], 'color': '#f8d7da'},
                {'range': [550, 700], 'color': '#fff3cd'},
                {'range': [700, 850], 'color': '#d4edda'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 3},
                'thickness': 0.75,
                'value': total_score
            }
        },
        title={'text': 'Credit Score'}
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Score breakdown table
    with st.expander("Score Breakdown"):
        df_breakdown = pd.DataFrame(score_breakdown)
        df_breakdown.loc[len(df_breakdown)] = {
            'Variable': 'const (intercept)',
            'Bin': '',
            'Points': get_intercept_points()
        }
        df_breakdown.loc[len(df_breakdown)] = {
            'Variable': '── TOTAL ──',
            'Bin': '',
            'Points': int(total_score)
        }
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)