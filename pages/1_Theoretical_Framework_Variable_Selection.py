import streamlit as st
import pandas as pd

st.set_page_config(page_title="Theoretical Framework", layout="wide")
st.title("Theoretical Framework & Variable Selection")

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    var_summary = pd.read_csv('data/variable_summary.csv')
    woe_data = pd.read_csv('data/woe_rebinned.csv')
    woe_data = pd.read_csv('data/woe_original.csv')
    return var_summary, woe_data

var_summary, woe_data = load_data()

st.header("1. The 5 Cs of Credit Framework")

st.markdown("""
The starting set includes **19 candidate variables** spanning four 5C dimensions. 
Collateral is absent because Lending Club loans are unsecured. Section 3 below 
explains which variables were kept and which were dropped through the selection process.
""")

five_c_df = pd.DataFrame({
    'C': ['Character', 'Capacity', 'Capital', 'Conditions', 'Collateral'],
    'Question': [
        'Does the borrower have a history of repayment discipline?',
        'Does the borrower have sufficient income to service the debt?',
        'Does the borrower have available credit resources / assets?',
        'What is the macroeconomic and product-specific environment?',
        'What assets secure the loan in case of default?'
    ],
    'Candidate Variables': [
        'grade, int_rate, verification_status, inq_last_6mths, delinq_2yrs, mths_since_earliest_cr_line',
        'dti, annual_inc, installment, emp_length_int',
        'home_ownership, total_rev_hi_lim, total_acc, open_acc',
        'term, purpose, addr_state, initial_list_status',
        'N/A — unsecured retail lending'
    ]
})
st.dataframe(five_c_df, use_container_width=True, hide_index=True)

st.info("""
**Note on Collateral:** Lending Club loans are unsecured consumer credit. 
There is no asset to pledge against the loan, so Collateral does not apply. 
This is why LGD for this portfolio is high (~85%) — recovery depends entirely 
on collections, not asset liquidation.
""")

st.markdown("---")
st.header("2. Variable Selection — Information Value")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(r"""
    **Weight of Evidence (WoE)** measures the log ratio between the proportion 
    of good borrowers and bad borrowers within a bin:
    """)
    
    st.latex(r"WoE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right)")
    
    st.markdown(r"""
    **Information Value (IV)** aggregates the WoE across all bins, weighted by 
    the distribution difference:
    """)
    
    st.latex(r"IV = \sum_i (\%Good_i - \%Bad_i) \times \ln\left(\frac{\%Good_i}{\%Bad_i}\right)")
    
    st.markdown("""
    **Selection logic:**
    
    Variables are kept if IV ≥ 0.02 (sufficient statistical signal) **or** 
    if IV < 0.02 but the variable captures a 5C dimension not already 
    represented by stronger predictors. Variables below threshold AND 
    redundant with stronger variables are dropped.
    """)

with col_right:
    iv_decision_table = pd.DataFrame({
        'IV Range': [
            '< 0.02',
            '< 0.02',
            '0.02 – 0.10',
            '0.10 – 0.30',
            '> 0.30'
        ],
        'Strength': [
            'Useless',
            'Useless',
            'Weak',
            'Moderate',
            'Strong'
        ],
        'Condition': [
            'Redundant with other variable',
            'Captures a 5C dimension',
            '—',
            '—',
            '—'
        ],
        'Decision': [
            'DROP',
            'KEEP',
            'KEEP',
            'KEEP',
            'KEEP'
        ],
        'Example': [
            'total_acc, open_acc',
            'purpose, addr_state',
            'inq_last_6mths',
            'term',
            'grade, int_rate'
        ]
    })
    
    def color_decision(val):
        if val == 'KEEP':
            return 'background-color: #d4edda'
        elif val == 'DROP':
            return 'background-color: #f8d7da'
        return ''
    
    styled = iv_decision_table.style.applymap(color_decision, subset=['Decision'])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    st.markdown("---")
st.header("3. Variable Selection Outcomes")

st.markdown("""
Each candidate variable is evaluated against the IV decision rule from Section 2. 
The table below shows the full selection outcome — final variables that 
enter the PD model are in green.
""")

# ─── Load from your saved CSV ───────────────────────────────
@st.cache_data
def load_variable_summary():
    df = pd.read_csv('data/variable_summary.csv')
    
    # Override rationale strings (single source of truth in the page file)
    rationale_overrides = {
        'addr_state': 'Geographic environment (Conditions). 50 states grouped into 12 WoE-homogeneous regions. IV below threshold but kept for 5C coverage.',
        'mths_since_earliest_cr_line': 'Length of credit history (Character). Longer history → lower risk. IV below threshold but kept as Character signal.',
        'purpose': 'Use of funds (Conditions). Small business / medical highest risk; credit card lowest. Kept for 5C coverage.',
    }
    
    df['rationale'] = df.apply(
        lambda row: rationale_overrides.get(row['variable'], row['rationale']),
        axis=1
    )
    
    return df

variable_outcomes = load_variable_summary()

# ─── Filter controls ────────────────────────────────────────
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    filter_category = st.multiselect(
        "Filter by 5C category:",
        options=sorted(variable_outcomes['category'].unique()),
        default=sorted(variable_outcomes['category'].unique())
    )

with col_filter2:
    filter_decision = st.multiselect(
        "Filter by decision:",
        options=sorted(variable_outcomes['decision'].unique()),
        default=sorted(variable_outcomes['decision'].unique())
    )

filtered_df = variable_outcomes[
    variable_outcomes['category'].isin(filter_category) &
    variable_outcomes['decision'].isin(filter_decision)
].copy()

# Sort by IV descending for the displayed view
filtered_df = filtered_df.sort_values('iv_rebinned', ascending=False)

# ─── Display with color coding ──────────────────────────────
def style_decision(val):
    colors = {
        'KEEP': 'background-color: #d4edda; color: #155724; font-weight: bold',
        'DROP': 'background-color: #f8d7da; color: #721c24'
    }
    return colors.get(val, '')

styled = (filtered_df.style
          .applymap(style_decision, subset=['decision'])
          .format({'iv_original': '{:.4f}', 'iv_rebinned': '{:.4f}'}))

st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

# ─── Summary metrics ────────────────────────────────────────
st.markdown("##### Selection Summary")
col1, col2, col3, col4 = st.columns(4)

kept = (variable_outcomes['decision'] == 'KEEP').sum()
dropped = (variable_outcomes['decision'] == 'DROP').sum()
total = len(variable_outcomes)

col1.metric("Total Candidates", total)
col2.metric("Kept", kept, delta=f"{kept/total:.0%}")
col3.metric("Dropped", dropped, delta=f"-{dropped/total:.0%}", delta_color="inverse")

# Average IV of kept variables
avg_iv_kept = variable_outcomes[variable_outcomes['decision'] == 'KEEP']['iv_rebinned'].mean()
col4.metric("Average IV (Kept)", f"{avg_iv_kept:.4f}")

# ─── Final model statement ──────────────────────────────────
kept_vars = variable_outcomes[variable_outcomes['decision'] == 'KEEP']
categories_covered = kept_vars['category'].nunique()
strongest = kept_vars.nlargest(3, 'iv_rebinned')['variable'].tolist()

st.success(f"""
**Final PD model: {kept} variables across {categories_covered} 5C dimensions.**

Strongest predictors: **{', '.join(strongest)}**.  
Collateral N/A — unsecured retail lending.
""")

st.info("""
**Note on `mths_since_issue_d`:** Excluded from variable selection by design. 
The temporal train/test split uses issue date as the splitting criterion, so 
including a variable derived from issue date would contaminate the model with 
information about the split itself.
""")