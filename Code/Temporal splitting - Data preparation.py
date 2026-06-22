# =============================================================================
# CREDIT RISK MODEL — PD MODEL DEVELOPMENT
# Theoretical Framework: The 5C's of Credit
# =============================================================================
#
# ┌─────────────────────────┬─────────────────────────────────────────────────┐
# │ Variable                │ 5C Category & Rationale                         │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ grade                   │ CHARACTER — LC risk grade based on FICO & DTI   │
# │ delinq_2yrs             │ CHARACTER — Recent delinquency history           │
# │ inq_last_6mths          │ CHARACTER — Recent credit-seeking behaviour      │
# │ verification_status     │ CHARACTER — Income credibility signal            │
# │ mths_since_earliest_cr  │ CHARACTER — Length of credit history             │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ annual_inc              │ CAPACITY  — Income level                        │
# │ dti                     │ CAPACITY  — Debt burden relative to income      │
# │ emp_length              │ CAPACITY  — Employment stability                │
# │ installment             │ CAPACITY  — Monthly repayment obligation        │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ total_rev_hi_lim        │ CAPITAL   — Total revolving credit access       │
# │ total_acc               │ CAPITAL   — Breadth of credit relationships     │
# │ open_acc                │ CAPITAL   — Active credit lines                 │
# │ home_ownership          │ CAPITAL   — Property as financial asset         │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ int_rate                │ CONDITIONS — Risk-based pricing of the loan     │
# │ term                    │ CONDITIONS — Loan repayment horizon             │
# │ purpose                 │ CONDITIONS — Use of funds                       │
# │ mths_since_issue_d      │ CONDITIONS — Loan seasoning / market timing     │
# │ initial_list_status     │ CONDITIONS — Platform listing type              │
# │ addr_state              │ CONDITIONS — Geographic / economic environment  │
# └─────────────────────────┴─────────────────────────────────────────────────┘

# %%
# =============================================================================
# 0. MEMORY CHECK
# =============================================================================
import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
print(f"Used memory:      {psutil.virtual_memory().used   / 1024**3:.2f} GB")
print(f"Total memory:     {psutil.virtual_memory().total  / 1024**3:.2f} GB")
print(f"Memory usage:     {psutil.virtual_memory().percent:.1f}%")

# %%
# =============================================================================
# 1. IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.4f}'.format
sns.set()

# %%
# =============================================================================
# 2. DATA
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 SOURCE
# -----------------------------------------------------------------------------
# Dataset : LendingClub Loan Data (2007–2018 Q4)
# File    : accepted_2007_to_2018q4.csv
# Source  : LendingClub public dataset
# Size    : ~2.2 million accepted loan applications
# Period  : January 2007 – December 2018
# -----------------------------------------------------------------------------

cols_needed = [
    'id',
    # Target
    'loan_status',
    # CHARACTER
    'grade', 'sub_grade',
    'delinq_2yrs', 'mths_since_last_delinq',
    'pub_rec', 'acc_now_delinq',
    'inq_last_6mths',
    'earliest_cr_line',
    'verification_status',
    # CAPACITY
    'annual_inc', 'annual_inc_joint',
    'dti', 'dti_joint',
    'emp_length',
    'installment',
    'application_type',
    # CAPITAL
    'total_rev_hi_lim',
    'total_acc',
    'open_acc',
    'home_ownership',
    'funded_amnt',
    # CONDITIONS
    'int_rate',
    'term',
    'purpose',
    'issue_d',
    'initial_list_status',
    'addr_state',
]

loan_data = pd.read_csv(
    'ifrs9_app/data/accepted_2007_to_2018q4.csv',
    usecols=cols_needed
)
print(f"Raw dataset shape: {loan_data.shape}")
del cols_needed

# %%
# -----------------------------------------------------------------------------
# 2.2 TARGET DEFINITION — GOOD / BAD FLAG
# -----------------------------------------------------------------------------
# BAD  (0): Charged Off, Default,
#           Does not meet credit policy — Charged Off
# GOOD (1): Fully Paid,
#           Does not meet credit policy — Fully Paid
# -----------------------------------------------------------------------------
finished_statuses = [
    'Fully Paid',
    'Charged Off',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
]

loan_data = loan_data[loan_data['loan_status'].isin(finished_statuses)]

print(f"Loans after cycle filter: {len(loan_data):,}")
print(loan_data['loan_status'].value_counts())

# -----------------------------------------------------------------------------
loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin([
        'Charged Off',
        'Does not meet the credit policy. Status:Charged Off',
    ]), 0, 1)

good_bad_pct = loan_data['good_bad'].value_counts(normalize=True) * 100
print(f"\nGood (1): {good_bad_pct[1]:.1f}%  |  Bad (0): {good_bad_pct[0]:.1f}%")

# %%
# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

# --- emp_length → numeric [CAPACITY] ---
loan_data['emp_length_int'] = (loan_data['emp_length']
    .str.replace('\+ years', '',    regex=True)
    .str.replace('< 1 year',  '0', regex=True)
    .str.replace(' years',    '',   regex=True)
    .str.replace(' year',     '',   regex=True)
    .fillna(0).astype(float))
loan_data.drop(columns=['emp_length'], inplace=True)

# %%
# --- earliest_cr_line → months since [CHARACTER] ---
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format='%b-%Y')
loan_data['mths_since_earliest_cr_line'] = round(
    pd.to_numeric((pd.to_datetime('2026-01-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'D')) / 30.5)
loan_data['mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].fillna(0)
with pd.option_context('display.float_format', '{:.2f}'.format):
    print(loan_data['mths_since_earliest_cr_line'].describe())
loan_data.drop(columns=['earliest_cr_line_date'], inplace=True)    

# %%
# --- term → numeric [CONDITIONS] ---
loan_data['term_int'] = pd.to_numeric(
    loan_data['term'].str.strip().str.replace(' months', '', regex=False))
loan_data.drop(columns=['term'], inplace=True)

# %%
# --- issue_d → months since [CONDITIONS] ---
loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%Y')

# %%
# =============================================================================
# 4. MISSING VALUE TREATMENT
# =============================================================================
# total_rev_hi_lim → funded_amnt proxy       [CAPITAL]
# annual_inc       → mean imputation          [CAPACITY]
# Count variables  → 0 (no events recorded)  [CHARACTER / CAPITAL]
# -----------------------------------------------------------------------------
loan_data['total_rev_hi_lim'] = loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'])
loan_data['annual_inc']        = loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean())

for col in ['mths_since_earliest_cr_line', 'acc_now_delinq', 'total_acc',
            'pub_rec', 'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'emp_length_int']:
    loan_data[col] = loan_data[col].fillna(0)

# %%
# =============================================================================
# 5. DUMMY VARIABLES
# =============================================================================
loan_data_dummies = pd.concat([
    # CHARACTER
    pd.get_dummies(loan_data['grade'],              prefix='grade',              prefix_sep=':'),
    pd.get_dummies(loan_data['verification_status'], prefix='verification_status', prefix_sep=':'),
    pd.get_dummies(loan_data['loan_status'],         prefix='loan_status',         prefix_sep=':'),
    # CAPITAL
    pd.get_dummies(loan_data['home_ownership'],      prefix='home_ownership',      prefix_sep=':'),
    # CONDITIONS
    pd.get_dummies(loan_data['initial_list_status'], prefix='initial_list_status', prefix_sep=':'),
    pd.get_dummies(loan_data['addr_state'],          prefix='addr_state',          prefix_sep=':'),
    pd.get_dummies(loan_data['purpose'],             prefix='purpose',             prefix_sep=':'),
], axis=1)

loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)
del loan_data_dummies

# %%
# =============================================================================
# 6. TRAIN / TEST SPLIT — Temporal (time-based on issue_d_date)
# =============================================================================
cutoff = loan_data['issue_d_date'].quantile(0.80)
print(f"Temporal cutoff: {cutoff.strftime('%b-%Y')}")

train_mask = loan_data['issue_d_date'] <= cutoff   # older  80%
test_mask  = loan_data['issue_d_date'] >  cutoff   # recent 20%

loan_data_train = loan_data[train_mask].drop(['issue_d_date'], axis=1)
loan_data_test  = loan_data[test_mask].drop(['issue_d_date'], axis=1)

print(f"Train : {len(loan_data_train):,}  | Bad rate: {1 - loan_data_train['good_bad'].mean():.2%}")
print(f"Test  : {len(loan_data_test):,}  | Bad rate: {1 - loan_data_test['good_bad'].mean():.2%}")
# %%
# =============================================================================
# 7. WoE & IV FUNCTIONS
# =============================================================================
# WoE(i) = ln[ P(Good|i) / P(Bad|i) ]
# IV      = Σ (P(Good|i) - P(Bad|i)) × WoE(i)
#
# IV thresholds:
#   < 0.02        → Useless    → DROP
#   0.02 – 0.10   → Weak       → CONSIDER
#   0.10 – 0.30   → Medium     → KEEP
#   0.30 – 0.50   → Strong     → KEEP
#   > 0.50        → Suspicious → REVIEW
# =============================================================================
def woe_discrete(df, discrete_variable_name, good_bad_variable_name='good_bad'):
    df = df[[discrete_variable_name, good_bad_variable_name]].copy().reset_index(drop=True)
    df.columns = [discrete_variable_name, 'good_bad']
    
    grouped = df.groupby(discrete_variable_name, as_index=False)['good_bad'].agg(
        n_obs='count', prop_good='mean')
    grouped['prop_n_obs']  = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good']      = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad']       = (1 - grouped['prop_good']) * grouped['n_obs']
    grouped['prop_n_good'] = grouped['n_good'] / grouped['n_good'].sum()
    grouped['prop_n_bad']  = grouped['n_bad']  / grouped['n_bad'].sum()
    grouped['WoE']         = np.log((grouped['prop_n_good'] + 0.00001) / (grouped['prop_n_bad'] + 0.00001))
    grouped = grouped.sort_values(['WoE']).reset_index(drop=True)
    grouped['diff_prop_good'] = grouped['prop_good'].diff().abs()
    grouped['diff_WoE']       = grouped['WoE'].diff().abs()
    grouped['IV']             = (grouped['prop_n_good'] - grouped['prop_n_bad']) * grouped['WoE']
    grouped['IV']             = grouped['IV'].sum()
    return grouped 

def woe_continuous(df, continuous_variable_name, good_bad_variable_name='good_bad'):
    df = df[[continuous_variable_name, good_bad_variable_name]].copy().reset_index(drop=True)
    df.columns = [continuous_variable_name, 'good_bad']
    
    grouped = df.groupby(continuous_variable_name, as_index=False)['good_bad'].agg(
        n_obs='count', prop_good='mean')
    grouped['prop_n_obs']  = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good']      = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad']       = (1 - grouped['prop_good']) * grouped['n_obs']
    grouped['prop_n_good'] = grouped['n_good'] / grouped['n_good'].sum()
    grouped['prop_n_bad']  = grouped['n_bad']  / grouped['n_bad'].sum()
    grouped['WoE']         = np.log((grouped['prop_n_good'] + 0.00001) / (grouped['prop_n_bad'] + 0.00001))
    grouped['diff_prop_good'] = grouped['prop_good'].diff().abs()
    grouped['diff_WoE']       = grouped['WoE'].diff().abs()
    grouped['IV']             = (grouped['prop_n_good'] - grouped['prop_n_bad']) * grouped['WoE']
    grouped['IV']             = grouped['IV'].sum()
    return grouped

def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    x     = np.array(df_WoE.iloc[:, 0].apply(str))
    y_WoE = df_WoE['WoE']
    y_obs = df_WoE['prop_n_obs']
    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.bar(x, y_obs, color='steelblue', alpha=0.6, label='Proportion of Observations')
    ax1.set_xlabel(df_WoE.columns[0])
    ax1.set_ylabel('Proportion of Observations', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticklabels(x, rotation=rotation_of_x_axis_labels)
    ax2 = ax1.twinx()
    ax2.plot(x, y_WoE, marker='o', linestyle='--', color='k', label='WoE')
    ax2.set_ylabel('Weight of Evidence', color='k')
    plt.title('Weight of Evidence by ' + df_WoE.columns[0])
    fig.legend(loc='upper right')
    plt.show()

# ── Three CSV Collectors ──────────────────────────────────────────────────────
original_results = []   # fine-classing  → woe_original.csv
rebinned_results = []   # coarse-classing → woe_rebinned.csv
summary_results  = []   # one row/variable → variable_summary.csv

def collect_original(df_temp, variable_name):
    """Granular WoE before binning."""
    df = df_temp[[df_temp.columns[0], 'prop_n_obs', 'prop_n_good',
                  'prop_n_bad', 'WoE', 'IV']].copy()
    df.columns = ['bin', 'proportion', 'prop_n_good', 'prop_n_bad', 'WoE', 'IV']
    df['iv_contribution'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df.insert(0, 'variable', variable_name)
    return df[['variable', 'bin', 'proportion', 'WoE', 'iv_contribution', 'IV']]

def collect_rebinned(df_temp, variable_name, decision):
    """Final WoE after coarse-classing."""
    df = df_temp[[df_temp.columns[0], 'prop_n_obs', 'prop_n_good',
                  'prop_n_bad', 'WoE', 'IV']].copy()
    df.columns = ['bin', 'proportion', 'prop_n_good', 'prop_n_bad', 'WoE', 'IV']
    df['iv_contribution'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df.insert(0, 'variable', variable_name)
    df['decision'] = decision
    return df[['variable', 'bin', 'proportion', 'WoE', 'iv_contribution', 'IV', 'decision']]

def collect_summary(variable, category, iv_original, iv_rebinned,
                    n_bins, decision, rationale):
    """One row per variable."""
    summary_results.append({
        'variable':    variable,
        'category':    category,
        'iv_original': round(iv_original, 4),
        'iv_rebinned': round(iv_rebinned, 4),
        'n_bins':      n_bins,
        'decision':    decision,
        'rationale':   rationale,
    })
    
# =============================================================================
# 8. SELECT DATASET FOR PREPROCESSING
# =============================================================================
#####
# Variable/IV selection is a training-time decision — compute WoE & IV on the
# training split only, never the out-of-time test set (avoids look-ahead bias).
df_inputs_prepr  = loan_data_train
#####
# df_inputs_prepr  = loan_data_inputs_test
# df_targets_prepr = loan_data_targets_test

# %%
# =============================================================================
# 9. WoE ANALYSIS & COARSE CLASSING — ORGANISED BY 5C FRAMEWORK
# =============================================================================

# #############################################################################
# C1 — CHARACTER
# #############################################################################

# -----------------------------------------------------------------------------
# grade                    [CHARACTER — Primary LC risk indicator]
# -----------------------------------------------------------------------------
df_temp = woe_discrete(df_inputs_prepr, 'grade')
plot_by_woe(df_temp)
iv_grade = df_temp['IV'].values[0]
print(f"IV (grade): {iv_grade:.4f}")
# DECISION: KEEP — 7 categories, no grouping needed.
original_results.append(collect_original(df_temp, 'grade'))   # same before/after (no rebinning)
rebinned_results.append(collect_rebinned(df_temp, 'grade', 'KEEP'))
collect_summary('grade', 'CHARACTER', iv_grade, iv_grade, 7, 'KEEP',
                'Strong monotonic WoE. LC risk grade captures FICO + DTI. No grouping needed.')

# %%
# -----------------------------------------------------------------------------
# delinq_2yrs              [CHARACTER — Recent delinquency history]
# -----------------------------------------------------------------------------
df_temp_fine = woe_continuous(df_inputs_prepr, 'delinq_2yrs')
plot_by_woe(df_temp_fine)
iv_fine = df_temp_fine['IV'].values[0]
print(f"IV (delinq_2yrs — fine): {iv_fine:.4f}")

df_inputs_prepr['delinq_2yrs_binned'] = pd.cut(
    df_inputs_prepr['delinq_2yrs'],
    bins=[-1, 0, 1, df_inputs_prepr['delinq_2yrs'].max()],
    labels=['0', '1', '>1'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'delinq_2yrs_binned')
plot_by_woe(df_temp_coarse)
iv_coarse = df_temp_coarse['IV'].values[0]
print(f"IV (delinq_2yrs — coarse): {iv_coarse:.4f}")
df_inputs_prepr.drop(columns=['delinq_2yrs_binned'], inplace=True)

# %%
# DECISION: KEEP — 3 bins.
original_results.append(collect_original(df_temp_fine,   'delinq_2yrs'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'delinq_2yrs', 'DROP'))
collect_summary('delinq_2yrs', 'CHARACTER', iv_fine, iv_coarse, 3, 'DROP',
                'Recent delinquency history. 3 bins: 0, 1, >1.')

df_inputs_prepr['delinq_2yrs:0']  = np.where( df_inputs_prepr['delinq_2yrs'] == 0, 1, 0)
df_inputs_prepr['delinq_2yrs:1']  = np.where( df_inputs_prepr['delinq_2yrs'] == 1, 1, 0)
df_inputs_prepr['delinq_2yrs:>1'] = np.where( df_inputs_prepr['delinq_2yrs'] >  1, 1, 0)

# %%
# -----------------------------------------------------------------------------
# inq_last_6mths           [CHARACTER — Recent credit-seeking behaviour]
# -----------------------------------------------------------------------------
df_temp_fine = woe_continuous(df_inputs_prepr, 'inq_last_6mths')
plot_by_woe(df_temp_fine)
iv_fine = df_temp_fine['IV'].values[0]

df_inputs_prepr['inq_last_6mths_binned'] = pd.cut(
    df_inputs_prepr['inq_last_6mths'],
    bins=[-1, 0, 1, 2, df_inputs_prepr['inq_last_6mths'].max()],
    labels=['0', '1', '2', '>2'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'inq_last_6mths_binned')
plot_by_woe(df_temp_coarse)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['inq_last_6mths_binned'], inplace=True)

# DECISION: KEEP — 4 bins.
original_results.append(collect_original(df_temp_fine,   'inq_last_6mths'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'inq_last_6mths', 'KEEP'))
collect_summary('inq_last_6mths', 'CHARACTER', iv_fine, iv_coarse, 4, 'KEEP',
                'Recent credit-seeking behaviour. Monotonically decreasing WoE.')

df_inputs_prepr['inq_last_6mths:0']  = np.where( df_inputs_prepr['inq_last_6mths'] == 0, 1, 0)
df_inputs_prepr['inq_last_6mths:1']  = np.where((df_inputs_prepr['inq_last_6mths'] > 0) & (df_inputs_prepr['inq_last_6mths'] <= 1), 1, 0)
df_inputs_prepr['inq_last_6mths:2']  = np.where((df_inputs_prepr['inq_last_6mths'] > 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:>2'] = np.where( df_inputs_prepr['inq_last_6mths'] > 2,  1, 0)

# %%
# -----------------------------------------------------------------------------
# verification_status      [CHARACTER — Income credibility]
# -----------------------------------------------------------------------------
df_temp = woe_discrete(df_inputs_prepr, 'verification_status')
plot_by_woe(df_temp)
iv_vs = df_temp['IV'].values[0]
# DECISION: KEEP — 3 categories, no grouping needed.
original_results.append(collect_original(df_temp, 'verification_status'))
rebinned_results.append(collect_rebinned(df_temp, 'verification_status', 'KEEP'))
collect_summary('verification_status', 'CHARACTER', iv_vs, iv_vs, 3, 'KEEP',
                'Income credibility signal. 3 categories with clear WoE separation.')

# %%
# -----------------------------------------------------------------------------
# mths_since_earliest_cr_line_factor      [CHARACTER — Length of credit history]
# -----------------------------------------------------------------------------
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(
    df_inputs_prepr['mths_since_earliest_cr_line'], 50)
df_temp_fine = woe_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor')
plot_by_woe(df_temp_fine, 90)
iv_fine = df_temp_fine['IV'].values[0]
df_inputs_prepr.drop(columns=['mths_since_earliest_cr_line_factor'], inplace=True)

df_inputs_prepr['mths_since_earliest_cr_line_binned'] = pd.cut(
    df_inputs_prepr['mths_since_earliest_cr_line'],
    bins=[-np.inf, 250, 350, np.inf], labels=['<250', '250-350', '>350'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_binned')
plot_by_woe(df_temp_coarse, 90)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['mths_since_earliest_cr_line_binned'], inplace=True)

# DECISION: KEEP — 3 bins.
original_results.append(collect_original(df_temp_fine,   'mths_since_earliest_cr_line'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'mths_since_earliest_cr_line', 'KEEP'))
collect_summary('mths_since_earliest_cr_line', 'CHARACTER', iv_fine, iv_coarse, 3, 'KEEP',
                'Length of credit history. Longer history → lower risk.')

df_inputs_prepr['mths_since_earliest_cr_line:<250']   = np.where( df_inputs_prepr['mths_since_earliest_cr_line'] < 250, 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:250-350'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line'] >= 250) & (df_inputs_prepr['mths_since_earliest_cr_line'] < 300), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>350']   = np.where( df_inputs_prepr['mths_since_earliest_cr_line'] >= 350, 1, 0)

# %%
# #############################################################################
# C2 — CAPACITY
# #############################################################################

# -----------------------------------------------------------------------------
# annual_inc               [CAPACITY — Income level]
# -----------------------------------------------------------------------------
print(f"95th percentile annual_inc: {df_inputs_prepr['annual_inc'].quantile(0.95):,.0f}")

# 50 equal-width fine bins up to 160K (keeps the fine WoE readable against income
# outliers), plus one extra bin for the >160K tail so the fine IV still covers the
# full population — same rows as the coarse binning below.
fine_edges = np.linspace(df_inputs_prepr['annual_inc'].min(), 160000, 51)
fine_edges = np.append(fine_edges, df_inputs_prepr['annual_inc'].max())
df_inputs_prepr['annual_inc_factor'] = pd.cut(
    df_inputs_prepr['annual_inc'], bins=fine_edges, include_lowest=True)
df_temp_fine = woe_continuous(df_inputs_prepr, 'annual_inc_factor')
plot_by_woe(df_temp_fine, 90)
iv_fine = df_temp_fine['IV'].values[0]
df_inputs_prepr.drop(columns=['annual_inc_factor'], inplace=True)

df_inputs_prepr['annual_inc_binned'] = pd.cut(
    df_inputs_prepr['annual_inc'],
    bins=[-1, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
          100000, 120000, 150000, df_inputs_prepr['annual_inc'].max()],
    labels=['<30K','30K-40K','40K-50K','50K-60K','60K-70K','70K-80K',
            '80K-90K','90K-100K','100K-120K','120K-150K','>150K'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'annual_inc_binned')
plot_by_woe(df_temp_coarse, 90)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['annual_inc_binned'], inplace=True)

# %%
# DECISION: KEEP — 11 bins.
original_results.append(collect_original(df_temp_fine,   'annual_inc'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'annual_inc', 'KEEP'))
collect_summary('annual_inc', 'CAPACITY', iv_fine, iv_coarse, 11, 'KEEP',
                'Income level. Monotonically increasing WoE. 11 bins up to >150K.')

new_cols = {
    'annual_inc:<30K':      df_inputs_prepr['annual_inc'] <= 30000,
    'annual_inc:30K-40K':   (df_inputs_prepr['annual_inc'] > 30000)  & (df_inputs_prepr['annual_inc'] <= 40000),
    'annual_inc:40K-50K':   (df_inputs_prepr['annual_inc'] > 40000)  & (df_inputs_prepr['annual_inc'] <= 50000),
    'annual_inc:50K-60K':   (df_inputs_prepr['annual_inc'] > 50000)  & (df_inputs_prepr['annual_inc'] <= 60000),
    'annual_inc:60K-70K':   (df_inputs_prepr['annual_inc'] > 60000)  & (df_inputs_prepr['annual_inc'] <= 70000),
    'annual_inc:70K-80K':   (df_inputs_prepr['annual_inc'] > 70000)  & (df_inputs_prepr['annual_inc'] <= 80000),
    'annual_inc:80K-90K':   (df_inputs_prepr['annual_inc'] > 80000)  & (df_inputs_prepr['annual_inc'] <= 90000),
    'annual_inc:90K-100K':  (df_inputs_prepr['annual_inc'] > 90000)  & (df_inputs_prepr['annual_inc'] <= 100000),
    'annual_inc:100K-120K': (df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000),
    'annual_inc:120K-150K': (df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 150000),
    'annual_inc:>150K':     df_inputs_prepr['annual_inc'] > 150000,
}
df_inputs_prepr = pd.concat(
    [df_inputs_prepr, pd.DataFrame(new_cols, index=df_inputs_prepr.index).astype(int)], axis=1)

# %%
# -----------------------------------------------------------------------------
# dti                      [CAPACITY — Debt burden relative to income]
# -----------------------------------------------------------------------------
# 50 equal-width fine bins up to 39 (keeps the fine WoE readable against DTI
# outliers, which reach 999), plus one extra bin for the >39 tail so the fine IV
# still covers the full population — same rows as the coarse binning below.
fine_edges = np.linspace(df_inputs_prepr['dti'].min(), 39, 51)
fine_edges = np.append(fine_edges, df_inputs_prepr['dti'].max())
df_inputs_prepr['dti_factor'] = pd.cut(
    df_inputs_prepr['dti'], bins=fine_edges, include_lowest=True)
df_temp_fine = woe_continuous(df_inputs_prepr, 'dti_factor')
plot_by_woe(df_temp_fine, 90)
iv_fine = df_temp_fine['IV'].values[0]
df_inputs_prepr.drop(columns=['dti_factor'], inplace=True)

df_inputs_prepr['dti_binned'] = pd.cut(
    df_inputs_prepr['dti'],
    bins=[-1, 8.6, 11, 12.6, 15, 18.2, 22.2, 26.2, 30.2, df_inputs_prepr['dti'].max()],
    labels=['<8.6','8.6-11','11-12.6','12.6-15','15-18.2','18.2-22.2','22.2-26.2','26.2-30.2','>=30.2'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'dti_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['dti_binned'], inplace=True)

# %%
# DECISION: KEEP — 9 bins.
original_results.append(collect_original(df_temp_fine,   'dti'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'dti', 'KEEP'))
collect_summary('dti', 'CAPACITY', iv_fine, iv_coarse, 9, 'KEEP',
                'Debt burden relative to income. Monotonically decreasing WoE.')

new_cols = {
    'dti:<8.6':      df_inputs_prepr['dti'] < 8.6,
    'dti:8.6-11':    (df_inputs_prepr['dti'] >= 8.6)  & (df_inputs_prepr['dti'] < 11),
    'dti:11-12.6':   (df_inputs_prepr['dti'] >= 11)   & (df_inputs_prepr['dti'] < 12.6),
    'dti:12.6-15':   (df_inputs_prepr['dti'] >= 12.6) & (df_inputs_prepr['dti'] < 15),
    'dti:15-18.2':   (df_inputs_prepr['dti'] >= 15)   & (df_inputs_prepr['dti'] < 18.2),
    'dti:18.2-22.2': (df_inputs_prepr['dti'] >= 18.2) & (df_inputs_prepr['dti'] < 22.2),
    'dti:22.2-26.2': (df_inputs_prepr['dti'] >= 22.2) & (df_inputs_prepr['dti'] < 26.2),
    'dti:26.2-30.2': (df_inputs_prepr['dti'] >= 26.2) & (df_inputs_prepr['dti'] < 30.2),
    'dti:>=30.2':    df_inputs_prepr['dti'] >= 30.2,
}
df_inputs_prepr = pd.concat(
    [df_inputs_prepr, pd.DataFrame(new_cols, index=df_inputs_prepr.index).astype(int)], axis=1)

# %%
# -----------------------------------------------------------------------------
# emp_length_int           [CAPACITY — Employment stability]
# -----------------------------------------------------------------------------
# Years employed (0–10, where 10 = 10+ years). Proxy for income stability.
# WoE: Increasing trend — longer employment correlates with lower risk.
# Fine-classing: 11 bins (0–10).
# Binning: 5 bins (0 | 1 | 2-4 | 5-9 | 10).
# IV:  Weak (<0.02).
# DECISION: KEEP — 5 bins.
# -----------------------------------------------------------------------------
df_temp_fine = woe_continuous(df_inputs_prepr, 'emp_length_int')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr['emp_length_int_binned'] = pd.cut(
    df_inputs_prepr['emp_length_int'],
    bins=[-np.inf, 0, 4, 9, np.inf], labels=['0', '1-4', '5-9', '10'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'emp_length_int_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['emp_length_int_binned'], inplace=True)

# %%
# DECISION: KEEP — 4 bins.
original_results.append(collect_original(df_temp_fine,   'emp_length_int'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'emp_length_int', 'DROP'))
collect_summary('emp_length_int', 'CAPACITY', iv_fine, iv_coarse, 4, 'DROP',
                'Employment stability. Longer tenure correlates with lower risk.')

df_inputs_prepr['emp_length_int:0']    = np.where( df_inputs_prepr['emp_length_int'] == 0, 1, 0)
df_inputs_prepr['emp_length_int:1-4']  = np.where((df_inputs_prepr['emp_length_int'] >= 1) & (df_inputs_prepr['emp_length_int'] <= 4), 1, 0)
df_inputs_prepr['emp_length_int:5-9']  = np.where((df_inputs_prepr['emp_length_int'] >= 5) & (df_inputs_prepr['emp_length_int'] <= 9), 1, 0)
df_inputs_prepr['emp_length_int:10']   = np.where( df_inputs_prepr['emp_length_int'] == 10, 1, 0)

# %%
# -----------------------------------------------------------------------------
# installment              [CAPACITY — Monthly payment obligation]
# -----------------------------------------------------------------------------
# Fixed monthly payment amount.
# WoE: Monotonically decreasing — higher payments correlate with higher risk.
# Fine-classing: 50 bins.
# Binning: 6 bins with clear WoE boundaries.
# IV:  Medium (~0.03).
# DECISION: KEEP — 6 bins.
# -----------------------------------------------------------------------------
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
df_temp_fine = woe_continuous(df_inputs_prepr, 'installment_factor')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr.drop(columns=['installment_factor'], inplace=True)

df_inputs_prepr['installment_binned'] = pd.cut(
    df_inputs_prepr['installment'],
    bins=[-np.inf, 116.359, 218.708, 321.058, 423.407, 798.687, np.inf],
    labels=['<116.359','116.359-218.708','218.708-321.058','321.058-423.407','423.407-798.687','>798.687'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'installment_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['installment_binned'], inplace=True)

# %%
# DECISION: KEEP — 6 bins.
original_results.append(collect_original(df_temp_fine,   'installment'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'installment', 'KEEP'))
collect_summary('installment', 'CAPACITY', iv_fine, iv_coarse, 6, 'KEEP',
                'Monthly payment obligation. Lower installment → better credit quality.')

df_inputs_prepr['installment:<116.359']      = np.where( df_inputs_prepr['installment'] < 116.359, 1, 0)
df_inputs_prepr['installment:116.359-218.708'] = np.where((df_inputs_prepr['installment'] >= 116.359) & (df_inputs_prepr['installment'] < 218.708), 1, 0)
df_inputs_prepr['installment:218.708-321.058'] = np.where((df_inputs_prepr['installment'] >= 218.708) & (df_inputs_prepr['installment'] < 321.058), 1, 0)
df_inputs_prepr['installment:321.058-423.407'] = np.where((df_inputs_prepr['installment'] >= 321.058) & (df_inputs_prepr['installment'] < 423.407), 1, 0)
df_inputs_prepr['installment:423.407-798.687'] = np.where((df_inputs_prepr['installment'] >= 423.407) & (df_inputs_prepr['installment'] < 798.687), 1, 0)
df_inputs_prepr['installment:>798.687']      = np.where( df_inputs_prepr['installment'] >= 798.687, 1, 0)

# %%
# #############################################################################
# C3 — CAPITAL
# #############################################################################

# -----------------------------------------------------------------------------
# total_rev_hi_lim         [CAPITAL — Total revolving credit access]
# -----------------------------------------------------------------------------
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
df_temp_fine = woe_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr.drop(columns=['total_rev_hi_lim_factor'], inplace=True)

df_inputs_prepr['total_rev_hi_lim_binned'] = pd.cut(
    df_inputs_prepr['total_rev_hi_lim'],
    bins=[-1, 20000, 30000, 40000, 55000, 95000, df_inputs_prepr['total_rev_hi_lim'].max()],
    labels=['<=20K','20K-30K','30K-40K','40K-55K','55K-95K','>95K'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'total_rev_hi_lim_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['total_rev_hi_lim_binned'], inplace=True)

# %%
# DECISION: KEEP — 6 bins.
original_results.append(collect_original(df_temp_fine,   'total_rev_hi_lim'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'total_rev_hi_lim', 'KEEP'))
collect_summary('total_rev_hi_lim', 'CAPITAL', iv_fine, iv_coarse, 6, 'KEEP',
                'Total revolving credit access. Higher limit → lower risk.')

new_cols = {
    'total_rev_hi_lim:<=20K':   df_inputs_prepr['total_rev_hi_lim'] <= 20000,
    'total_rev_hi_lim:20K-30K': (df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000),
    'total_rev_hi_lim:30K-40K': (df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000),
    'total_rev_hi_lim:40K-55K': (df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000),
    'total_rev_hi_lim:55K-95K': (df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000),
    'total_rev_hi_lim:>95K':    df_inputs_prepr['total_rev_hi_lim'] > 95000,
}
df_inputs_prepr = pd.concat(
    [df_inputs_prepr, pd.DataFrame(new_cols, index=df_inputs_prepr.index).astype(int)], axis=1)

# %%
# -----------------------------------------------------------------------------
# home_ownership           [CAPITAL — Property as financial asset]
# -----------------------------------------------------------------------------
df_temp_fine = woe_discrete(df_inputs_prepr, 'home_ownership')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)

# Test coarse bins: OWN | MORTGAGE | RENT_OTHER_NONE_ANY (3 groups)
home_ownership_map = {
    'OWN':      'OWN',
    'MORTGAGE': 'MORTGAGE',
    'RENT':     'RENT_OTHER_NONE_ANY',
    'OTHER':    'RENT_OTHER_NONE_ANY',
    'NONE':     'RENT_OTHER_NONE_ANY',
    'ANY':      'RENT_OTHER_NONE_ANY',
}
df_inputs_prepr['home_ownership_binned'] = df_inputs_prepr['home_ownership'].map(home_ownership_map)
df_temp_coarse = woe_discrete(df_inputs_prepr, 'home_ownership_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['home_ownership_binned'], inplace=True)

# %%
# DECISION: KEEP — 3 groups.
original_results.append(collect_original(df_temp_fine,   'home_ownership'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'home_ownership', 'KEEP'))
collect_summary('home_ownership', 'CAPITAL', iv_fine, iv_coarse, 3, 'KEEP',
                'Property as financial asset. RENT/OTHER/NONE/ANY merged (similar WoE).')

df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([
    df_inputs_prepr['home_ownership:RENT'],  df_inputs_prepr['home_ownership:OTHER'],
    df_inputs_prepr['home_ownership:NONE'],  df_inputs_prepr['home_ownership:ANY']
])

# %%
# -----------------------------------------------------------------------------
# total_acc                [CAPITAL — Total credit lines ever opened]
# -----------------------------------------------------------------------------
# Total number of credit lines in the borrower's credit file.
# WoE: Monotonically increasing — more credit lines correlates with lower risk.
# Fine-classing: 50 bins.
# Binning: 5 bins (0-10 | 11-20 | 21-30 | 31-40 | >40).
# IV:  Negligible (0.0014) — below threshold.
# DECISION: DROP — insufficient predictive power.
# -----------------------------------------------------------------------------
df_temp_fine = woe_continuous(df_inputs_prepr, 'total_acc')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr['total_acc_binned'] = pd.cut(
    df_inputs_prepr['total_acc'], bins=[-1, 20, 30, 40, 100],
    labels=['0-20','21-30','31-40','>40'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'total_acc_binned')
plot_by_woe(df_temp_coarse, 90)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['total_acc_binned'], inplace=True)
# DECISION: DROP.
original_results.append(collect_original(df_temp_fine,   'total_acc'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'total_acc', 'DROP'))
collect_summary('total_acc', 'CAPITAL', iv_fine, iv_coarse, 4, 'DROP',
                'IV = 0.0014 — below 0.02 threshold. Insufficient predictive power.')

# %%
# -----------------------------------------------------------------------------
# open_acc                 [CAPITAL — Current open credit lines]
# -----------------------------------------------------------------------------
# Number of open credit lines at time of application.
# WoE: Monotonically decreasing — more open accounts correlates with higher risk.
# Fine-classing: 50 bins.
# Binning: 5 bins (0-5 | 6-10 | 11-15 | 16-20 | >20).
# IV:  Weak (~0.008).
# DECISION: DROP.
# -----------------------------------------------------------------------------
df_temp_fine = woe_continuous(df_inputs_prepr, 'open_acc')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr['open_acc_binned'] = pd.cut(
    df_inputs_prepr['open_acc'], bins=[-1, 5, 10, 15, 20, 100],
    labels=['0-5','6-10','11-15','16-20','>20'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'open_acc_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['open_acc_binned'], inplace=True)
# DECISION: DROP.
original_results.append(collect_original(df_temp_fine,   'open_acc'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'open_acc', 'DROP'))
collect_summary('open_acc', 'CAPITAL', iv_fine, iv_coarse, 5, 'DROP',
                'IV below threshold after binning. Weak predictive power.')

# %%
# #############################################################################
# C4 — CONDITIONS
# #############################################################################

# -----------------------------------------------------------------------------
# int_rate                 [CONDITIONS — Risk-based loan pricing]
# -----------------------------------------------------------------------------
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
df_temp_fine = woe_continuous(df_inputs_prepr, 'int_rate_factor')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
df_inputs_prepr.drop(columns=['int_rate_factor'], inplace=True)

df_inputs_prepr['int_rate_binned'] = pd.cut(
    df_inputs_prepr['int_rate'],
    bins=[5.284, 9.419, 11.987, 13.014, 15.068, 17.123, 22.772,
          df_inputs_prepr['int_rate'].max()],
    labels=['5.284-9.419','9.419-11.987','11.987-13.014','13.014-15.068',
            '15.068-17.123','17.123-22.772','>22.772'])
df_temp_coarse = woe_continuous(df_inputs_prepr, 'int_rate_binned')
plot_by_woe(df_temp_coarse, 90)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['int_rate_binned'], inplace=True)
# DECISION: KEEP — 7 bins.
original_results.append(collect_original(df_temp_fine,   'int_rate'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'int_rate', 'KEEP'))
collect_summary('int_rate', 'CONDITIONS', iv_fine, iv_coarse, 7, 'KEEP',
                'Risk-based pricing. Strongest predictor — strong monotonic WoE.')

df_inputs_prepr['int_rate:5.284_9.419']   = np.where((df_inputs_prepr['int_rate'] > 5.284)  & (df_inputs_prepr['int_rate'] <= 9.419),  1, 0)
df_inputs_prepr['int_rate:9.419_11.987']  = np.where((df_inputs_prepr['int_rate'] > 9.419)  & (df_inputs_prepr['int_rate'] <= 11.987), 1, 0)
df_inputs_prepr['int_rate:11.987_13.014'] = np.where((df_inputs_prepr['int_rate'] > 11.987) & (df_inputs_prepr['int_rate'] <= 13.014), 1, 0)
df_inputs_prepr['int_rate:13.014_15.068'] = np.where((df_inputs_prepr['int_rate'] > 13.014) & (df_inputs_prepr['int_rate'] <= 15.068), 1, 0)
df_inputs_prepr['int_rate:15.068_17.123'] = np.where((df_inputs_prepr['int_rate'] > 15.068) & (df_inputs_prepr['int_rate'] <= 17.123), 1, 0)
df_inputs_prepr['int_rate:17.123_22.772'] = np.where((df_inputs_prepr['int_rate'] > 17.123) & (df_inputs_prepr['int_rate'] <= 22.772), 1, 0)
df_inputs_prepr['int_rate:>22.772']       = np.where( df_inputs_prepr['int_rate'] > 22.772,  1, 0)

# %%
# -----------------------------------------------------------------------------
# term                     [CONDITIONS — Loan repayment horizon]
# -----------------------------------------------------------------------------
df_temp = woe_continuous(df_inputs_prepr, 'term_int')
plot_by_woe(df_temp)
iv_term = df_temp['IV'].values[0]
# DECISION: KEEP — 2 categories.
original_results.append(collect_original(df_temp, 'term'))
rebinned_results.append(collect_rebinned(df_temp, 'term', 'KEEP'))
collect_summary('term', 'CONDITIONS', iv_term, iv_term, 2, 'KEEP',
                '36 vs 60 months. Longer term correlates with higher risk.')

df_inputs_prepr['term:36'] = np.where(df_inputs_prepr['term_int'] == 36, 1, 0)
df_inputs_prepr['term:60'] = np.where(df_inputs_prepr['term_int'] == 60, 1, 0)

# %%
# -----------------------------------------------------------------------------
# purpose                  [CONDITIONS — Use of loan proceeds]
# -----------------------------------------------------------------------------
df_temp_fine = woe_discrete(df_inputs_prepr, 'purpose')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
# %%
# Test coarse bins: 5 groups
purpose_map = {
    'small_business':   'small_business_renewable_energy_moving_house_medical',
    'renewable_energy': 'small_business_renewable_energy_moving_house_medical',
    'moving':           'small_business_renewable_energy_moving_house_medical',
    'house':            'small_business_renewable_energy_moving_house_medical',
    'medical':          'small_business_renewable_energy_moving_house_medical',
    'debt_consolidation': 'debt_consolidation',
    'educational':      'educational_other',
    'other':            'educational_other',
    'vacation':         'vacation_major_purchase_home_improvement',
    'major_purchase':   'vacation_major_purchase_home_improvement',
    'home_improvement': 'vacation_major_purchase_home_improvement',
    'credit_card':      'credit_card_car_wedding',
    'car':              'credit_card_car_wedding',
    'wedding':          'credit_card_car_wedding',
}

df_inputs_prepr['purpose_binned'] = df_inputs_prepr['purpose'].map(purpose_map)
df_temp_coarse = woe_discrete(df_inputs_prepr, 'purpose_binned')
iv_coarse = df_temp_coarse['IV'].values[0]
plot_by_woe(df_temp_coarse, 90)
df_inputs_prepr.drop(columns=['purpose_binned'], inplace=True)
# DECISION: KEEP — 5 groups.
original_results.append(collect_original(df_temp_fine,   'purpose'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'purpose', 'KEEP'))
collect_summary('purpose', 'CONDITIONS', iv_fine, iv_coarse, 5, 'KEEP',
                'Use of funds. Small business / medical highest risk; credit card lowest.')

df_inputs_prepr['purpose:small_business_renewable_energy_moving_house_medical'] = np.where(
    df_inputs_prepr['purpose'].isin(['small_business', 'renewable_energy', 'moving', 'house', 'medical']), 1, 0)
df_inputs_prepr['purpose:debt_consolidation'] = np.where(
    df_inputs_prepr['purpose'] == 'debt_consolidation', 1, 0)
df_inputs_prepr['purpose:educational_other'] = np.where(
    df_inputs_prepr['purpose'].isin(['educational', 'other']), 1, 0)
df_inputs_prepr['purpose:vacation_major_purchase_home_improvement'] = np.where(
    df_inputs_prepr['purpose'].isin(['vacation', 'major_purchase', 'home_improvement']), 1, 0)
df_inputs_prepr['purpose:credit_card_car_wedding'] = np.where(
    df_inputs_prepr['purpose'].isin(['credit_card', 'car', 'wedding']), 1, 0)

# %%
# -----------------------------------------------------------------------------
# initial_list_status      [CONDITIONS — Platform listing type]
# -----------------------------------------------------------------------------
df_temp = woe_discrete(df_inputs_prepr, 'initial_list_status')
plot_by_woe(df_temp)
iv_ils = df_temp['IV'].values[0]
# DECISION: DROP.
original_results.append(collect_original(df_temp, 'initial_list_status'))
rebinned_results.append(collect_rebinned(df_temp, 'initial_list_status', 'DROP'))
collect_summary('initial_list_status', 'CONDITIONS', iv_ils, iv_ils, 2, 'DROP',
                'Platform listing type. Low IV — insufficient predictive power.')

# %%
# -----------------------------------------------------------------------------
# addr_state               [CONDITIONS — Geographic & economic environment]
# -----------------------------------------------------------------------------
df_temp_fine = woe_discrete(df_inputs_prepr, 'addr_state')
iv_fine = df_temp_fine['IV'].values[0]
plot_by_woe(df_temp_fine, 90)
# %%
# Test coarse bins: 12 groups
addr_state_map = {
    'NE': 'NE_MS_AL_AR_OK', 'MS': 'NE_MS_AL_AR_OK', 'AL': 'NE_MS_AL_AR_OK',
    'AR': 'NE_MS_AL_AR_OK', 'OK': 'NE_MS_AL_AR_OK',
    'LA': 'LA_NV_TN_SD', 'NV': 'LA_NV_TN_SD', 'TN': 'LA_NV_TN_SD', 'SD': 'LA_NV_TN_SD',
    'NY': 'NY',
    'IN': 'IN_NM_MO', 'NM': 'IN_NM_MO', 'MO': 'IN_NM_MO',
    'FL': 'FL',
    'KY': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID', 'MD': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'ND': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID', 'IA': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'NJ': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID', 'OH': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'PA': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID', 'NC': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'ID': 'KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'HI': 'HI_MI_VA_DE_MN_AZ', 'MI': 'HI_MI_VA_DE_MN_AZ', 'VA': 'HI_MI_VA_DE_MN_AZ',
    'DE': 'HI_MI_VA_DE_MN_AZ', 'MN': 'HI_MI_VA_DE_MN_AZ', 'AZ': 'HI_MI_VA_DE_MN_AZ',
    'TX': 'TX_AK', 'AK': 'TX_AK',
    'CA': 'CA',
    'MA': 'MA_GA_RI_IL_UT', 'GA': 'MA_GA_RI_IL_UT', 'RI': 'MA_GA_RI_IL_UT',
    'IL': 'MA_GA_RI_IL_UT', 'UT': 'MA_GA_RI_IL_UT',
    'WI': 'WI_CT_MT_WY_SC_KS', 'CT': 'WI_CT_MT_WY_SC_KS', 'MT': 'WI_CT_MT_WY_SC_KS',
    'WY': 'WI_CT_MT_WY_SC_KS', 'SC': 'WI_CT_MT_WY_SC_KS', 'KS': 'WI_CT_MT_WY_SC_KS',
    'WV': 'WV_WA_CO_ME_OR_NH_VT_DC', 'WA': 'WV_WA_CO_ME_OR_NH_VT_DC',
    'CO': 'WV_WA_CO_ME_OR_NH_VT_DC', 'ME': 'WV_WA_CO_ME_OR_NH_VT_DC',
    'OR': 'WV_WA_CO_ME_OR_NH_VT_DC', 'NH': 'WV_WA_CO_ME_OR_NH_VT_DC',
    'VT': 'WV_WA_CO_ME_OR_NH_VT_DC', 'DC': 'WV_WA_CO_ME_OR_NH_VT_DC',
}

df_inputs_prepr['addr_state_binned'] = df_inputs_prepr['addr_state'].map(addr_state_map)
df_temp_coarse = woe_discrete(df_inputs_prepr, 'addr_state_binned')
plot_by_woe(df_temp_coarse, 90)
iv_coarse = df_temp_coarse['IV'].values[0]
df_inputs_prepr.drop(columns=['addr_state_binned'], inplace=True)
# DECISION: KEEP — 12 groups.
original_results.append(collect_original(df_temp_fine,   'addr_state'))
rebinned_results.append(collect_rebinned(df_temp_coarse, 'addr_state', 'KEEP'))
collect_summary('addr_state', 'CONDITIONS', iv_fine, iv_coarse, 12, 'KEEP',
                'Geographic environment. 50 states grouped into 12 WoE-homogeneous regions.')

df_inputs_prepr['addr_state:NE_MS_AL_AR_OK'] = np.where(
    df_inputs_prepr['addr_state'].isin(['NE', 'MS', 'AL', 'AR', 'OK']), 1, 0)
df_inputs_prepr['addr_state:LA_NV_TN_SD'] = np.where(
    df_inputs_prepr['addr_state'].isin(['LA', 'NV', 'TN', 'SD']), 1, 0)
df_inputs_prepr['addr_state:NY'] = np.where(
    df_inputs_prepr['addr_state'] == 'NY', 1, 0)
df_inputs_prepr['addr_state:IN_NM_MO'] = np.where(
    df_inputs_prepr['addr_state'].isin(['IN', 'NM', 'MO']), 1, 0)
df_inputs_prepr['addr_state:FL'] = np.where(
    df_inputs_prepr['addr_state'] == 'FL', 1, 0)
df_inputs_prepr['addr_state:KY_MD_ND_IA_NJ_OH_PA_NC_ID'] = np.where(
    df_inputs_prepr['addr_state'].isin(['KY', 'MD', 'ND', 'IA', 'NJ', 'OH', 'PA', 'NC', 'ID']), 1, 0)
df_inputs_prepr['addr_state:HI_MI_VA_DE_MN_AZ'] = np.where(
    df_inputs_prepr['addr_state'].isin(['HI', 'MI', 'VA', 'DE', 'MN', 'AZ']), 1, 0)
df_inputs_prepr['addr_state:TX_AK'] = np.where(
    df_inputs_prepr['addr_state'].isin(['TX', 'AK']), 1, 0)
df_inputs_prepr['addr_state:CA'] = np.where(
    df_inputs_prepr['addr_state'] == 'CA', 1, 0)
df_inputs_prepr['addr_state:MA_GA_RI_IL_UT'] = np.where(
    df_inputs_prepr['addr_state'].isin(['MA', 'GA', 'RI', 'IL', 'UT']), 1, 0)
df_inputs_prepr['addr_state:WI_CT_MT_WY_SC_KS'] = np.where(
    df_inputs_prepr['addr_state'].isin(['WI', 'CT', 'MT', 'WY', 'SC', 'KS']), 1, 0)
df_inputs_prepr['addr_state:WV_WA_CO_ME_OR_NH_VT_DC'] = np.where(
    df_inputs_prepr['addr_state'].isin(['WV', 'WA', 'CO', 'ME', 'OR', 'NH', 'VT', 'DC']), 1, 0)

# %%
# =============================================================================
# 11. FINAL COLUMN SELECTION
# =============================================================================
cols_to_keep = [
    # ID
    'id',
    # target
    'good_bad',
    # ── CHARACTER ──────────────────────────────────────────────────────────
    # Grade (7)
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    # Verification Status (3)
    'verification_status:Verified', 'verification_status:Source Verified',
    'verification_status:Not Verified',
    # Inquiries (4)
    'inq_last_6mths:0', 'inq_last_6mths:1', 'inq_last_6mths:2', 'inq_last_6mths:>2',
    # Length of credit history (3)
    'mths_since_earliest_cr_line:<250',
    'mths_since_earliest_cr_line:250-350',
    'mths_since_earliest_cr_line:>350',
    # ── CAPACITY ───────────────────────────────────────────────────────────
    # Annual Income (11)
    'annual_inc:<30K', 'annual_inc:30K-40K', 
    'annual_inc:40K-50K', 'annual_inc:50K-60K',
    'annual_inc:60K-70K', 'annual_inc:70K-80K',
    'annual_inc:80K-90K', 'annual_inc:90K-100K',
    'annual_inc:100K-120K', 'annual_inc:120K-150K',
    'annual_inc:>150K',
    # DTI (9)
    'dti:<8.6', 'dti:8.6-11', 'dti:11-12.6', 'dti:12.6-15', 'dti:15-18.2',
    'dti:18.2-22.2', 'dti:22.2-26.2', 'dti:26.2-30.2', 'dti:>=30.2',
    # Installment (6)
    'installment:<116.359', 'installment:116.359-218.708',
    'installment:218.708-321.058', 'installment:321.058-423.407',
    'installment:423.407-798.687', 'installment:>798.687',
    # ── CAPITAL ────────────────────────────────────────────────────────────
    # Total Revolving High Limit (6)
    'total_rev_hi_lim:<=20K', 'total_rev_hi_lim:20K-30K',
    'total_rev_hi_lim:30K-40K', 'total_rev_hi_lim:40K-55K',
    'total_rev_hi_lim:55K-95K', 'total_rev_hi_lim:>95K',
    # Home Ownership (3)
    'home_ownership:OWN', 'home_ownership:MORTGAGE', 'home_ownership:RENT_OTHER_NONE_ANY',
    # ── CONDITIONS ─────────────────────────────────────────────────────────
    # Interest Rate (7)
    'int_rate:5.284_9.419', 'int_rate:9.419_11.987', 'int_rate:11.987_13.014',
    'int_rate:13.014_15.068', 'int_rate:15.068_17.123', 'int_rate:17.123_22.772',
    'int_rate:>22.772',
    # Term (2)
    'term:36', 'term:60',
    # Purpose (5)
    'purpose:small_business_renewable_energy_moving_house_medical',
    'purpose:debt_consolidation',
    'purpose:educational_other',
    'purpose:vacation_major_purchase_home_improvement',
    'purpose:credit_card_car_wedding',
    # Address State (12)
    'addr_state:NE_MS_AL_AR_OK', 'addr_state:LA_NV_TN_SD',
    'addr_state:NY', 'addr_state:IN_NM_MO',
    'addr_state:FL', 'addr_state:KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'addr_state:HI_MI_VA_DE_MN_AZ', 'addr_state:TX_AK',
    'addr_state:CA', 'addr_state:MA_GA_RI_IL_UT',
    'addr_state:WI_CT_MT_WY_SC_KS', 'addr_state:WV_WA_CO_ME_OR_NH_VT_DC',
]

df_inputs_prepr = df_inputs_prepr[cols_to_keep]
print(f"Final shape:        {df_inputs_prepr.shape}")
print(f"Variables selected: {len(cols_to_keep) - 1} dummies + 1 ID")

# %%
# =============================================================================
# 12. SAVE PREPROCESSED DATA
# =============================================================================
loan_data_train = df_inputs_prepr
# %%
# ── Save variable summary for Streamlit (Page 1) ─────────────────────────────
# Page 1 consumes only variable_summary.csv. The granular woe_original / woe_rebinned
# tables are diagnostics used during binning and are not read by the app, so they
# are not written to disk.

# Variable summary — one row per variable
df_variable_summary = pd.DataFrame(summary_results)
df_variable_summary = df_variable_summary.sort_values(
    ['decision', 'iv_rebinned'], ascending=[True, False]).reset_index(drop=True)
df_variable_summary.to_csv('ifrs9_app/data/variable_summary.csv', index=False)
print(f"variable_summary.csv — {len(df_variable_summary)} variables")
print(df_variable_summary[['variable', 'category', 'iv_original',
                            'iv_rebinned', 'n_bins', 'decision']].to_string())
# %%
