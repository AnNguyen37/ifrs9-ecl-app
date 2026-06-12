# %% =========================================================================
# 1. LOAD DATA
# =============================================================================
%reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

inputs_train_with_ref_cat = pd.read_csv('inputs_train_with_ref_cat.csv', index_col=0)
df_scorecard = pd.read_csv('df_scorecard.csv')
# %%
# New monitoring data — 2019
loan_data = pd.read_csv('LoanStats_2019Q1.csv', skiprows=1, low_memory=False).copy()
print(loan_data.shape)
loan_data.head()

# %% =========================================================================
# 2. GENERAL PREPROCESSING  (mirrors your main preprocessing notebook)
# =============================================================================

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
loan_data['mths_since_issue_d'] = round(
    pd.to_numeric((pd.to_datetime('2019-01-01') - loan_data['issue_d_date'])
    / np.timedelta64(1, 'D')) / 30.5)
loan_data.drop(columns=['issue_d', 'issue_d_date'], inplace=True)

# Convert int_rate from string "10.5%" → float 10.5
loan_data['int_rate'] = pd.to_numeric(
    loan_data['int_rate'].astype(str).str.replace('%', '').str.strip(),
    errors='coerce'
)
# %% =========================================================================
# 3. DUMMY VARIABLES
# =============================================================================
loan_data_dummies = pd.concat([
    pd.get_dummies(loan_data['grade'],               prefix='grade',               prefix_sep=':'),
    pd.get_dummies(loan_data['home_ownership'],      prefix='home_ownership',      prefix_sep=':'),
    pd.get_dummies(loan_data['verification_status'], prefix='verification_status', prefix_sep=':'),
    pd.get_dummies(loan_data['purpose'],             prefix='purpose',             prefix_sep=':'),
    pd.get_dummies(loan_data['addr_state'],          prefix='addr_state',          prefix_sep=':'),
    pd.get_dummies(loan_data['initial_list_status'], prefix='initial_list_status', prefix_sep=':'),
], axis=1)
loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)

# %% =========================================================================
# 4. MISSING VALUE TREATMENT
# =============================================================================
loan_data['total_rev_hi_lim']            = loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'])
loan_data['mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].fillna(0)
loan_data['inq_last_6mths']              = loan_data['inq_last_6mths'].fillna(0)
loan_data['delinq_2yrs']                 = loan_data['delinq_2yrs'].fillna(0)
loan_data['annual_inc']                  = loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean())

# %% =========================================================================
# 5. GOOD / BAD
# =============================================================================
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

# %%
# -----------------------------------------------------------------------------
loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin([
        'Charged Off',
        'Does not meet the credit policy. Status:Charged Off',
    ]), 0, 1)

good_bad_pct = loan_data['good_bad'].value_counts(normalize=True) * 100
print(f"\nGood (1): {good_bad_pct[1]:.1f}%  |  Bad (0): {good_bad_pct[0]:.1f}%")

# %% =========================================================================
# 6. SPLIT INTO INPUTS / TARGETS
# =============================================================================
df_inputs_prepr  = loan_data.drop('good_bad', axis=1)
df_targets_prepr = loan_data['good_bad']

# %% =========================================================================
# 7. COARSE CLASSING — ALL VARIABLES  (exact bins from your training notebook)
# =============================================================================

# #############################################################################
# C1 — CHARACTER
# #############################################################################

# -----------------------------------------------------------------------------
# grade                    [CHARACTER — Primary LC risk indicator]
# -----------------------------------------------------------------------------

# %%
# -----------------------------------------------------------------------------
# inq_last_6mths           [CHARACTER — Recent credit-seeking behaviour]
# -----------------------------------------------------------------------------
df_inputs_prepr['inq_last_6mths:0']  = np.where( df_inputs_prepr['inq_last_6mths'] == 0, 1, 0)
df_inputs_prepr['inq_last_6mths:1']  = np.where((df_inputs_prepr['inq_last_6mths'] > 0) & (df_inputs_prepr['inq_last_6mths'] <= 1), 1, 0)
df_inputs_prepr['inq_last_6mths:2']  = np.where((df_inputs_prepr['inq_last_6mths'] > 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:>2'] = np.where( df_inputs_prepr['inq_last_6mths'] > 2,  1, 0)

# %%
# -----------------------------------------------------------------------------
# verification_status      [CHARACTER — Income credibility]
# -----------------------------------------------------------------------------

# %%
# -----------------------------------------------------------------------------
# mths_since_earliest_cr_line_factor      [CHARACTER — Length of credit history]
# -----------------------------------------------------------------------------
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
# installment              [CAPACITY — Monthly payment obligation]
# -----------------------------------------------------------------------------
# Fixed monthly payment amount.
# WoE: Monotonically decreasing — higher payments correlate with higher risk.
# Fine-classing: 50 bins.
# Binning: 9 bins with clear WoE boundaries.
# IV:  Medium (~0.10).
# DECISION: KEEP — 9 bins.
# -----------------------------------------------------------------------------
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
df_inputs_prepr['home_ownership:OTHER'] = 0
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([
    df_inputs_prepr['home_ownership:RENT'],  df_inputs_prepr['home_ownership:OTHER'],
    df_inputs_prepr['home_ownership:NONE'],  df_inputs_prepr['home_ownership:ANY']
])

# %%
# -----------------------------------------------------------------------------
# total_acc                [CAPITAL — Total credit lines ever opened]
# -----------------------------------------------------------------------------
# Total number of credit lines in the borrower's credit file.
# WoE: Weak and non-monotonic — no consistent direction of effect.
# IV:  Weak (<0.02).
# DECISION: DROP — insufficient predictive power.
# -----------------------------------------------------------------------------

# %%
# -----------------------------------------------------------------------------
# open_acc                 [CAPITAL — Current open credit lines]
# -----------------------------------------------------------------------------
# Number of open credit lines at time of application.
# WoE: Weak signal; partially captured by total_rev_hi_lim and total_acc.
# IV:  Weak (<0.02).
# DECISION: DROP — insufficient predictive power; information overlaps
#           with total_rev_hi_lim.
# -----------------------------------------------------------------------------

# %%
# #############################################################################
# C4 — CONDITIONS
# #############################################################################

# -----------------------------------------------------------------------------
# int_rate                 [CONDITIONS — Risk-based loan pricing]
# -----------------------------------------------------------------------------
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
df_inputs_prepr['term:36'] = np.where(df_inputs_prepr['term_int'] == 36, 1, 0)
df_inputs_prepr['term:60'] = np.where(df_inputs_prepr['term_int'] == 60, 1, 0)

# %%
# -----------------------------------------------------------------------------
# purpose                  [CONDITIONS — Use of loan proceeds]
# -----------------------------------------------------------------------------
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
# DECISION: KEEP — 2 categories, clear WoE separation, no grouping needed.

# %%
# -----------------------------------------------------------------------------
# addr_state               [CONDITIONS — Geographic & economic environment]
# -----------------------------------------------------------------------------
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


# %% =========================================================================
# 8. ASSIGN MONITORING DATASET
# =============================================================================
loan_data_inputs_2019Q1  = df_inputs_prepr
loan_data_targets_2019Q1 = df_targets_prepr

print(f"Q1 2019 shape: {loan_data_inputs_2019Q1.shape}")
print(f"Training shape: {inputs_train_with_ref_cat.shape}")

# %% =========================================================================
# 9. SELECT COLUMNS MATCHING THE SCORECARD  (same order as inputs_train_with_ref_cat)
# =============================================================================
scorecard_cols = [
    # ID
    'id',
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

inputs_2019Q1_with_ref_cat = loan_data_inputs_2019Q1.loc[:, scorecard_cols]

print(f"Training columns : {inputs_train_with_ref_cat.shape[1]}")
print(f"Q1 2019 columns  : {inputs_2019Q1_with_ref_cat.shape[1]}")

# %% =========================================================================
# 10. COMPUTE SCORECARD SCORES
# =============================================================================
scorecard_scores = df_scorecard['Score - Final'].values.reshape(-1, 1)

# %% 
print(df_scorecard.columns.tolist())
# %% 
# Training scores
inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat.copy()
inputs_train_with_ref_cat_w_intercept.insert(0, 'const', 1)
inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values]
y_scores_train = inputs_train_with_ref_cat_w_intercept.astype(float).dot(scorecard_scores)

# Q1 2019 scores
inputs_2019Q1_with_ref_cat_w_intercept = inputs_2019Q1_with_ref_cat.copy()
inputs_2019Q1_with_ref_cat_w_intercept.insert(0, 'const', 1)
inputs_2019Q1_with_ref_cat_w_intercept = inputs_2019Q1_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values]
y_scores_2019Q1 = inputs_2019Q1_with_ref_cat_w_intercept.astype(float).dot(scorecard_scores)

# %% 
# Append Score column
inputs_train_with_ref_cat_w_intercept['Score'] = y_scores_train.values.flatten()
inputs_2019Q1_with_ref_cat_w_intercept['Score'] = y_scores_2019Q1.values.flatten()

# %% 
print("Score distribution — Q1 2019:")
print(inputs_2019Q1_with_ref_cat_w_intercept['Score'].describe().round(1))

# %% =========================================================================
# 11. SCORE BAND DUMMIES
# =============================================================================
score_bands = [(300,350),(350,400),(400,450),(450,500),(500,550),
               (550,600),(600,650),(650,700),(700,750),(750,800),(800,851)]

for df in [inputs_train_with_ref_cat_w_intercept, inputs_2019Q1_with_ref_cat_w_intercept]:
    for lo, hi in score_bands:
        label = f'Score:{lo}-{min(hi, 850)}'
        df[label] = np.where((df['Score'] >= lo) & (df['Score'] < hi), 1, 0)

# %% =========================================================================
# 12. PSI CALCULATION
# =============================================================================
PSI_calc_train  = inputs_train_with_ref_cat_w_intercept.sum()  / inputs_train_with_ref_cat_w_intercept.shape[0]
PSI_calc_2019Q1 = inputs_2019Q1_with_ref_cat_w_intercept.sum() / inputs_2019Q1_with_ref_cat_w_intercept.shape[0]

PSI_calc = pd.concat([PSI_calc_train, PSI_calc_2019Q1], axis=1).reset_index()
PSI_calc.columns = ['index', 'Proportions_Train', 'Proportions_2019Q1']
PSI_calc['Original feature name'] = PSI_calc['index'].str.split(':').str[0]
PSI_calc = PSI_calc[['index', 'Original feature name', 'Proportions_Train', 'Proportions_2019Q1']]

# Remove non-feature rows
PSI_calc = PSI_calc[~PSI_calc['index'].isin(['const', 'Score'])]

# PSI contribution per bin
PSI_calc['Contribution'] = np.where(
    (PSI_calc['Proportions_Train'] == 0) | (PSI_calc['Proportions_2019Q1'] == 0), 0,
    (PSI_calc['Proportions_2019Q1'] - PSI_calc['Proportions_Train']) *
     np.log(PSI_calc['Proportions_2019Q1'] / PSI_calc['Proportions_Train'])
)

# %% =========================================================================
# 13. PSI SUMMARY PER VARIABLE
# =============================================================================
PSI_summary = (PSI_calc
               .groupby('Original feature name')['Contribution']
               .sum()
               .reset_index()
               .rename(columns={'Contribution': 'PSI'}))

PSI_summary['Stability'] = pd.cut(
    PSI_summary['PSI'],
    bins   = [-np.inf, 0.1, 0.25, np.inf],
    labels = ['Stable (<0.10)', 'Minor shift (0.10–0.25)', 'Major shift (>0.25)'])

PSI_summary = PSI_summary.sort_values('PSI', ascending=False).reset_index(drop=True)

print("\n=== PSI SUMMARY — Q1 2019 vs Training ===")
pd.options.display.max_rows = None
print(PSI_summary.to_string(index=False))
pd.options.display.max_rows = 100

# %% =========================================================================
# 14. SCORE PSI (population-level stability)
# =============================================================================
score_psi = PSI_calc[PSI_calc['Original feature name'] == 'Score']['Contribution'].sum()
print(f"\nScore PSI: {score_psi:.4f}  →  ", end='')
print('Stable' if score_psi < 0.1 else ('Minor shift' if score_psi < 0.25 else 'MAJOR SHIFT — recalibration recommended'))

import pickle, os
from sklearn.metrics import roc_auc_score
os.makedirs('data', exist_ok=True)

# %% =========================================================================
# 15. SAVE — File 1: Score PSI breakdown (bin-level)
# =============================================================================
score_values_train  = y_scores_train.values.flatten()
score_values_2019Q1 = y_scores_2019Q1.values.flatten()

score_bins_arr = np.arange(300, 876, 25)
hist_train, edges = np.histogram(score_values_train,  bins=score_bins_arr)
hist_2019,  _    = np.histogram(score_values_2019Q1, bins=score_bins_arr)

train_pct = hist_train / hist_train.sum()
test_pct  = hist_2019  / hist_2019.sum()
train_safe = np.where(train_pct == 0, 0.0001, train_pct)
test_safe  = np.where(test_pct  == 0, 0.0001, test_pct)

score_psi_breakdown = pd.DataFrame({
    'bin_lower':        edges[:-1],
    'bin_upper':        edges[1:],
    'train_pct':        train_pct,
    'monitoring_pct':   test_pct,
    'difference':       test_pct - train_pct,
    'psi_contribution': (test_safe - train_safe) * np.log(test_safe / train_safe),
})
score_psi_breakdown.to_csv('ifrs9_app/data/score_psi_breakdown.csv', index=False)
print(f"✓ score_psi_breakdown.csv — {len(score_psi_breakdown)} bins")

# Score histogram for visualisation
score_histogram = pd.DataFrame({
    'bin_center':            (edges[:-1] + edges[1:]) / 2,
    'train_proportion':      train_pct,
    'monitoring_proportion': test_pct,
    'train_count':           hist_train,
    'monitoring_count':      hist_2019,
})
score_histogram.to_csv('ifrs9_app/data/score_distribution_comparison.csv', index=False)
print(f"✓ score_distribution_comparison.csv saved")

# %% =========================================================================
# 16. SAVE — File 2: Variable-level PSI
# =============================================================================
PSI_summary.to_csv('ifrs9_app/data/psi_summary.csv', index=False)
PSI_calc.to_csv('ifrs9_app/data/psi_bin_breakdown.csv', index=False)
print(f"✓ psi_summary.csv — {len(PSI_summary)} variables")

# %% =========================================================================
# 17. SAVE — File 3: Vintage Gini (2019 Q1)
# =============================================================================
min_score    = 300
max_score    = 850
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

def scores_to_pd(scores):
    sum_coef = (scores - min_score) / (max_score - min_score) * (max_sum_coef - min_sum_coef) + min_sum_coef
    return 1 - np.exp(sum_coef) / (1 + np.exp(sum_coef))   # PD = 1 - P(good)

pd_2019Q1      = scores_to_pd(score_values_2019Q1)
y_actual       = loan_data_targets_2019Q1.values            # 1=good, 0=bad
p_good_2019Q1  = 1 - pd_2019Q1

auc   = roc_auc_score(y_actual, p_good_2019Q1)
gini  = 2 * auc - 1

print(f"Gini 2019Q1: {gini:.4f}")

# %% =========================================================================
# 19. SAVE — Monitoring summary
# =============================================================================
monitoring_summary = {
    'reporting_period':      '2019 Q1',
    'training_period':       '2007-2018',
    'n_train':               len(inputs_train_with_ref_cat_w_intercept),
    'n_monitoring':          len(inputs_2019Q1_with_ref_cat_w_intercept),
    'score_psi':             float(score_psi),
    'max_variable_psi':      float(PSI_summary['PSI'].max()),
    'n_stable':              int((PSI_summary['PSI'] < 0.10).sum()),
    'n_minor':               int(((PSI_summary['PSI'] >= 0.10) & (PSI_summary['PSI'] < 0.25)).sum()),
    'n_major':               int((PSI_summary['PSI'] >= 0.25).sum()),
    'gini_2019Q1':           float(gini),
    'mean_predicted_pd':     float(pd_2019Q1.mean()),
    'observed_default_rate': float(1 - y_actual.mean()),
    'reporting_date':        '2026-05-22',
}
pickle.dump(monitoring_summary, open('ifrs9_app/data/monitoring_summary.pkl', 'wb'))
print(f"✓ monitoring_summary.pkl saved")

print('\n=== All monitoring files saved ===')
print('  data/score_psi_breakdown.csv           — bin-level score PSI')
print('  data/score_distribution_comparison.csv — score histogram train vs 2019Q1')
print('  data/psi_summary.csv                   — variable-level PSI summary')
print('  data/psi_bin_breakdown.csv             — bin-level PSI for all variables')
print('  data/monitoring_summary.pkl            — summary statistics')
# %%
