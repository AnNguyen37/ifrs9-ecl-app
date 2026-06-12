# %%
%reset -f

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

sns.set()

# %%
# =============================================================================
# 1. LOADING THE DATA
# =============================================================================
loan_data_train = pd.read_csv('loan_data_train.csv', index_col=0, low_memory=False)
loan_data_test  = pd.read_csv('loan_data_test.csv',  index_col=0, low_memory=False)

# Set id as index if not already
if loan_data_train.index.name != 'id' and 'id' in loan_data_train.columns:
    loan_data_train = loan_data_train.set_index('id')
    loan_data_test  = loan_data_test.set_index('id')

print(loan_data_train.index[:5])  # should show loan ids

# %%
# Explore Data
loan_data_train.head()

# %%
# =============================================================================
# 2. SELECTING THE FEATURES
# =============================================================================
feature_cols = [
    # ── CHARACTER ──────────────────────────────────────────────────────────
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'verification_status:Verified', 'verification_status:Source Verified',
    'verification_status:Not Verified',
    'inq_last_6mths:0', 'inq_last_6mths:1', 'inq_last_6mths:2', 'inq_last_6mths:>2',
    'mths_since_earliest_cr_line:<250',
    'mths_since_earliest_cr_line:250-350',
    'mths_since_earliest_cr_line:>350',
    # ── CAPACITY ───────────────────────────────────────────────────────────
    'annual_inc:<30K', 'annual_inc:30K-40K',
    'annual_inc:40K-50K', 'annual_inc:50K-60K',
    'annual_inc:60K-70K', 'annual_inc:70K-80K',
    'annual_inc:80K-90K', 'annual_inc:90K-100K',
    'annual_inc:100K-120K', 'annual_inc:120K-150K',
    'annual_inc:>150K',
    'dti:<8.6', 'dti:8.6-11', 'dti:11-12.6', 'dti:12.6-15', 'dti:15-18.2',
    'dti:18.2-22.2', 'dti:22.2-26.2', 'dti:26.2-30.2', 'dti:>=30.2',
    'installment:<116.359', 'installment:116.359-218.708',
    'installment:218.708-321.058', 'installment:321.058-423.407',
    'installment:423.407-798.687', 'installment:>798.687',
    # ── CAPITAL ────────────────────────────────────────────────────────────
    'total_rev_hi_lim:<=20K', 'total_rev_hi_lim:20K-30K',
    'total_rev_hi_lim:30K-40K', 'total_rev_hi_lim:40K-55K',
    'total_rev_hi_lim:55K-95K', 'total_rev_hi_lim:>95K',
    'home_ownership:OWN', 'home_ownership:MORTGAGE', 'home_ownership:RENT_OTHER_NONE_ANY',
    # ── CONDITIONS ─────────────────────────────────────────────────────────
    'int_rate:5.284_9.419', 'int_rate:9.419_11.987', 'int_rate:11.987_13.014',
    'int_rate:13.014_15.068', 'int_rate:15.068_17.123', 'int_rate:17.123_22.772',
    'int_rate:>22.772',
    'term:36', 'term:60',
    'purpose:small_business_renewable_energy_moving_house_medical',
    'purpose:debt_consolidation',
    'purpose:educational_other',
    'purpose:vacation_major_purchase_home_improvement',
    'purpose:credit_card_car_wedding',
    'addr_state:NE_MS_AL_AR_OK', 'addr_state:LA_NV_TN_SD',
    'addr_state:NY', 'addr_state:IN_NM_MO',
    'addr_state:FL', 'addr_state:KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'addr_state:HI_MI_VA_DE_MN_AZ', 'addr_state:TX_AK',
    'addr_state:CA', 'addr_state:MA_GA_RI_IL_UT',
    'addr_state:WI_CT_MT_WY_SC_KS', 'addr_state:WV_WA_CO_ME_OR_NH_VT_DC',
]

ref_categories = [
    'grade:G',
    'verification_status:Verified',
    'inq_last_6mths:>2',
    'mths_since_earliest_cr_line:<250',
    'annual_inc:<30K',
    'dti:>=30.2',
    'installment:>798.687',
    'total_rev_hi_lim:<=20K',
    'home_ownership:RENT_OTHER_NONE_ANY',
    'int_rate:>22.772',
    'term:60',
    'purpose:small_business_renewable_energy_moving_house_medical',
    'addr_state:NE_MS_AL_AR_OK',
]

# %%
inputs_train_with_ref_cat = loan_data_train.loc[:, feature_cols]
inputs_test_with_ref_cat  = loan_data_test.loc[:,  feature_cols]

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis=1).astype(float)
inputs_test  = inputs_test_with_ref_cat.drop(ref_categories,  axis=1).astype(float)

# %%
# =============================================================================
# 3. PD MODEL ESTIMATION — LOGISTIC REGRESSION (statsmodels)
# =============================================================================
X_train = sm.add_constant(inputs_train)
X_test  = sm.add_constant(inputs_test)

# %%
logit_model = sm.Logit(
    loan_data_train['good_bad'].astype(int).values,
    X_train.values
)
result = logit_model.fit()
print(result.summary())

# %%
# =============================================================================
# 4. SUMMARY TABLE — COEFFICIENTS & P-VALUES
# =============================================================================
summary_table = pd.DataFrame({
    'Feature name': X_train.columns,
    'Coefficients': result.params,
    'p_values':     result.pvalues.round(4)
})
summary_table

# %%
# =============================================================================
# 5. PREDICTED PROBABILITIES — keeping loan id as index
# =============================================================================
pd_train = pd.Series(result.predict(X_train.values),
                     index=loan_data_train.index,
                     name='PD')

pd_test  = pd.Series(result.predict(X_test.values),
                     index=loan_data_test.index,
                     name='PD')

print(pd_train.head())
print(pd_test.head())

# %%
# =============================================================================
# 6. CREATING THE SCORECARD
# =============================================================================
# Step 1 — Build reference category rows (coeff = 0, p_value = NaN)
df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
df_ref_categories['Coefficients'] = 0
df_ref_categories['p_values']     = np.nan
df_ref_categories

# %%
# Step 2 — Concatenate model rows + reference rows
df_scorecard = pd.concat([summary_table, df_ref_categories])
df_scorecard = df_scorecard.reset_index(drop=True)
df_scorecard

# %%
# Step 3 — Extract original feature name (part before the colon)
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
df_scorecard

# %%
# Step 4 — Score scaling parameters
min_score = 300
max_score = 850

# %%
df_scorecard.groupby('Original feature name')['Coefficients'].min()

# %%
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
min_sum_coef

# %%
df_scorecard.groupby('Original feature name')['Coefficients'].max()

# %%
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
max_sum_coef

# %%
# Step 5 — Scale coefficients to score range [300, 850]
# Intercept row (index 0) uses a different formula to anchor the scale.
df_scorecard['Score - Calculation'] = (
    df_scorecard['Coefficients'] *
    (max_score - min_score) / (max_sum_coef - min_sum_coef)
)
df_scorecard

# %%
df_scorecard.iloc[0, df_scorecard.columns.get_loc('Score - Calculation')] = (
    (df_scorecard['Coefficients'].iloc[0] - min_sum_coef) /
    (max_sum_coef - min_sum_coef) *
    (max_score - min_score) + min_score
)
df_scorecard

# %%
# Step 6 — Round to preliminary integer scores
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
df_scorecard

# %%
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
min_sum_score_prel

# %%
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
max_sum_score_prel

# %% Score min and max within the range. Rename the score premilinary to score final
df_scorecard = df_scorecard.rename(columns={'Score - Preliminary': 'Score - Final'})

# %%
# Step 7 — Compute rounding difference to identify which bin needs a ±1 adjustment
# to make max_sum_score_prel exactly equal to max_score (850).
#df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
#df_scorecard

# %%
# ── Step 8: Find which bin is the minimum for each variable ───────
#min_bins = df_scorecard.loc[
#    df_scorecard.groupby('Original feature name')['Score - Preliminary'].idxmin(),
#    ['Feature name', 'Original feature name',
#     'Score - Calculation', 'Score - Preliminary', 'Difference']
#].sort_values('Difference')

#min_bins
# %%
# ── Step 9: Add 1 to the min bin with the largest negative Difference
# (the one that was rounded down the most — most deserving of +1)
#adjust_idx = min_bins['Difference'].idxmin()
#print(f"Adjusting: {df_scorecard.loc[adjust_idx, 'Feature name']}")

#df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary'].copy()
#df_scorecard.iloc[
#    df_scorecard.index.get_loc(adjust_idx),
#    df_scorecard.columns.get_loc('Score - Final')
#] = df_scorecard.loc[adjust_idx, 'Score - Preliminary'] + 1

# %%
# Step 10: Verify ─────────────────────────────────────────────────
min_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
max_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
print(f"Min score: {min_sum_score_final}  (target: {min_score})")
print(f"Max score: {max_sum_score_final}  (target: {max_score})")

# %%
# =============================================================================
# 7. CALCULATING CREDIT SCORES FOR THE TEST SET
# =============================================================================
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat.copy()
inputs_test_with_ref_cat_w_intercept.insert(0, 'const', 1)
inputs_test_with_ref_cat_w_intercept.head()

# %%
# Keep only columns that appear in the scorecard (in the same order)
# Change 'Intercept' → 'const' to match statsmodels naming
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values
]
inputs_test_with_ref_cat_w_intercept.head()

# %%
scorecard_scores = df_scorecard['Score - Final'].values.reshape(-1, 1)
# reshape(-1, 1) is dynamic — no hardcoded row count needed.
scorecard_scores

# %%
print(inputs_test_with_ref_cat_w_intercept.shape)
print(scorecard_scores.shape)

# %%
y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
# Matrix multiplication: each row of inputs × score per bin = total credit score.
y_scores.head()

# %%
y_scores.tail()

# %%
# =============================================================================
# 8. FROM CREDIT SCORE TO PD
# =============================================================================
sum_coef_from_score = (
    (y_scores.astype(float) - min_score) / (max_score - min_score) *
    (max_sum_coef - min_sum_coef) + min_sum_coef
)

y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
y_hat_proba_from_score.head()

# %%
# Sanity check: PD from score should match PD from logit (first 5 rows)
print("From logit:  ", y_hat_test_proba[:5].round(6))
print("From score:  ", y_hat_proba_from_score.values.flatten()[:5].round(6))

# %%
df_scorecard.to_csv('ifrs9_app/data/df_scorecard.csv')

# %%
