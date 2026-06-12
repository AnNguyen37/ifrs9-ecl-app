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

if loan_data_train.index.name != 'id' and 'id' in loan_data_train.columns:
    loan_data_train = loan_data_train.set_index('id')
    loan_data_test  = loan_data_test.set_index('id')

print(loan_data_train.index[:5])

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

# ── Model fit statistics ───────────────────────────────────────────────────
fit_stats = pd.DataFrame({
    'Statistic': [
        'Log-Likelihood',
        'LL-Null',
        'LLR p-value',
        'Pseudo R-squared (McFadden)',
        'AIC',
        'BIC',
        'N observations'
    ],
    'Value': [
        round(result.llf, 4),
        round(result.llnull, 4),
        round(result.llr_pvalue, 6),
        round(result.prsquared, 4),
        round(result.aic, 4),
        round(result.bic, 4),
        int(result.nobs)
    ]
})

print(summary_table)
print()
print(fit_stats.to_string(index=False))

# %%
# ── Save ───────────────────────────────────────────────────────────────────
summary_table.to_csv('ifrs9_app/data/pd_summary_table.csv', index=False)
fit_stats.to_csv('ifrs9_app/data/pd_fit_stats.csv', index=False)
print("\n✓ pd_summary_table.csv and pd_fit_stats.csv saved")
# %%
# =============================================================================
# 5. PREDICTED PROBABILITIES
# =============================================================================
pd_train = pd.Series(result.predict(X_train.values),
                     index=loan_data_train.index, name='PD_life')
pd_test  = pd.Series(result.predict(X_test.values),
                     index=loan_data_test.index,  name='PD_life')

# %%
# =============================================================================
# 6. COMBINE TRAIN & TEST — LIFETIME PD
# =============================================================================
pd_life = pd.concat([pd_train, pd_test]).rename('PD_life').sort_index()
print(f"Total loans : {len(pd_life):,}")

# %%
# =============================================================================
# 7. PREDICTIONS & ACTUALS DATAFRAME
# =============================================================================
threshold = 0.8

df_predictions = pd.concat([
    pd.DataFrame({
        'y_actual':    loan_data_train['good_bad'].astype(int),
        'y_pred_proba': pd_train,
        'y_pred':      (pd_train > threshold).astype(int),
        'split':       'train'
    }),
    pd.DataFrame({
        'y_actual':    loan_data_test['good_bad'].astype(int),
        'y_pred_proba': pd_test,
        'y_pred':      (pd_test > threshold).astype(int),
        'split':       'test'
    })
]).sort_index()

df_predictions.index.name = 'id'

# %%
# =============================================================================
# 8. ROC CURVE & AUROC
# =============================================================================
y_actual_test = loan_data_test['good_bad'].astype(int).values
y_proba_test  = pd_test.values

fpr, tpr, thresholds = roc_curve(y_actual_test, y_proba_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='PD Model')
plt.plot(fpr, fpr, linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

AUROC_test = roc_auc_score(y_actual_test, y_proba_test)
print(f"AUROC (test): {AUROC_test:.4f}")

# %%
# =============================================================================
# 9. GINI & KS
# =============================================================================
df_ks = (pd.DataFrame({'y_actual': y_actual_test, 'y_pred_proba': y_proba_test})
           .sort_values('y_pred_proba')
           .reset_index(drop=True))

n_total = len(df_ks)
n_good  = df_ks['y_actual'].sum()
n_bad   = n_total - n_good

df_ks['cum_pop']  = (df_ks.index + 1) / n_total
df_ks['cum_good'] = df_ks['y_actual'].cumsum() / n_good
df_ks['cum_bad']  = (df_ks.index + 1 - df_ks['y_actual'].cumsum()) / n_bad

# ── Gini ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(df_ks['cum_pop'], df_ks['cum_bad'], label='PD Model')
plt.plot(df_ks['cum_pop'], df_ks['cum_pop'], linestyle='--', color='k', label='Random')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini (Lorenz Curve)')
plt.legend()
plt.tight_layout()
plt.show()

# ── KS ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(df_ks['y_pred_proba'], df_ks['cum_bad'],  color='r', label='Bad')
plt.plot(df_ks['y_pred_proba'], df_ks['cum_good'], color='b', label='Good')
plt.xlabel('Estimated Probability of Being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
plt.legend()
plt.tight_layout()
plt.show()

Gini_test = AUROC_test * 2 - 1
KS_test   = (df_ks['cum_bad'] - df_ks['cum_good']).max()
print(f"Gini (test): {Gini_test:.4f}")
print(f"KS   (test): {KS_test:.4f}")

# %%
# =============================================================================
# 10. CONFUSION MATRIX & ACCURACY
# =============================================================================
conf_matrix = pd.crosstab(
    df_predictions.loc[df_predictions['split'] == 'test', 'y_actual'],
    df_predictions.loc[df_predictions['split'] == 'test', 'y_pred'],
    rownames=['Actual'], colnames=['Predicted'])
print(conf_matrix)

conf_matrix_pct = conf_matrix / conf_matrix.values.sum()
print(conf_matrix_pct.round(4))

accuracy_test = conf_matrix_pct.iloc[0, 0] + conf_matrix_pct.iloc[1, 1]
print(f"\nAccuracy (test): {accuracy_test:.4f}  (threshold = {threshold})")

# %%
# =============================================================================
# 11. TRAIN vs TEST GINI COMPARISON
# =============================================================================
AUROC_train = roc_auc_score(
    loan_data_train['good_bad'].astype(int).values,
    pd_train.values)
Gini_train = AUROC_train * 2 - 1

print("=" * 40)
print(f"  Gini — Train : {Gini_train:.4f}")
print(f"  Gini — Test  : {Gini_test:.4f}")
print(f"  Difference   : {Gini_train - Gini_test:.4f}")
print("=" * 40)

# %%
# SAVE PERFORMANCE METRICS & ROC CURVE DATA

# ── Train metrics ─────────────────────────────────────────────────────────
y_actual_train = loan_data_train['good_bad'].astype(int).values
y_proba_train  = pd_train.values

# ── Performance metrics ───────────────────────────────────────────────────
pd_metrics = {
    'AUROC_train'    : round(AUROC_train, 4),
    'AUROC_test'     : round(AUROC_test,  4),
    'Gini_train'     : round(Gini_train,  4),
    'Gini_test'      : round(Gini_test,   4),
    'KS_test'        : round(KS_test,     4),
    'accuracy_test'  : round(accuracy_test, 4),
    'threshold'      : threshold,
    'gini_gap'       : round(abs(Gini_train - Gini_test), 4),
    'n_train'        : int(len(y_actual_train)),
    'n_test'         : int(len(y_actual_test)),
    'bad_rate_train' : round(1 - y_actual_train.mean(), 4),
    'bad_rate_test'  : round(1 - y_actual_test.mean(),  4),
}

# ── ROC curve data ────────────────────────────────────────────────────────
fpr_train, tpr_train, _ = roc_curve(y_actual_train, y_proba_train)
fpr_test,  tpr_test,  _ = roc_curve(y_actual_test,  y_proba_test)

roc_train = pd.DataFrame({'fpr': fpr_train, 'tpr': tpr_train, 'split': 'train'})
roc_test  = pd.DataFrame({'fpr': fpr_test,  'tpr': tpr_test,  'split': 'test'})
df_roc    = pd.concat([roc_train, roc_test], ignore_index=True)

# ── Downsample for Streamlit ───────────────────────────────────────────────
roc_full  = df_roc   # already in memory, no need to re-read

roc_train = roc_full[roc_full['split'] == 'train'].sort_values('fpr')
roc_test  = roc_full[roc_full['split'] == 'test'].sort_values('fpr')

def downsample_roc(df, n_points=500):
    """Keep evenly-spaced points along the curve."""
    if len(df) <= n_points:
        return df
    indices = np.linspace(0, len(df) - 1, n_points).astype(int)
    return df.iloc[indices]

roc_train_small = downsample_roc(roc_train, 500)
roc_test_small = downsample_roc(roc_test, 500)
roc_small = pd.concat([roc_train_small, roc_test_small])
roc_small.to_csv('ifrs9_app/data/pd_roc_curve_small.csv', index=False)
print(f'ROC downsampled: {len(roc_full):,} → {len(roc_small):,} rows')

# %%
# =============================================================================
# 12. CREATING THE SCORECARD
# =============================================================================
# Step 1 — Build reference category rows (coeff = 0, p_value = NaN)
df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
df_ref_categories['Coefficients'] = 0
df_ref_categories['p_values']     = np.nan

# %%
# Step 2 — Concatenate model rows + reference rows
df_scorecard = pd.concat([summary_table, df_ref_categories])
df_scorecard = df_scorecard.reset_index(drop=True)

# %%
# Step 3 — Extract original feature name (part before the colon)
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

# %%
# Step 4 — Score scaling parameters
min_score = 300
max_score = 850

min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

print(f"Min sum coef: {min_sum_coef:.4f}")
print(f"Max sum coef: {max_sum_coef:.4f}")

# %%
# Step 5 — Scale coefficients to score range [300, 850]
df_scorecard['Score - Calculation'] = (
    df_scorecard['Coefficients'] *
    (max_score - min_score) / (max_sum_coef - min_sum_coef)
)

# Intercept anchors the scale
df_scorecard.iloc[0, df_scorecard.columns.get_loc('Score - Calculation')] = (
    (df_scorecard['Coefficients'].iloc[0] - min_sum_coef) /
    (max_sum_coef - min_sum_coef) *
    (max_score - min_score) + min_score
)

# %%
# Step 6 — Round to final integer scores
df_scorecard['Score - Final'] = df_scorecard['Score - Calculation'].round()

# %%
# Step 7 — Verify range
min_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
max_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
print(f"Min score: {min_sum_score_final}  (target: {min_score})")
print(f"Max score: {max_sum_score_final}  (target: {max_score})")

# %%
# ── Find the rounding gap ─────────────────────────────────────────────────────
df_scorecard['Difference'] = df_scorecard['Score - Final'] - df_scorecard['Score - Calculation']

# Find the min bin per variable (the one that contributes to max_sum_score_final)
max_bins = df_scorecard.loc[
    df_scorecard.groupby('Original feature name')['Score - Final'].idxmax(),
    ['Feature name', 'Original feature name', 'Score - Calculation', 'Score - Final', 'Difference']
].sort_values('Difference')

print(max_bins)

# ── Add 1 to the max bin with the largest negative rounding difference ────────
adjust_idx = max_bins['Difference'].idxmin()
print(f"\nAdjusting: {df_scorecard.loc[adjust_idx, 'Feature name']}")

df_scorecard.loc[adjust_idx, 'Score - Final'] = df_scorecard.loc[adjust_idx, 'Score - Final'] + 1

# ── Verify ────────────────────────────────────────────────────────────────────
min_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
max_sum_score_final = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
print(f"Min score: {min_sum_score_final}  (target: {min_score})")
print(f"Max score: {max_sum_score_final}  (target: {max_score})")

# %%
# =============================================================================
# 13. CALCULATING CREDIT SCORES FOR THE TEST SET
# =============================================================================
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat.copy()
inputs_test_with_ref_cat_w_intercept.insert(0, 'const', 1)

# Keep only scorecard columns in scorecard order
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values
]

scorecard_scores = df_scorecard['Score - Final'].values.reshape(-1, 1)
y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)

print(f"Score range — Min: {y_scores.values.min():.0f}  Max: {y_scores.values.max():.0f}")
y_scores.head()

# %%
# =============================================================================
# 14. FROM CREDIT SCORE TO PD
# =============================================================================
sum_coef_from_score = (
    (y_scores.astype(float) - min_score) / (max_score - min_score) *
    (max_sum_coef - min_sum_coef) + min_sum_coef
)
y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)

# ── Sanity check ──────────────────────────────────────────────────────────────
print("From logit : ", pd_test.values[:5].round(6))
print("From score : ", y_hat_proba_from_score.values.flatten()[:5].round(6))

# ── Compute train scores ───────────────────────────────────────────────────
inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat.copy()
inputs_train_with_ref_cat_w_intercept.insert(0, 'const', 1)
inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values
]
train_scores = inputs_train_with_ref_cat_w_intercept.dot(scorecard_scores)

# ── Score distribution ─────────────────────────────────────────────────────
df_score_dist = pd.concat([
    pd.DataFrame({'score': train_scores.values.flatten(), 'split': 'train'}),
    pd.DataFrame({'score': y_scores.values.flatten(),     'split': 'test'})
], ignore_index=True)

df_score_dist.to_csv('pd_score_distribution.csv', index=False)
print(f"✓ pd_score_distribution.csv saved — {len(df_score_dist):,} rows")

# ── Summary stats ──────────────────────────────────────────────────────────
print(df_score_dist.groupby('split')['score'].describe().round(1))
# %%
# =============================================================================
# 12. SAVE ARTEFACTS
# =============================================================================
# ── Add credit score and score-derived PD to df_predictions (test set only) ──
df_predictions.loc[df_predictions['split'] == 'test', 'credit_score'] = y_scores.values.flatten()
df_predictions.loc[df_predictions['split'] == 'test', 'y_pred_proba_from_score'] = \
    y_hat_proba_from_score.values.flatten()

# ── 3. Model metrics ──────────────────────────────────────────────────────────
pd_metrics = {
    'AUROC_train':    round(AUROC_train,    4),
    'AUROC_test':     round(AUROC_test,     4),
    'Gini_train':     round(Gini_train,     4),
    'Gini_test':      round(Gini_test,      4),
    'KS_test':        round(KS_test,        4),
    'accuracy_test':  round(accuracy_test,  4),
    'threshold':      threshold,
    'gini_gap':       round(Gini_train - Gini_test, 4),
    'n_train':        len(loan_data_train),
    'n_test':         len(loan_data_test),
    'bad_rate_train': round(1 - loan_data_train['good_bad'].mean(), 4),
    'bad_rate_test':  round(1 - loan_data_test['good_bad'].mean(),  4),
    'score_min':      int(min_sum_score_final),
    'score_max':      int(max_sum_score_final),
    'score_target_min': min_score,
    'score_target_max': max_score,
}
# ── 4. Feature name mapping ───────────────────────────────────────────────────
feature_mapping = pd.DataFrame({
    'index':        range(len(X_train.columns)),
    'feature_name': X_train.columns,
    'is_ref_cat':   [col in ref_categories for col in X_train.columns],
    'coefficient':  result.params,
    'p_value':      result.pvalues.round(4),
})
feature_mapping['category'] = feature_mapping['feature_name'].apply(lambda x:
    'CHARACTER'  if any(x.startswith(p) for p in
        ['grade:', 'verification_status:', 'inq_last_6mths:',
         'mths_since_earliest_cr_line:', 'delinq_2yrs:']) else
    'CAPACITY'   if any(x.startswith(p) for p in
        ['annual_inc:', 'dti:', 'installment:', 'emp_length_int:']) else
    'CAPITAL'    if any(x.startswith(p) for p in
        ['total_rev_hi_lim:', 'home_ownership:', 'total_acc:', 'open_acc:']) else
    'CONDITIONS' if any(x.startswith(p) for p in
        ['int_rate:', 'term:', 'purpose:', 'addr_state:', 'initial_list_status:',
         'mths_since_issue_d:']) else
    'INTERCEPT'
)
feature_mapping.to_csv('ifrs9_app/data/pd_feature_mapping.csv', index=False)

# ── 5. Scorecard ──────────────────────────────────────────────────────────────
df_scorecard.to_csv('ifrs9_app/data/df_scorecard.csv', index=False)

# ── 6. Scorecard scaling parameters ──────────────────────────────────────────
scorecard_params = {
    'min_score':     min_score,
    'max_score':     max_score,
    'min_sum_coef':  min_sum_coef,
    'max_sum_coef':  max_sum_coef,
}
with open('ifrs9_app/data/scorecard_params.pkl', 'wb') as f:
    pickle.dump(scorecard_params, f)


# ─── Bin score distribution into histograms ─────────────────
bins = np.arange(300, 850, 10)

hist_train, edges = np.histogram(
    df_score_dist[df_score_dist['split'] == 'train']['score'], bins=bins
)
hist_test, _ = np.histogram(
    df_score_dist[df_score_dist['split'] == 'test']['score'], bins=bins
)

score_hist = pd.DataFrame({
    'bin_center':        (edges[:-1] + edges[1:]) / 2,
    'train_proportion':  hist_train / hist_train.sum(),
    'test_proportion':   hist_test  / hist_test.sum(),
    'train_count':       hist_train,
    'test_count':        hist_test
})
score_hist.to_csv('ifrs9_app/data/pd_score_histogram.csv', index=False)
print(f'Score distribution binned: {len(df_score_dist):,} rows → {len(score_hist)} bins')

pd.DataFrame([pd_metrics]).to_csv('ifrs9_app/data/pd_metrics.csv', index=False)
print("✓ pd_metrics.csv saved")
# %%
