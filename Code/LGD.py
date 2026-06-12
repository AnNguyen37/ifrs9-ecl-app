# %% 
%reset -f

# =============================================================================
# 0. MEMORY CHECK
# =============================================================================
import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
print(f"Used memory:      {psutil.virtual_memory().used   / 1024**3:.2f} GB")
print(f"Total memory:     {psutil.virtual_memory().total  / 1024**3:.2f} GB")
print(f"Memory usage:     {psutil.virtual_memory().percent:.1f}%")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
sns.set()

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# %%
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
    'total_rec_prncp',
    'recoveries',
    'funded_amnt',
    'issue_d',
    # ── CHARACTER ──────────────────────────────────────────────────────────
    'grade',
    'verification_status',
    'inq_last_6mths',
    'earliest_cr_line',
    # ── CAPACITY ───────────────────────────────────────────────────────────
    'annual_inc',
    'dti',
    'installment',
    # ── CAPITAL ────────────────────────────────────────────────────────────
    'total_rev_hi_lim',
    'home_ownership', 
    # ── CONDITIONS ─────────────────────────────────────────────────────────
    'int_rate',
    'term',
    'purpose',
    'addr_state', 
]

loan_data = pd.read_csv(
    'ifrs9_app/data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018q4.csv',
    usecols=cols_needed
)
print(f"Raw dataset shape: {loan_data.shape}")
del cols_needed

# =============================================================================
# 2. FILTER TO DEFAULTED LOANS ONLY
# =============================================================================
# %%
# LGD and EAD are estimated on the defaulted population only.
# We use Charged Off and policy violations — these have resolved outcomes.
# 'Late (31-120 days)' excluded here as recovery is not yet finalised.
finished_statuses = [
    'Fully Paid',
    'Charged Off',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
]

loan_data = loan_data[loan_data['loan_status'].isin(finished_statuses)]

loan_data_defaults = loan_data[
    loan_data['loan_status'].isin([
        'Charged Off',
        'Does not meet the credit policy. Status:Charged Off'
    ])
].copy()

print(f"Total loans       : {loan_data.shape[0]:,.0f}")
print(f"Defaulted loans   : {loan_data_defaults.shape[0]:,.0f}")
print(f"Default rate      : {loan_data_defaults.shape[0]/loan_data.shape[0]*100:.2f}%")

# =============================================================================
# 3. LGD — TARGET VARIABLE
# =============================================================================
# %%
# Recovery Rate = fraction of funded amount recovered after default.
# LGD = 1 - Recovery Rate
#
# Components:
#   total_rec_prncp  — principal recovered during loan lifetime
#   recoveries       — post-charge-off recovery amount
#   funded_amnt      — original disbursed amount (= EAD proxy for term loans)

loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp'])

# Cap between 0 and 1 (edge cases: over-recovery or data issues)
loan_data_defaults['recovery_rate'] = loan_data_defaults['recovery_rate'].clip(lower=0, upper=1)

loan_data_defaults['LGD'] = 1 - loan_data_defaults['recovery_rate']

# %%
print("Recovery Rate distribution:")
print(loan_data_defaults['recovery_rate'].describe().round(4))
print(f"\nFully recovered (RR = 1) : {(loan_data_defaults['recovery_rate'] == 1).sum():,.0f}")
print(f"Zero recovery   (RR = 0) : {(loan_data_defaults['recovery_rate'] == 0).sum():,.0f}")
print(f"Partial recovery         : {((loan_data_defaults['recovery_rate'] > 0) & (loan_data_defaults['recovery_rate'] < 1)).sum():,.0f}")

# %%
plt.figure(figsize=(10, 5))
loan_data_defaults['recovery_rate'].hist(bins=100, color='steelblue', edgecolor='white')
plt.xlabel('Recovery Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Recovery Rate (Defaulted Loans)')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# =============================================================================
# 4. EAD — TARGET VARIABLE
# =============================================================================
# %%
# For fully drawn term loans (LendingClub), EAD ratio = 1.
# EAD = outstanding principal balance at time of default.
# EAD ratio = EAD / funded_amnt (credit utilisation at default)

loan_data_defaults['EAD']        = loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']
loan_data_defaults['EAD_ratio']        = loan_data_defaults['EAD'] / loan_data_defaults['funded_amnt']
loan_data_defaults['EAD_ratio']        = loan_data_defaults['EAD_ratio'].clip(lower=0, upper=1)

# %%
print("EAD distribution:")
print(loan_data_defaults['EAD'].describe().round(2))
print("EAD ratio distribution:")
print(loan_data_defaults['EAD_ratio'].describe().round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
loan_data_defaults['EAD'].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('EAD Distribution')
axes[0].set_xlabel('EAD (USD)')

loan_data_defaults['EAD_ratio'].hist(bins=50, ax=axes[1], color='darkorange', edgecolor='white')
axes[1].set_title('EAD ratio Distribution')
axes[1].set_xlabel('EAD ratio')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

# %%
# --- earliest_cr_line → months since [CHARACTER] ---
loan_data_defaults['earliest_cr_line_date'] = pd.to_datetime(loan_data_defaults['earliest_cr_line'], format='%b-%Y')
loan_data_defaults['mths_since_earliest_cr_line'] = round(
    pd.to_numeric((pd.to_datetime('2026-01-01') - loan_data_defaults['earliest_cr_line_date']) / np.timedelta64(1, 'D')) / 30.5)
loan_data_defaults['mths_since_earliest_cr_line'] = loan_data_defaults['mths_since_earliest_cr_line'].fillna(0)
with pd.option_context('display.float_format', '{:.2f}'.format):
    print(loan_data_defaults['mths_since_earliest_cr_line'].describe())
loan_data_defaults.drop(columns=['earliest_cr_line_date'], inplace=True)    

# %%
# --- term → numeric [CONDITIONS] ---
loan_data_defaults['term_int'] = pd.to_numeric(
    loan_data_defaults['term'].str.strip().str.replace(' months', '', regex=False))
loan_data_defaults.drop(columns=['term'], inplace=True)

# %%
# --- issue_d → months since [CONDITIONS] ---
loan_data_defaults['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%Y')


# %%
# =============================================================================
# 6. MISSING VALUE TREATMENT
# =============================================================================
# total_rev_hi_lim → funded_amnt proxy       [CAPITAL]
# annual_inc       → mean imputation          [CAPACITY]
# Count variables  → 0 (no events recorded)  [CHARACTER / CAPITAL]
# -----------------------------------------------------------------------------
loan_data_defaults['total_rev_hi_lim'] = loan_data_defaults['total_rev_hi_lim'].fillna(loan_data_defaults['funded_amnt'])
loan_data_defaults['annual_inc']        = loan_data_defaults['annual_inc'].fillna(loan_data_defaults['annual_inc'].mean())

for col in ['mths_since_earliest_cr_line', 'inq_last_6mths']:
    loan_data_defaults[col] = loan_data_defaults[col].fillna(0)

# %%
# =============================================================================
# 7. DUMMY VARIABLES
# =============================================================================
loan_data_dummies = pd.concat([
    # CHARACTER
    pd.get_dummies(loan_data_defaults['grade'],              prefix='grade',              prefix_sep=':'),
    pd.get_dummies(loan_data_defaults['verification_status'], prefix='verification_status', prefix_sep=':'),
    # CAPITAL
    pd.get_dummies(loan_data_defaults['home_ownership'],      prefix='home_ownership',      prefix_sep=':'),
    # CONDITIONS
    pd.get_dummies(loan_data_defaults['addr_state'],          prefix='addr_state',          prefix_sep=':'),
    pd.get_dummies(loan_data_defaults['purpose'],             prefix='purpose',             prefix_sep=':'),
], axis=1)

loan_data_defaults = pd.concat([loan_data_defaults, loan_data_dummies], axis=1)
del loan_data_dummies

# %%
# --- purpose
loan_data_defaults['purpose:small_business_renewable_energy_moving_house_medical'] = np.where(
    loan_data_defaults['purpose'].isin(['small_business', 'renewable_energy', 'moving', 'house', 'medical']), 1, 0)
loan_data_defaults['purpose:debt_consolidation'] = np.where(
    loan_data_defaults['purpose'] == 'debt_consolidation', 1, 0)
loan_data_defaults['purpose:educational_other'] = np.where(
    loan_data_defaults['purpose'].isin(['educational', 'other']), 1, 0)
loan_data_defaults['purpose:vacation_major_purchase_home_improvement'] = np.where(
    loan_data_defaults['purpose'].isin(['vacation', 'major_purchase', 'home_improvement']), 1, 0)
loan_data_defaults['purpose:credit_card_car_wedding'] = np.where(
    loan_data_defaults['purpose'].isin(['credit_card', 'car', 'wedding']), 1, 0)

# %%
# --- homeowners
loan_data_defaults['home_ownership:RENT_OTHER_NONE_ANY'] = sum([
    loan_data_defaults['home_ownership:RENT'],  loan_data_defaults['home_ownership:OTHER'],
    loan_data_defaults['home_ownership:NONE'],  loan_data_defaults['home_ownership:ANY']
])

# %%
# --- addr_state
loan_data_defaults['addr_state:NE_MS_AL_AR_OK'] = np.where(
    loan_data_defaults['addr_state'].isin(['NE', 'MS', 'AL', 'AR', 'OK']), 1, 0)
loan_data_defaults['addr_state:LA_NV_TN_SD'] = np.where(
    loan_data_defaults['addr_state'].isin(['LA', 'NV', 'TN', 'SD']), 1, 0)
loan_data_defaults['addr_state:NY'] = np.where(
    loan_data_defaults['addr_state'] == 'NY', 1, 0)
loan_data_defaults['addr_state:IN_NM_MO'] = np.where(
    loan_data_defaults['addr_state'].isin(['IN', 'NM', 'MO']), 1, 0)
loan_data_defaults['addr_state:FL'] = np.where(
    loan_data_defaults['addr_state'] == 'FL', 1, 0)
loan_data_defaults['addr_state:KY_MD_ND_IA_NJ_OH_PA_NC_ID'] = np.where(
    loan_data_defaults['addr_state'].isin(['KY', 'MD', 'ND', 'IA', 'NJ', 'OH', 'PA', 'NC', 'ID']), 1, 0)
loan_data_defaults['addr_state:HI_MI_VA_DE_MN_AZ'] = np.where(
    loan_data_defaults['addr_state'].isin(['HI', 'MI', 'VA', 'DE', 'MN', 'AZ']), 1, 0)
loan_data_defaults['addr_state:TX_AK'] = np.where(
    loan_data_defaults['addr_state'].isin(['TX', 'AK']), 1, 0)
loan_data_defaults['addr_state:CA'] = np.where(
    loan_data_defaults['addr_state'] == 'CA', 1, 0)
loan_data_defaults['addr_state:MA_GA_RI_IL_UT'] = np.where(
    loan_data_defaults['addr_state'].isin(['MA', 'GA', 'RI', 'IL', 'UT']), 1, 0)
loan_data_defaults['addr_state:WI_CT_MT_WY_SC_KS'] = np.where(
    loan_data_defaults['addr_state'].isin(['WI', 'CT', 'MT', 'WY', 'SC', 'KS']), 1, 0)
loan_data_defaults['addr_state:WV_WA_CO_ME_OR_NH_VT_DC'] = np.where(
    loan_data_defaults['addr_state'].isin(['WV', 'WA', 'CO', 'ME', 'OR', 'NH', 'VT', 'DC']), 1, 0)


# %%
# =============================================================================
# 8. TRAIN / TEST SPLIT — Temporal (time-based on issue_d_date)
# =============================================================================
# =============================================================================
cutoff = loan_data_defaults['issue_d_date'].quantile(0.80)
print(f"Temporal cutoff: {cutoff.strftime('%b-%Y')}")

train_mask = loan_data_defaults['issue_d_date'] <= cutoff   # older  80%
test_mask  = loan_data_defaults['issue_d_date'] >  cutoff   # recent 20%
# %%
# ── LGD Stage 1: Recovery Rate = 0 vs > 0 ─────────────────────────────────────
loan_data_defaults['recovery_rate_0_gt0'] = np.where(
    loan_data_defaults['recovery_rate'] == 0, 0, 1)

print("Recovery Rate = 0 : {:,} ({:.1f}%)".format(
    (loan_data_defaults['recovery_rate_0_gt0'] == 0).sum(),
    (loan_data_defaults['recovery_rate_0_gt0'] == 0).mean() * 100))
print("Recovery Rate > 0 : {:,} ({:.1f}%)".format(
    (loan_data_defaults['recovery_rate_0_gt0'] == 1).sum(),
    (loan_data_defaults['recovery_rate_0_gt0'] == 1).mean() * 100))

# %%
# ── Train / Test Split ─────────────────────────────────────────────────────────
X_train_lgd = loan_data_defaults[train_mask].drop(
    columns=['recovery_rate', 'recovery_rate_0_gt0', 'issue_d_date'])
X_test_lgd  = loan_data_defaults[test_mask].drop(
    columns=['recovery_rate', 'recovery_rate_0_gt0', 'issue_d_date'])

y_train_lgd = loan_data_defaults[train_mask]['recovery_rate']
y_test_lgd  = loan_data_defaults[test_mask]['recovery_rate']

print(f"\nTrain : {len(X_train_lgd):,}  | Mean recovery rate: {y_train_lgd.mean():.4f}")
print(f"Test  : {len(X_test_lgd):,}   | Mean recovery rate: {y_test_lgd.mean():.4f}")

# =============================================================================
# 9. FEATURE SELECTION
# =============================================================================
# %%
lgd_ead_features = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:OWN', 'home_ownership:MORTGAGE', 'home_ownership:RENT_OTHER_NONE_ANY',
    'verification_status:Not Verified', 'verification_status:Source Verified', 'verification_status:Verified',
    'purpose:small_business_renewable_energy_moving_house_medical',
    'purpose:debt_consolidation',
    'purpose:educational_other',
    'purpose:vacation_major_purchase_home_improvement',
    'purpose:credit_card_car_wedding',
    'term_int', 'mths_since_earliest_cr_line',
    'funded_amnt', 'int_rate', 'installment', 'annual_inc',
    'dti', 'total_rev_hi_lim',
    'addr_state:NE_MS_AL_AR_OK', 'addr_state:LA_NV_TN_SD',
    'addr_state:NY', 'addr_state:IN_NM_MO',
    'addr_state:FL', 'addr_state:KY_MD_ND_IA_NJ_OH_PA_NC_ID',
    'addr_state:HI_MI_VA_DE_MN_AZ', 'addr_state:TX_AK',
    'addr_state:CA', 'addr_state:MA_GA_RI_IL_UT',
    'addr_state:WI_CT_MT_WY_SC_KS', 'addr_state:WV_WA_CO_ME_OR_NH_VT_DC',
]

features_reference_cat = [
    'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY',
    'verification_status:Verified',
    'purpose:small_business_renewable_energy_moving_house_medical',
    'addr_state:NE_MS_AL_AR_OK',
]

# %%
inputs_train_with_ref_cat = X_train_lgd.loc[:, lgd_ead_features]
inputs_test_with_ref_cat  = X_test_lgd.loc[:,  lgd_ead_features]

# %%
inputs_train = inputs_train_with_ref_cat.drop(features_reference_cat, axis=1).astype(float)
inputs_test  = inputs_test_with_ref_cat.drop(features_reference_cat,  axis=1).astype(float)
# %%
# =============================================================================
# UNIFIED METRICS FUNCTION
# =============================================================================
def model_metrics(y_true, y_pred, label):
    """
    Primary   : MAE (calibration), Bias (calibration), Correlation (discrimination)
    Secondary : RMSE, R², Mean actual vs predicted
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Primary
    mae  = mean_absolute_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    # Secondary
    rmse          = np.sqrt(mean_squared_error(y_true, y_pred))
    r2            = r2_score(y_true, y_pred)
    mean_actual   = y_true.mean()
    mean_predicted= y_pred.mean()

    print(f"\n  {label}")
    print(f"  ── Primary ──────────────────────────────")
    print(f"    MAE         : {mae:.4f}   ← calibration")
    print(f"    Bias        : {bias:+.4f}  ← calibration")
    print(f"    Correlation : {corr:.4f}   ← discrimination")
    print(f"  ── Secondary ────────────────────────────")
    print(f"    RMSE        : {rmse:.4f}")
    print(f"    R²          : {r2:.4f}")
    print(f"    Mean actual : {mean_actual:.4f}")
    print(f"    Mean pred   : {mean_predicted:.4f}")

    return {
        # Primary
        'mae':             round(mae,  4),
        'bias':            round(bias, 4),
        'correlation':     round(corr, 4),
        # Secondary
        'rmse':            round(rmse,           4),
        'r2':              round(r2,             4),
        'mean_actual':     round(mean_actual,    4),
        'mean_predicted':  round(mean_predicted, 4),
        'label':           label,
    }

# %%
X_train = sm.add_constant(inputs_train)
X_test  = sm.add_constant(inputs_test)

from sklearn.metrics import (roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from statsmodels.othermod.betareg import BetaModel
import pickle

def beta_transform(y):
    n = len(y)
    return (y * (n - 1) + 0.5) / n

# ── Targets ───────────────────────────────────────────────────────────────────
y_train_lgd    = loan_data_defaults[train_mask]['recovery_rate']
y_test_lgd     = loan_data_defaults[test_mask]['recovery_rate']
y_train_stage1 = loan_data_defaults[train_mask]['recovery_rate_0_gt0']
y_test_stage1  = loan_data_defaults[test_mask]['recovery_rate_0_gt0']

# =============================================================================
# MODEL 1 — STAGE 1: Logistic Regression P(RR > 0)
# =============================================================================
mask_s1_train = X_train.notna().all(axis=1) & y_train_stage1.notna()
mask_s1_test  = X_test.notna().all(axis=1)  & y_test_stage1.notna()

X_train_s1 = X_train[mask_s1_train]
X_test_s1  = X_test[mask_s1_test]
y_train_s1 = y_train_stage1[mask_s1_train]
y_test_s1  = y_test_stage1[mask_s1_test]

logit_stage1  = sm.Logit(y_train_s1.astype(int).values, X_train_s1.values)
result_stage1 = logit_stage1.fit(disp=False, maxiter=200)

p_rr_gt0_test  = result_stage1.predict(X_test_s1.values)
p_rr_gt0_train = result_stage1.predict(X_train_s1.values)

auroc_s1       = roc_auc_score(y_test_s1, p_rr_gt0_test)
auroc_s1_train = roc_auc_score(y_train_s1, p_rr_gt0_train)

print(f"Stage 1 — AUROC train: {auroc_s1_train:.4f} | test: {auroc_s1:.4f}")
print(f"Class balance — 0: {(y_test_s1==0).mean():.1%}  1: {(y_test_s1==1).mean():.1%}")

# =============================================================================
# MODEL 1 — STAGE 2: Beta Regression on RR > 0 subset only
# =============================================================================
# Filter X_train / X_test directly to RR > 0 loans
train_gt0_idx = loan_data_defaults[
    train_mask & (loan_data_defaults['recovery_rate'] > 0)].index
test_gt0_idx  = loan_data_defaults[
    test_mask  & (loan_data_defaults['recovery_rate'] > 0)].index

X_train_s2 = X_train.loc[X_train.index.intersection(train_gt0_idx)]
X_test_s2  = X_test.loc[X_test.index.intersection(test_gt0_idx)]
y_train_s2 = y_train_lgd.loc[X_train_s2.index]
y_test_s2  = y_test_lgd.loc[X_test_s2.index]

mask_s2_train = X_train_s2.notna().all(axis=1) & y_train_s2.notna()
mask_s2_test  = X_test_s2.notna().all(axis=1)  & y_test_s2.notna()

X_train_s2_clean = X_train_s2[mask_s2_train]
X_test_s2_clean  = X_test_s2[mask_s2_test]
y_train_s2_clean = y_train_s2[mask_s2_train]
y_test_s2_clean  = y_test_s2[mask_s2_test]

y_train_s2_beta = beta_transform(y_train_s2_clean.values)
beta_model_s2   = BetaModel(y_train_s2_beta, X_train_s2_clean.values)
beta_result_s2  = beta_model_s2.fit(disp=False, maxiter=200)

y_pred_s2 = beta_result_s2.predict(X_test_s2_clean.values)

print(f"Stage 2 — n_train (RR>0): {len(X_train_s2_clean):,} | n_test: {len(X_test_s2_clean):,}")
print(f"Mean RR train: {y_train_s2_clean.mean():.4f} | test: {y_test_s2_clean.mean():.4f}")

# =============================================================================
# MODEL 1 — COMBINED 2-STAGE PREDICTION
# =============================================================================
common_idx = X_test_s1.index.intersection(X_test_s2_clean.index)

p_rr_gt0_aligned = pd.Series(p_rr_gt0_test,
                              index=X_test_s1.index).loc[common_idx]
p_rr_given_gt0   = pd.Series(y_pred_s2,
                              index=X_test_s2_clean.index).loc[common_idx]
y_actual_common  = y_test_lgd.loc[common_idx]

rr_pred_2stage  = p_rr_gt0_aligned * p_rr_given_gt0
lgd_pred_2stage = 1 - rr_pred_2stage
lgd_actual      = 1 - y_actual_common

print(f"\nCommon test set: {len(common_idx):,} loans")

# =============================================================================
# MODEL 2 — DIRECT BETA REGRESSION on full train dataset
# =============================================================================
mask_m2_train = X_train.notna().all(axis=1) & y_train_lgd.notna()
mask_m2_test  = X_test.notna().all(axis=1)  & y_test_lgd.notna()

X_train_m2 = X_train[mask_m2_train]
X_test_m2  = X_test[mask_m2_test]
y_train_m2 = y_train_lgd[mask_m2_train]
y_test_m2  = y_test_lgd[mask_m2_test]

y_train_m2_beta = beta_transform(y_train_m2.values)
beta_model      = BetaModel(y_train_m2_beta, X_train_m2.values)
beta_result     = beta_model.fit(disp=False, maxiter=200)

y_pred_direct   = beta_result.predict(X_test_m2.values)
lgd_pred_direct = pd.Series(1 - y_pred_direct,
                             index=X_test_m2.index).loc[common_idx]

print(f"Model 2 — n_train: {len(X_train_m2):,} | n_test: {len(X_test_m2):,}")
print(f"Mean RR train: {y_train_m2.mean():.4f} | test: {y_test_m2.mean():.4f}")
# %%
# =============================================================================
# LGD EVALUATION — MODEL 1 vs MODEL 2
# =============================================================================
print("=" * 55)
print("  LGD MODEL COMPARISON — Test Set")
print("=" * 55)
m1 = model_metrics(lgd_actual, lgd_pred_2stage, 'Model 1 — 2-Stage')
m2 = model_metrics(lgd_actual, lgd_pred_direct, 'Model 2 — Direct Beta Regression')
print("=" * 55)

# %%
# ── Prediction distribution comparison ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(lgd_pred_2stage, bins=50, color='steelblue',
             alpha=0.7, edgecolor='white', label='2-Stage')
axes[0].hist(lgd_pred_direct, bins=50, color='darkorange',
             alpha=0.7, edgecolor='white', label='Direct Beta')
axes[0].axvline(lgd_actual.mean(), color='k', linestyle='--', label='Actual mean')
axes[0].set_title('Predicted LGD Distribution')
axes[0].set_xlabel('LGD')
axes[0].legend()

axes[1].scatter(lgd_actual, lgd_pred_2stage,
                alpha=0.05, s=3, color='steelblue', label='2-Stage')
axes[1].scatter(lgd_actual, lgd_pred_direct,
                alpha=0.05, s=3, color='darkorange', label='Direct Beta')
axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect')
axes[1].set_title('Actual vs Predicted LGD')
axes[1].set_xlabel('Actual LGD')
axes[1].set_ylabel('Predicted LGD')
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# LGD SEGMENT-LEVEL COMPARISON
# =============================================================================
df_seg = pd.DataFrame({
    'actual_rr':   y_actual_common.values,
    'actual_lgd':  lgd_actual.values,
    'pred_lgd_m1': lgd_pred_2stage.values,
    'pred_lgd_m2': lgd_pred_direct.values,
}, index=common_idx)

seg_cols = loan_data_defaults.loc[common_idx,
           ['grade', 'term_int', 'purpose', 'home_ownership']]
df_seg   = df_seg.join(seg_cols)

purpose_map = {
    'small_business': 'small_business_renewable_energy_moving_house_medical',
    'renewable_energy':'small_business_renewable_energy_moving_house_medical',
    'moving':         'small_business_renewable_energy_moving_house_medical',
    'house':          'small_business_renewable_energy_moving_house_medical',
    'medical':        'small_business_renewable_energy_moving_house_medical',
    'debt_consolidation': 'debt_consolidation',
    'educational':    'educational_other',
    'other':          'educational_other',
    'vacation':       'vacation_major_purchase_home_improvement',
    'major_purchase': 'vacation_major_purchase_home_improvement',
    'home_improvement':'vacation_major_purchase_home_improvement',
    'credit_card':    'credit_card_car_wedding',
    'car':            'credit_card_car_wedding',
    'wedding':        'credit_card_car_wedding',
}
home_map = {
    'OWN': 'OWN', 'MORTGAGE': 'MORTGAGE',
    'RENT': 'RENT_OTHER_NONE_ANY', 'OTHER': 'RENT_OTHER_NONE_ANY',
    'NONE': 'RENT_OTHER_NONE_ANY', 'ANY':   'RENT_OTHER_NONE_ANY',
}
df_seg['purpose_group']        = df_seg['purpose'].map(purpose_map)
df_seg['home_ownership_group'] = df_seg['home_ownership'].map(home_map)
df_seg['term_label']           = df_seg['term_int'].map({36:'36 months', 60:'60 months'})

def segment_comparison(df, segment_col, segment_label):
    rows = []
    for seg_val, grp in df.groupby(segment_col):
        actual  = grp['actual_lgd'].values
        pred_m1 = grp['pred_lgd_m1'].values
        pred_m2 = grp['pred_lgd_m2'].values
        rows.append({
            'segment_type':    segment_label,
            'bin':             seg_val,
            'n':               len(grp),
            'actual_lgd_mean': round(actual.mean(), 4),
            # ── Primary ──
            'm1_mae':          round(mean_absolute_error(actual, pred_m1), 4),
            'm1_bias':         round((pred_m1 - actual).mean(), 4),
            'm1_corr':         round(np.corrcoef(actual, pred_m1)[0,1], 4),
            'm1_lgd_mean':     round(pred_m1.mean(), 4),
            # ── Secondary ──
            'm1_rmse':         round(np.sqrt(mean_squared_error(actual, pred_m1)), 4),
            'm2_mae':          round(mean_absolute_error(actual, pred_m2), 4),
            'm2_bias':         round((pred_m2 - actual).mean(), 4),
            'm2_corr':         round(np.corrcoef(actual, pred_m2)[0,1], 4),
            'm2_lgd_mean':     round(pred_m2.mean(), 4),
            'm2_rmse':         round(np.sqrt(mean_squared_error(actual, pred_m2)), 4),
        })
    df_out = pd.DataFrame(rows)
    df_out['better_mae']  = np.where(df_out['m1_mae']  < df_out['m2_mae'],  'Model 1', 'Model 2')
    df_out['better_bias'] = np.where(df_out['m1_bias'].abs() < df_out['m2_bias'].abs(), 'Model 1', 'Model 2')
    df_out['better_corr'] = np.where(df_out['m1_corr'] > df_out['m2_corr'], 'Model 1', 'Model 2')
    return df_out

seg_grade = segment_comparison(df_seg, 'grade',               'grade')
seg_term  = segment_comparison(df_seg, 'term_label',          'term')
seg_purp  = segment_comparison(df_seg, 'purpose_group',       'purpose')
seg_home  = segment_comparison(df_seg, 'home_ownership_group','home_ownership')

df_lgd_segments = pd.concat([seg_grade, seg_term, seg_purp, seg_home],
                              ignore_index=True)

for seg_type, grp in df_lgd_segments.groupby('segment_type'):
    print(f"\n── Segment: {seg_type.upper()} ──")
    print(grp[['bin', 'n', 'actual_lgd_mean',
                'm1_mae', 'm1_bias', 'm1_corr',
                'm2_mae', 'm2_bias', 'm2_corr',
                'better_mae']].to_string(index=False))

grade_plot = seg_grade.set_index('bin')[
    ['actual_lgd_mean', 'm1_lgd_mean', 'm2_lgd_mean']].sort_index()
grade_plot.columns = ['Actual', 'Model 1 (2-Stage)', 'Model 2 (Direct Beta)']
grade_plot.plot(kind='bar', figsize=(10, 5),
                color=['steelblue', 'darkorange', 'green'], edgecolor='white')
plt.title('Actual vs Predicted LGD by Grade')
plt.xlabel('Grade')
plt.ylabel('Mean LGD')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ── Constant LGD benchmark ────────────────────────────────────────────────
constant_lgd = 0.85

mae_benchmark = mean_absolute_error(lgd_actual, np.full(len(lgd_actual), constant_lgd))
mae_model1    = mean_absolute_error(lgd_actual, lgd_pred_2stage)
mae_model2    = mean_absolute_error(lgd_actual, lgd_pred_direct)

print(f"Mean LGD (constant prediction) : {constant_lgd:.4f}")
print(f"")
print(f"MAE — Benchmark (constant)     : {mae_benchmark:.4f}")
print(f"MAE — Model 1 (2-Stage)        : {mae_model1:.4f}  (improvement: {mae_benchmark - mae_model1:.4f})")
print(f"MAE — Model 2 (Direct Beta)    : {mae_model2:.4f}  (improvement: {mae_benchmark - mae_model2:.4f})")
# %%
# Calculate the baseline MAE explicitly
actual_lgd_test = 1- y_test_lgd  # your test set actual LGD values
benchmark_pred = 0.8555  # constant prediction

baseline_mae = np.mean(np.abs(actual_lgd_test - benchmark_pred))
print(f'Baseline MAE: {baseline_mae:.4f}')
print(f'2-stage model MAE: 0.0442')
print(f'Improvement: {(1 - 0.0442/baseline_mae)*100:.1f}%')

# %%
# =============================================================================
# EAD MODEL — OLS Regression on EAD Ratio (CCF)
# =============================================================================
# EAD ratio = (funded_amnt - total_rec_prncp) / funded_amnt
#           = fraction of funded amount still outstanding at default
# Clipped to [0, 1] — CCF interpretation
# =============================================================================

# ── Targets ───────────────────────────────────────────────────────────────────
y_train_ead = loan_data_defaults[train_mask]['EAD_ratio']
y_test_ead  = loan_data_defaults[test_mask]['EAD_ratio']

# ── Drop NaN rows ──────────────────────────────────────────────────────────────
mask_ead_train = X_train.notna().all(axis=1) & y_train_ead.notna()
mask_ead_test  = X_test.notna().all(axis=1)  & y_test_ead.notna()

X_train_ead = X_train[mask_ead_train]
X_test_ead  = X_test[mask_ead_test]
y_train_ead = y_train_ead[mask_ead_train]
y_test_ead  = y_test_ead[mask_ead_test]

print(f"Train : {len(X_train_ead):,}  | Mean EAD ratio: {y_train_ead.mean():.4f}")
print(f"Test  : {len(X_test_ead):,}   | Mean EAD ratio: {y_test_ead.mean():.4f}")

# %%
# ── Fit OLS ───────────────────────────────────────────────────────────────────
ead_model  = sm.OLS(y_train_ead, X_train_ead)
ead_result = ead_model.fit()
print(ead_result.summary())

# %%
# ── Summary Table ─────────────────────────────────────────────────────────────
summary_table_ead = pd.DataFrame({
    'Feature name': X_train_ead.columns,
    'Coefficients': ead_result.params.values,
    'p_values':     ead_result.pvalues.round(4).values
})
print(summary_table_ead)


# %%
# ── Predict & Clip to [0, 1] ──────────────────────────────────────────────────
y_hat_ead_train = np.clip(ead_result.predict(X_train_ead.values), 0, 1)
y_hat_ead_test  = np.clip(ead_result.predict(X_test_ead.values),  0, 1)

# %%
# =============================================================================
# EAD MODEL — RECALIBRATION
# =============================================================================
# Method: Fix slope coefficients, refit intercept only on full dataset.
# This corrects for systematic over/under-prediction without changing
# the rank-ordering of the model.
# =============================================================================

# ── Step 1: Compute offset predictions (slopes only, no intercept) ────────────
# Remove intercept contribution from original predictions
intercept_orig = ead_result.params['const']
coef_no_intercept = ead_result.params.drop('const')

# ── Step 2: Compute slope-only predictions on full dataset ───────────────────
X_all_clean_ead  = pd.concat([X_train_ead, X_test_ead])
y_all_ead        = pd.concat([y_train_ead, y_test_ead])

# Slope component only (exclude const column)
X_slopes = X_all_clean_ead.drop(columns=['const'])
y_hat_slopes = X_slopes.values @ coef_no_intercept.values  # no intercept

# ── Step 3: Refit intercept using OLS: y = intercept + slope_preds ────────────
# Equivalent to: intercept_new = mean(y_actual) - mean(slope_preds)
intercept_new = (y_all_ead.values - y_hat_slopes).mean()

print(f"Original intercept : {intercept_orig:.6f}")
print(f"Recalibrated intercept: {intercept_new:.6f}")
print(f"Adjustment         : {intercept_new - intercept_orig:+.6f}")

# ── Step 4: Recalibrated predictions ─────────────────────────────────────────
def predict_recalibrated(X, coef_no_intercept, intercept_new):
    X_slopes = X.drop(columns=['const'])
    y_hat    = X_slopes.values @ coef_no_intercept.values + intercept_new
    return np.clip(y_hat, 0, 1)

y_hat_ead_train_recal = predict_recalibrated(X_train_ead, coef_no_intercept, intercept_new)
y_hat_ead_test_recal  = predict_recalibrated(X_test_ead,  coef_no_intercept, intercept_new)
# %%
# =============================================================================
# EAD EVALUATION — ORIGINAL vs RECALIBRATED
# =============================================================================
print("─" * 55)
print("EAD Model Performance")
print("─" * 55)
metrics_ead_train       = model_metrics(y_train_ead, y_hat_ead_train,       'Train — Original')
metrics_ead_test        = model_metrics(y_test_ead,  y_hat_ead_test,        'Test  — Original')
metrics_ead_train_recal = model_metrics(y_train_ead, y_hat_ead_train_recal, 'Train — Recalibrated')
metrics_ead_test_recal  = model_metrics(y_test_ead,  y_hat_ead_test_recal,  'Test  — Recalibrated')
print("─" * 55)
print(f"\n  R² gap (original)     : {metrics_ead_train['r2'] - metrics_ead_test['r2']:.4f}")
print(f"  R² gap (recalibrated) : {metrics_ead_train_recal['r2'] - metrics_ead_test_recal['r2']:.4f}")

# ── Residuals plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_test_ead.values - y_hat_ead_test, bins=50,
             color='steelblue', alpha=0.7, edgecolor='white', label='Original')
axes[0].hist(y_test_ead.values - y_hat_ead_test_recal, bins=50,
             color='darkorange', alpha=0.7, edgecolor='white', label='Recalibrated')
axes[0].axvline(0, color='k', linestyle='--')
axes[0].set_title('EAD — Residuals Distribution (Test)')
axes[0].set_xlabel('Residual')
axes[0].legend()

axes[1].scatter(y_test_ead, y_hat_ead_test,
                alpha=0.05, s=3, color='steelblue', label='Original')
axes[1].scatter(y_test_ead, y_hat_ead_test_recal,
                alpha=0.05, s=3, color='darkorange', label='Recalibrated')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title('Actual vs Predicted EAD Ratio (Test)')
axes[1].set_xlabel('Actual EAD Ratio')
axes[1].set_ylabel('Predicted EAD Ratio')
axes[1].legend()
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# EAD FULL PORTFOLIO — RECALIBRATED
# =============================================================================
X_all        = pd.concat([X_train, X_test])
X_all_ead    = X_all[X_all.notna().all(axis=1)]

df_ead_portfolio = pd.DataFrame({
    'EAD_ratio_original':     np.clip(ead_result.predict(X_all_ead.values), 0, 1).round(6),
    'EAD_ratio_recalibrated': predict_recalibrated(X_all_ead, coef_no_intercept, intercept_new).round(6),
}, index=X_all_ead.index)
df_ead_portfolio.index.name  = 'id'
df_ead_portfolio['funded_amnt']      = loan_data_defaults.loc[X_all_ead.index, 'funded_amnt']
df_ead_portfolio['EAD_original']     = (df_ead_portfolio['EAD_ratio_original']     * df_ead_portfolio['funded_amnt']).round(2)
df_ead_portfolio['EAD_recalibrated'] = (df_ead_portfolio['EAD_ratio_recalibrated'] * df_ead_portfolio['funded_amnt']).round(2)

print(f"EAD ratio — Original     mean: {df_ead_portfolio['EAD_ratio_original'].mean():.4f}")
print(f"EAD ratio — Recalibrated mean: {df_ead_portfolio['EAD_ratio_recalibrated'].mean():.4f}")
print(f"Actual    — Full dataset mean: {y_all_ead.mean():.4f}")
print(f"Total EAD (recalibrated): ${df_ead_portfolio['EAD_recalibrated'].sum():,.0f}")

# %%
# =============================================================================
# SAVE — Page 4 LGD Estimation App Data
# =============================================================================

import os, pandas as pd, numpy as np
os.makedirs('data', exist_ok=True)

lgd_summary_csv = pd.DataFrame([
    {
        'model':          'Model 1 — 2-Stage',
        'auroc_stage1':   round(auroc_s1, 4),
        'gini_stage1':    round(auroc_s1 * 2 - 1, 4),
        'n_train':        len(X_train_s1),
        'n_test':         len(X_test_s1),
        'mae':            m1['mae'],
        'bias':           m1['bias'],
        'correlation':    m1['correlation'],
        'rmse':           m1['rmse'],
        'r2':             m1['r2'],
        'mean_lgd_pred':  m1['mean_predicted'],
        'mean_lgd_actual':m1['mean_actual'],
    },
    {
        'model':          'Model 2 — Direct Beta',
        'auroc_stage1':   None,
        'gini_stage1':    None,
        'n_train':        len(X_train_m2),
        'n_test':         len(X_test_m2),
        'mae':            m2['mae'],
        'bias':           m2['bias'],
        'correlation':    m2['correlation'],
        'rmse':           m2['rmse'],
        'r2':             m2['r2'],
        'mean_lgd_pred':  m2['mean_predicted'],
        'mean_lgd_actual':m2['mean_actual'],
    },
])

lgd_summary_csv.to_csv('ifrs9_app/data/lgd_summary.csv', index=False)
print("✓ data/lgd_summary.csv saved")
print(lgd_summary_csv.to_string(index=False))

# %%
# ── lgd_segments.csv → data/ ─────────────────────────────────────────────────
df_lgd_segments.to_csv('ifrs9_app/data/lgd_segments.csv', index=False)
print(f"✓ data/lgd_segments.csv — {len(df_lgd_segments)} rows")


# %%
# ── lgd_head_to_head.csv ─────────────────────────────────────────────────────
# 3-approach comparison: Model 1 (2-Stage), Model 2 (Direct Beta), Constant
constant_lgd = lgd_actual.mean()
m_const = {
    'mae':         round(mean_absolute_error(lgd_actual, np.full(len(lgd_actual), constant_lgd)), 4),
    'bias':        round(0.0, 4),
    'correlation': round(0.0, 4),
    'rmse':        round(np.sqrt(mean_squared_error(lgd_actual, np.full(len(lgd_actual), constant_lgd))), 4),
    'r2':          round(r2_score(lgd_actual, np.full(len(lgd_actual), constant_lgd)), 4),
    'mean_actual':    round(float(lgd_actual.mean()), 4),
    'mean_predicted': round(float(constant_lgd), 4),
}

lgd_head_to_head = pd.DataFrame([
    {
        'model':          'Model 1 — 2-Stage',
        'mae':            m1['mae'],
        'bias':           m1['bias'],
        'correlation':    m1['correlation'],
        'rmse':           m1['rmse'],
        'r2':             m1['r2'],
        'mean_predicted': m1['mean_predicted'],
        'mean_actual':    m1['mean_actual'],
    },
    {
        'model':          'Model 2 — Direct Beta',
        'mae':            m2['mae'],
        'bias':           m2['bias'],
        'correlation':    m2['correlation'],
        'rmse':           m2['rmse'],
        'r2':             m2['r2'],
        'mean_predicted': m2['mean_predicted'],
        'mean_actual':    m2['mean_actual'],
    },
    {
        'model':          'Benchmark — Constant LGD',
        'mae':            m_const['mae'],
        'bias':           m_const['bias'],
        'correlation':    m_const['correlation'],
        'rmse':           m_const['rmse'],
        'r2':             m_const['r2'],
        'mean_predicted': m_const['mean_predicted'],
        'mean_actual':    m_const['mean_actual'],
    },
])
lgd_head_to_head.to_csv('ifrs9_app/data/lgd_head_to_head.csv', index=False)
print(f"✓ data/lgd_head_to_head.csv — {len(lgd_head_to_head)} models")
# %%
stage1_coef = pd.DataFrame({
    'feature':     X_train_s1.columns,
    'coefficient': result_stage1.params,
    'p_value':     result_stage1.pvalues.round(4),
    'significant': result_stage1.pvalues < 0.05,
})
stage1_coef.to_csv('ifrs9_app/data/lgd_stage1_coefficients.csv', index=False)
print(f"✓ data/lgd_stage1_coefficients.csv — {len(stage1_coef)} features")

# BetaModel params = [feature coefficients..., phi (precision)]
n_features = len(X_train_s2_clean.columns)

stage2_coef = pd.DataFrame({
    'feature':     list(X_train_s2_clean.columns) + ['phi (precision)'],
    'coefficient': beta_result_s2.params,
    'p_value':     beta_result_s2.pvalues.round(4),
    'significant': beta_result_s2.pvalues < 0.05,
})
stage2_coef.to_csv('ifrs9_app/data/lgd_stage2_coefficients.csv', index=False)
print(f"✓ data/lgd_stage2_coefficients.csv — {len(stage2_coef)} features (incl. phi)")
# %%
# ── lgd_predictions_sample.csv (optional) ────────────────────────────────────
sample_idx = pd.Series(common_idx).sample(min(5000, len(common_idx)), random_state=42).values
lgd_predictions_sample = pd.DataFrame({
    'loan_id':        sample_idx,
    'actual_lgd':     lgd_actual.loc[sample_idx].values,
    'pred_lgd_m1':    lgd_pred_2stage.loc[sample_idx].values,
    'pred_lgd_m2':    lgd_pred_direct.loc[sample_idx].values,
    'grade':          loan_data_defaults.loc[sample_idx, 'grade'].values,
    'term_int':       loan_data_defaults.loc[sample_idx, 'term_int'].values,
})
lgd_predictions_sample.to_csv('ifrs9_app/data/lgd_predictions_sample.csv', index=False)
print(f"✓ data/lgd_predictions_sample.csv — {len(lgd_predictions_sample):,} rows (sample)")

print("\n=== All Page 4 data files saved to data/ ===")
print("  Required : lgd_summary.pkl, lgd_segments.csv, lgd_head_to_head.csv")
print("  Optional : lgd_stage1_coefficients.csv, lgd_stage2_coefficients.csv,")
print("             lgd_predictions_sample.csv")
# %%
import os
os.makedirs('data', exist_ok=True)

# ── ead_summary.csv ───────────────────────────────────────────────────────────
ead_summary_csv = pd.DataFrame([
    {
        'split':       'train',
        'n':           len(X_train_ead),
        'mean_actual': metrics_ead_train['mean_actual'],
        'mean_pred':   metrics_ead_train['mean_predicted'],
        'mae':         metrics_ead_train['mae'],
        'bias':        metrics_ead_train['bias'],
        'correlation': metrics_ead_train['correlation'],
        'rmse':        metrics_ead_train['rmse'],
        'r2':          metrics_ead_train['r2'],
    },
    {
        'split':       'test',
        'n':           len(X_test_ead),
        'mean_actual': metrics_ead_test['mean_actual'],
        'mean_pred':   metrics_ead_test['mean_predicted'],
        'mae':         metrics_ead_test['mae'],
        'bias':        metrics_ead_test['bias'],
        'correlation': metrics_ead_test['correlation'],
        'rmse':        metrics_ead_test['rmse'],
        'r2':          metrics_ead_test['r2'],
    },
])
ead_summary_csv.to_csv('ifrs9_app/data/ead_summary.csv', index=False)
print(f"✓ data/ead_summary.csv saved")

# ── ead_recalibration_results.csv ─────────────────────────────────────────────
ead_recal_csv = pd.DataFrame([
    {
        'model':       'Original',
        'split':       'train',
        'intercept':   round(intercept_orig, 6),
        'mae':         metrics_ead_train['mae'],
        'bias':        metrics_ead_train['bias'],
        'correlation': metrics_ead_train['correlation'],
        'rmse':        metrics_ead_train['rmse'],
        'r2':          metrics_ead_train['r2'],
        'mean_pred':   metrics_ead_train['mean_predicted'],
        'mean_actual': metrics_ead_train['mean_actual'],
    },
    {
        'model':       'Original',
        'split':       'test',
        'intercept':   round(intercept_orig, 6),
        'mae':         metrics_ead_test['mae'],
        'bias':        metrics_ead_test['bias'],
        'correlation': metrics_ead_test['correlation'],
        'rmse':        metrics_ead_test['rmse'],
        'r2':          metrics_ead_test['r2'],
        'mean_pred':   metrics_ead_test['mean_predicted'],
        'mean_actual': metrics_ead_test['mean_actual'],
    },
    {
        'model':       'Recalibrated',
        'split':       'train',
        'intercept':   round(intercept_new, 6),
        'mae':         metrics_ead_train_recal['mae'],
        'bias':        metrics_ead_train_recal['bias'],
        'correlation': metrics_ead_train_recal['correlation'],
        'rmse':        metrics_ead_train_recal['rmse'],
        'r2':          metrics_ead_train_recal['r2'],
        'mean_pred':   metrics_ead_train_recal['mean_predicted'],
        'mean_actual': metrics_ead_train_recal['mean_actual'],
    },
    {
        'model':       'Recalibrated',
        'split':       'test',
        'intercept':   round(intercept_new, 6),
        'mae':         metrics_ead_test_recal['mae'],
        'bias':        metrics_ead_test_recal['bias'],
        'correlation': metrics_ead_test_recal['correlation'],
        'rmse':        metrics_ead_test_recal['rmse'],
        'r2':          metrics_ead_test_recal['r2'],
        'mean_pred':   metrics_ead_test_recal['mean_predicted'],
        'mean_actual': metrics_ead_test_recal['mean_actual'],
    },
])
ead_recal_csv.to_csv('ifrs9_app/data/ead_recalibration_results.csv', index=False)
print(f"✓ data/ead_recalibration_results.csv saved")

# ── ead_ols_coefficients.csv (optional) ──────────────────────────────────────
ead_coef = pd.DataFrame({
    'feature':     X_train_ead.columns,
    'coefficient': ead_result.params.values,
    'p_value':     ead_result.pvalues.round(4).values,
    'significant': ead_result.pvalues.values < 0.05,
})
ead_coef.to_csv('ifrs9_app/data/ead_ols_coefficients.csv', index=False)
print(f"✓ data/ead_ols_coefficients.csv — {len(ead_coef)} features")
# %%
