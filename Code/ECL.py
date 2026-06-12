# %% ===========================================================================
# IFRS 9 LIFETIME ECL — STEP-BY-STEP IMPLEMENTATION
# =============================================================================
%reset -f

# %% ===========================================================================
# 1. IMPORTS
# =============================================================================
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:.4f}'.format
sns.set()
os.makedirs('data', exist_ok=True)

# %% ===========================================================================
# 2. LOAD DATA
# =============================================================================
# Dataset : LendingClub Loan Data (2007–2018 Q4)
# Source  : LendingClub public dataset (~2.2M accepted loan applications)
# Period  : January 2007 – December 2018
# -----------------------------------------------------------------------------
cols_needed = [
    'id',
    # Target
    'loan_status', 'total_rec_prncp', 'recoveries',
    # CHARACTER
    'grade', 'inq_last_6mths', 'earliest_cr_line', 'verification_status',
    # CAPACITY
    'annual_inc', 'dti', 'installment',
    # CAPITAL
    'total_rev_hi_lim', 'home_ownership', 'funded_amnt',
    # CONDITIONS
    'int_rate', 'term', 'purpose', 'issue_d', 'addr_state',
]

loan_data = pd.read_csv(
    'ifrs9_app/data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018q4.csv',
    usecols=cols_needed
)
del cols_needed
print(f"Raw dataset shape: {loan_data.shape}")

# ── Clean ID & set index ──────────────────────────────────────────────────────
loan_data = loan_data[pd.to_numeric(loan_data['id'], errors='coerce').notna()]
loan_data['id'] = loan_data['id'].astype(int)
loan_data = loan_data.set_index('id', drop=False)
loan_data.index.name = 'id'
print(f"Total loans: {len(loan_data):,}")

# ── Filter to resolved loans only ─────────────────────────────────────────────
finished_statuses = [
    'Fully Paid',
    'Charged Off',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
]
loan_data = loan_data[loan_data['loan_status'].isin(finished_statuses)]
print(f"Loans after status filter: {len(loan_data):,}")
print(loan_data['loan_status'].value_counts())

# %% ===========================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

# ── earliest_cr_line → months since ──────────────────────────────────────────
loan_data['earliest_cr_line_date'] = pd.to_datetime(
    loan_data['earliest_cr_line'], format='%b-%Y')
loan_data['mths_since_earliest_cr_line'] = round(
    pd.to_numeric(
        (pd.to_datetime('2026-01-01') - loan_data['earliest_cr_line_date'])
        / np.timedelta64(1, 'D')
    ) / 30.5
).fillna(0)
loan_data.drop(columns=['earliest_cr_line_date'], inplace=True)

# ── term → numeric ────────────────────────────────────────────────────────────
loan_data['term_int'] = pd.to_numeric(
    loan_data['term'].str.strip().str.replace(' months', '', regex=False))
loan_data.drop(columns=['term'], inplace=True)

# ── issue_d → months since ────────────────────────────────────────────────────
loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%Y')
loan_data['mths_since_issue_d'] = round(
    pd.to_numeric(
        (pd.to_datetime('2026-01-01') - loan_data['issue_d_date'])
        / np.timedelta64(1, 'D')
    ) / 30.5
)
loan_data.drop(columns=['issue_d', 'issue_d_date'], inplace=True)

# %% ===========================================================================
# 4. MISSING VALUE TREATMENT
# =============================================================================
loan_data['total_rev_hi_lim'] = loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'])
loan_data['annual_inc']       = loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean())
for col in ['mths_since_earliest_cr_line', 'inq_last_6mths']:
    loan_data[col] = loan_data[col].fillna(0)

# %% ===========================================================================
# 5. SURVIVAL SETUP — EVENT & DURATION
# =============================================================================
def months_to_default(funded_amnt, int_rate, term, total_rec_prncp, is_default):
    """
    Duration (months):
    - Non-default : full term (censored at repayment)
    - Default     : month when cumulative principal >= total_rec_prncp
    """
    if is_default == 0:
        return int(term)
    if total_rec_prncp <= 0:
        return 1

    monthly_rate = (int_rate / 100) / 12

    if monthly_rate == 0:
        monthly_principal = funded_amnt / term
        return min(int(term), int(np.ceil(total_rec_prncp / monthly_principal)))

    payment = funded_amnt * monthly_rate / (1 - (1 + monthly_rate) ** (-term))
    balance, cumulative_principal = funded_amnt, 0

    for month in range(1, int(term) + 1):
        interest              = balance * monthly_rate
        principal             = payment - interest
        cumulative_principal += principal
        balance              -= principal
        if cumulative_principal >= total_rec_prncp:
            return month

    return int(term)

# ── Apply ─────────────────────────────────────────────────────────────────────
loan_data['event'] = np.where(
    loan_data['loan_status'].isin([
        'Charged Off',
        'Does not meet the credit policy. Status:Charged Off'
    ]), 1, 0)

loan_data['duration'] = loan_data.apply(
    lambda row: months_to_default(
        funded_amnt    =row['funded_amnt'],
        int_rate       =row['int_rate'],
        term           =row['term_int'],
        total_rec_prncp=row['total_rec_prncp'],
        is_default     =row['event']
    ), axis=1)

print(f"Defaulted     : {loan_data['event'].sum():,}")
print(f"Non-defaulted : {(loan_data['event'] == 0).sum():,}")
print(f"Mean duration — default  : {loan_data[loan_data['event']==1]['duration'].mean():.1f} months")
print(f"Mean duration — paid off : {loan_data[loan_data['event']==0]['duration'].mean():.1f} months")

# %% ===========================================================================
# 6. GRADE BUCKETS & SEGMENTS
# =============================================================================
def grade_bucket(grade):
    if   grade == 'A': return 'A'
    elif grade == 'B': return 'B'
    elif grade == 'C': return 'C'
    elif grade == 'D': return 'D'
    else:              return 'E-G'

loan_data['grade_bucket'] = loan_data['grade'].apply(grade_bucket)
loan_data['segment']      = loan_data['term_int'].astype(str) + '_' + loan_data['grade_bucket']

print(loan_data['segment'].value_counts())

# %% ===========================================================================
# 7. DEFAULT TIMING WEIGHTS
# =============================================================================
def compute_timing_weights(df):
    """
    w_t = defaults_in_month_t / total_defaults_in_segment
    Uses defaulted loans only.
    """
    defaults       = df[df['event'] == 1].copy()
    timing_library = []

    for segment, grp in defaults.groupby('segment'):
        total_defaults = len(grp)
        monthly_counts = grp['duration'].value_counts().sort_index()
        for month, count in monthly_counts.items():
            timing_library.append({
                'segment': segment,
                'month':   month,
                'count':   count,
                'weight':  count / total_defaults,
            })

    df_timing = pd.DataFrame(timing_library)

    check = df_timing.groupby('segment')['weight'].sum().round(4)
    print("Weight sum check (all should = 1.0):")
    print(check)
    return df_timing

df_timing = compute_timing_weights(loan_data)

# ── Plot ──────────────────────────────────────────────────────────────────────
segments = sorted(df_timing['segment'].unique())
ncols    = 3
nrows    = int(np.ceil(len(segments) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()

for i, segment in enumerate(segments):
    seg_data = df_timing[df_timing['segment'] == segment]
    axes[i].bar(seg_data['month'], seg_data['weight'],
                color='steelblue', edgecolor='white')
    axes[i].set_title(f'Segment: {segment}')
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('w_t')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Default Timing Curves by Segment', fontsize=14)
plt.tight_layout()
plt.show()

# ── Save ──────────────────────────────────────────────────────────────────────
# Rename 'weight' → 'w_t' so the Streamlit app can read it without KeyError
df_timing_save = df_timing.rename(columns={'weight': 'w_t'})
df_timing_save.to_csv('ifrs9_app/data/timing_curves.csv', index=False)
print(f"✓ timing_curves.csv — {len(df_timing_save)} rows, {df_timing_save['segment'].nunique()} segments")

# %% ===========================================================================
# 8. KAPLAN-MEIER SURVIVAL CURVES
# =============================================================================
def kaplan_meier(durations, events):
    df  = pd.DataFrame({'duration': durations, 'event': events})
    S   = 1.0
    rows = []
    for t in sorted(df['duration'].unique()):
        n_at_risk = (df['duration'] >= t).sum()
        d         = ((df['duration'] == t) & (df['event'] == 1)).sum()
        if n_at_risk > 0:
            S *= (1 - d / n_at_risk)
        rows.append({'timeline': t, 'survival_prob': S,
                     'n_at_risk': n_at_risk, 'n_events': d})
    return pd.DataFrame(rows)

km_records = []
for segment, grp in loan_data.groupby('segment'):
    km_df            = kaplan_meier(grp['duration'], grp['event'])
    km_df['segment'] = segment
    km_records.append(km_df)

df_km = pd.concat(km_records, ignore_index=True)
df_km = df_km[['segment', 'timeline', 'survival_prob', 'n_at_risk', 'n_events']]


# %% ===========================================================================
# 9. MONTHLY PD SCHEDULE
# =============================================================================
# ── Load PD_life ──────────────────────────────────────────────────────────────
df_pd_all = pd.read_csv('pd_life_all.csv')
df_pd_all['id'] = df_pd_all['id'].astype(int)
df_pd_all = df_pd_all.set_index('id')

# ── Validate ID overlap ───────────────────────────────────────────────────────
ids_pd   = set(df_pd_all.index)
ids_loan = set(loan_data.index)
print(f"IDs in both               : {len(ids_pd & ids_loan):,}")
print(f"IDs in pd_all but NOT loan: {len(ids_pd - ids_loan):,}")
print(f"IDs in loan but NOT pd_all: {len(ids_loan - ids_pd):,}")

# ── Attach segment ────────────────────────────────────────────────────────────
df_pd_all['segment'] = loan_data['segment']
print(f"Missing segments: {df_pd_all['segment'].isna().sum()}")

# ── Convert P(good) → P(default) ─────────────────────────────────────────────
# The PD model was trained with good_bad=1 (good) / 0 (bad), so result.predict()
# returns P(good). ECL requires P(default) = 1 − P(good).
df_pd_all['PD_default'] = 1 - df_pd_all['PD_life']
print(f"PD_default mean: {df_pd_all['PD_default'].mean():.4%}  "
      f"(PD_life was {df_pd_all['PD_life'].mean():.4%} — P(good))")

# ── Vectorized PD_t = PD_default × w_t ───────────────────────────────────────
df_pd_monthly = (
    df_pd_all.reset_index()
    .rename(columns={'id': 'loan_id'})
    .merge(df_timing[['segment', 'month', 'weight']], on='segment', how='left')
)
df_pd_monthly['PD_t'] = df_pd_monthly['PD_default'] * df_pd_monthly['weight']
df_pd_monthly = df_pd_monthly.sort_values(['loan_id', 'month']).reset_index(drop=True)


# %% ===========================================================================
# 10. EAD SCHEDULE (VECTORIZED AMORTISATION)
# =============================================================================
# EAD_t = P × [(1+r)^n − (1+r)^(t−1)] / [(1+r)^n − 1]
# For r = 0: EAD_t = P × (n − t + 1) / n
# =============================================================================
def build_portfolio_ead_vectorized(loan_features):
    records = []
    for term in loan_features['term_int'].unique():
        term    = int(term)
        grp     = loan_features[loan_features['term_int'] == term].copy()
        months  = np.arange(1, term + 1)

        P = grp['funded_amnt'].values[:, np.newaxis]
        r = ((grp['int_rate'].values / 100) / 12)[:, np.newaxis]
        t = months[np.newaxis, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            ead_matrix = np.where(
                r == 0,
                P * (term - t + 1) / term,
                P * ((1 + r)**term - (1 + r)**(t - 1)) / ((1 + r)**term - 1)
            )
        ead_matrix = np.clip(ead_matrix, 0, None).round(2)

        records.append(pd.DataFrame({
            'loan_id':     np.repeat(grp.index.values, term),
            'month':       np.tile(months, len(grp)),
            'EAD':         ead_matrix.flatten(),
            'segment':     np.repeat(grp['segment'].values, term),
            'funded_amnt': np.repeat(grp['funded_amnt'].values, term),
        }))
    return pd.concat(records, ignore_index=True)

# ── Save loan_attrs before slicing loan_data ──────────────────────────────────
loan_attrs = loan_data[['grade', 'grade_bucket', 'term_int',
                         'funded_amnt', 'int_rate', 'segment']].copy()

start            = time.time()
df_ead_schedule  = build_portfolio_ead_vectorized(loan_attrs)
print(f"EAD schedule built in {time.time() - start:.1f}s — shape: {df_ead_schedule.shape}")

# %% ===========================================================================
# 11. ECL CALCULATION
# =============================================================================
# ECL_t = PD_t × LGD × EAD_t
# =============================================================================
df_ecl = df_ead_schedule.merge(
    df_pd_monthly[['loan_id', 'month', 'PD_t', 'PD_default']],
    on=['loan_id', 'month'],
    how='left'
)

df_ecl['LGD']   = 0.85
df_ecl['ECL_t'] = df_ecl['PD_t'] * df_ecl['LGD'] * df_ecl['EAD']

ecl_per_loan = df_ecl.groupby('loan_id')['ECL_t'].sum().rename('ECL_lifetime')

print(f"Total portfolio ECL  : ${ecl_per_loan.sum():,.2f}")
print(f"Average ECL per loan : ${ecl_per_loan.mean():,.2f}")
print(ecl_per_loan.describe())

# %% ===========================================================================
# 12. SAVE OUTPUTS
# =============================================================================

# ── Enrich ecl_per_loan with loan attributes ──────────────────────────────────
ecl_per_loan = ecl_per_loan.reset_index().rename(columns={'loan_id': 'loan_id'})
ecl_per_loan['loan_id'] = ecl_per_loan['loan_id'].astype(int)
ecl_per_loan = (
    ecl_per_loan
    .merge(loan_attrs[['grade', 'term_int', 'funded_amnt', 'int_rate']],
           left_on='loan_id', right_index=True, how='left')
    .merge(df_pd_all[['PD_life', 'PD_default']], left_on='loan_id', right_index=True, how='left')
    .rename(columns={'funded_amnt': 'exposure', 'term_int': 'term'})
)
# PD_life = P(good), PD_default = P(default) — use PD_default for all ECL metrics
print(f"ECL per-loan columns: {ecl_per_loan.columns.tolist()}")

# ── Stage 1 (12-month) vs Stage 2 (lifetime) ECL ─────────────────────────────
# Stage 1 = sum of ECL_t for months 1–12 only
# Stage 2 = lifetime ECL (all months) — already in ECL_lifetime
ecl_stage1_ser = (
    df_ecl[df_ecl['month'] <= 12]
    .groupby('loan_id')['ECL_t']
    .sum()
    .rename('ECL_stage1')
)

ecl_per_loan = ecl_per_loan.merge(
    ecl_stage1_ser, on='loan_id', how='left'
)
ecl_per_loan['ECL_stage1'] = ecl_per_loan['ECL_stage1'].fillna(0)
ecl_per_loan['ECL_stage2'] = ecl_per_loan['ECL_lifetime']   # stage 2 = full lifetime

# ── ECL by grade (with stage1 / stage2) ───────────────────────────────────────
ecl_by_grade = ecl_per_loan.groupby('grade').agg(
    n_loans        = ('loan_id',      'count'),
    total_exposure = ('exposure',     'sum'),
    avg_pd         = ('PD_default',   'mean'),   # P(default), not P(good)
    ecl_stage1     = ('ECL_stage1',   'sum'),
    ecl_stage2     = ('ECL_stage2',   'sum'),
).reset_index()
ecl_by_grade['coverage_stage1'] = ecl_by_grade['ecl_stage1'] / ecl_by_grade['total_exposure']
ecl_by_grade['coverage_stage2'] = ecl_by_grade['ecl_stage2'] / ecl_by_grade['total_exposure']
ecl_by_grade.to_csv('ifrs9_app/data/ecl_by_grade.csv', index=False)
print(f"✓ data/ecl_by_grade.csv — {len(ecl_by_grade)} rows")

# ── ECL by term (with stage1 / stage2) ───────────────────────────────────────
ecl_by_term = ecl_per_loan.groupby('term').agg(
    n_loans        = ('loan_id',      'count'),
    total_exposure = ('exposure',     'sum'),
    ecl_stage1     = ('ECL_stage1',   'sum'),
    ecl_stage2     = ('ECL_stage2',   'sum'),
).reset_index()
ecl_by_term['coverage_stage1'] = ecl_by_term['ecl_stage1'] / ecl_by_term['total_exposure']
ecl_by_term['coverage_stage2'] = ecl_by_term['ecl_stage2'] / ecl_by_term['total_exposure']
ecl_by_term.to_csv('ifrs9_app/data/ecl_by_term.csv', index=False)
print(f"✓ data/ecl_by_term.csv — {len(ecl_by_term)} rows")

# ── ECL timeline (monthly aggregate across portfolio) ─────────────────────────
# future_month, total_ecl, n_loans_active
ecl_timeline = (
    df_ecl
    .groupby('month')
    .agg(
        total_ecl      = ('ECL_t',    'sum'),
        n_loans_active = ('loan_id',  'nunique'),
    )
    .reset_index()
    .rename(columns={'month': 'future_month'})
    .sort_values('future_month')
)
ecl_timeline.to_csv('ifrs9_app/data/ecl_timeline.csv', index=False)
print(f"✓ data/ecl_timeline.csv — {len(ecl_timeline)} rows "
      f"(months {ecl_timeline['future_month'].min()}–{ecl_timeline['future_month'].max()})")

# ── Portfolio summary (CSV with stage1 / stage2) ──────────────────────────────
total_exposure    = ecl_per_loan['exposure'].sum()
total_ecl_stage1  = ecl_per_loan['ECL_stage1'].sum()
total_ecl_stage2  = ecl_per_loan['ECL_stage2'].sum()

portfolio_summary = {
    'n_loans':          len(ecl_per_loan),
    'total_exposure':   total_exposure,
    'total_ecl_stage1': total_ecl_stage1,
    'total_ecl_stage2': total_ecl_stage2,
    'coverage_stage1':  total_ecl_stage1 / total_exposure,
    'coverage_stage2':  total_ecl_stage2 / total_exposure,
    'avg_pd_default':   ecl_per_loan['PD_default'].mean(),   # P(default)
}
pd.DataFrame([portfolio_summary]).to_csv('ifrs9_app/data/ecl_portfolio_summary.csv', index=False)
print(f"✓ data/ecl_portfolio_summary.csv saved")

print('\n=== Portfolio Summary ===')
print(f"  Loans            : {portfolio_summary['n_loans']:,}")
print(f"  Total exposure   : ${total_exposure:,.0f}")
print(f"  ECL Stage 1 (12m): ${total_ecl_stage1:,.0f}  ({total_ecl_stage1/total_exposure:.4%})")
print(f"  ECL Stage 2 (life): ${total_ecl_stage2:,.0f}  ({total_ecl_stage2/total_exposure:.4%})")
print(f"  Avg PD (default) : {portfolio_summary['avg_pd_default']:.4%}")

# ── Timing segments ───────────────────────────────────────────────────────────
timing_segments = (
    loan_attrs.groupby(['grade_bucket', 'term_int', 'segment'])
    .size()
    .reset_index(name='n_loans')
    .rename(columns={'grade_bucket': 'grade'})
    .sort_values(['segment', 'grade', 'term_int'])
    .reset_index(drop=True)
)
timing_segments.to_csv('ifrs9_app/data/timing_segments.csv', index=False)
print(f"✓ data/timing_segments.csv — {len(timing_segments)} rows")
print(timing_segments.to_string(index=False))

print('\n=== All files saved ===')
print('  timing_curves.csv                — default timing weights per segment/month (col: w_t)')
print('  data/ecl_by_grade.csv            — ECL by grade  (ecl_stage1, ecl_stage2)')
print('  data/ecl_by_term.csv             — ECL by term   (ecl_stage1, ecl_stage2)')
print('  data/ecl_timeline.csv            — monthly portfolio ECL (future_month, total_ecl, n_loans_active)')
print('  data/ecl_portfolio_summary.csv   — 1-row summary (n_loans, exposure, stage1, stage2)')
print('  data/timing_segments.csv         — grade × term segment mapping')
# %%
