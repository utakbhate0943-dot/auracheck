"""
In-depth EDA script for `students_mental_health_survey_augmented_10000.csv`.
Generates saved outputs under `outputs/` including figures and cleaned CSV.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style='whitegrid')

# Toggle generation of PCA outputs
SKIP_PCA = True

def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "Dataset" / "students_mental_health_survey.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate repository root containing Dataset/students_mental_health_survey.csv")


ROOT = find_repo_root()
aug_candidates = [
    ROOT / 'Dataset' / 'students_mental_health_survey_augmented_10000_with_burnout.csv',
    ROOT / 'Dataset' / 'students_mental_health_survey_augmented_10000.csv',
    ROOT / 'Dataset' / 'students_mental_health_survey_with_burnout_final.csv',
    ROOT / 'Dataset' / 'students_mental_health_survey.csv',
]
AUG_PATH = str(next((p for p in aug_candidates if p.exists()), aug_candidates[0]))
OUT_DIR = os.path.join(str(ROOT), 'EDA', 'Edgar', 'outputs', 'augmented_deep')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load
print('Loading dataset:', os.path.relpath(AUG_PATH, str(ROOT)))
df = pd.read_csv(AUG_PATH, low_memory=False)
print('Shape:', df.shape)

# Basic head/info/describe
df_head = df.head(10)
df_info = df.dtypes
num_desc = df.describe(include=[np.number]).T
cat_desc = df.describe(include=['object']).T

# Save small text summary
with open(os.path.join(OUT_DIR, 'summary_text.txt'), 'w') as f:
    f.write(f'Path: {os.path.relpath(AUG_PATH, str(ROOT))}\n')
    f.write(f'Shape: {df.shape}\n\n')
    f.write('Numeric describe:\n')
    f.write(num_desc.to_string())
    f.write('\n\nCategorical describe:\n')
    f.write(cat_desc.to_string())

# Missing values summary
miss = df.isnull().sum().sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(3)
miss_df = pd.concat([miss, miss_pct], axis=1)
miss_df.columns = ['missing_count', 'missing_pct']
miss_df.to_csv(os.path.join(OUT_DIR, 'missing_summary.csv'))

# Data types and unique counts
dtype_counts = pd.DataFrame({'dtype': df.dtypes.astype(str)})
unique_counts = pd.Series({c: df[c].nunique(dropna=False) for c in df.columns})
unique_counts.to_csv(os.path.join(OUT_DIR, 'unique_counts.csv'))

# Convert obvious numeric columns
numeric_candidates = []
for c in df.columns:
    if df[c].dtype == 'object':
        sample = df[c].dropna().astype(str).sample(min(200, max(1, df[c].dropna().shape[0])))
        num_like = pd.to_numeric(sample.str.replace(',', '').str.strip(), errors='coerce')
        if num_like.notnull().mean() > 0.6:
            try:
                df[c] = pd.to_numeric(df[c].str.replace(',', '').str.strip(), errors='coerce')
                numeric_candidates.append(c)
            except Exception:
                pass

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# Save lists
pd.Series(num_cols).to_csv(os.path.join(OUT_DIR, 'numeric_columns.csv'), index=False)
pd.Series(cat_cols).to_csv(os.path.join(OUT_DIR, 'categorical_columns.csv'), index=False)

# Univariate numeric plots
for c in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[c].dropna(), kde=True, bins=30)
    plt.title(f'Histogram: {c}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'hist_{c}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[c].dropna())
    plt.title(f'Boxplot: {c}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'box_{c}.png'), dpi=150)
    plt.close()

# Univariate categorical plots (top categories)
for c in cat_cols:
    try:
        vc = df[c].value_counts(dropna=False).iloc[:20]
        plt.figure(figsize=(6, max(3, 0.25*len(vc))))
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f'Counts: {c}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'count_{c}.png'), dpi=150)
        plt.close()
    except Exception:
        pass

# Correlation matrix for numeric columns (improved layout)
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    n = len(num_cols)
    # scale figure size with number of numeric columns
    figsize = (max(8, 0.6 * n), max(6, 0.5 * n))
    plt.figure(figsize=figsize)
    # dynamic annot size: smaller font for many features
    annot_size = int(max(6, min(12, 180 // max(1, n))))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        annot_kws={'size': annot_size},
        linewidths=0.5,
        cbar_kws={'shrink': 0.7}
    )
    plt.title('Correlation matrix (numeric)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'corr_matrix.png'), dpi=200)
    plt.close()

# Pairplot (sample)
sample_for_pair = df[num_cols].dropna().sample(min(1000, max(10, len(df[num_cols].dropna())))) if len(num_cols) > 1 else None
if sample_for_pair is not None and sample_for_pair.shape[1] <= 6:
    sns.pairplot(sample_for_pair, diag_kind='kde', corner=True)
    plt.savefig(os.path.join(FIG_DIR, 'pairplot_sample.png'), dpi=150)
    plt.close()

# Group comparisons: if gender-like column exists
gender_col = None
for cand in ['Gender', 'gender', 'Sex', 'sex']:
    if cand in df.columns:
        gender_col = cand
        break

stats_summary = []
if gender_col is not None:
    groups = df[gender_col].dropna().unique()
    for col in ['Depression_Score', 'Anxiety_Score', 'Stress_Level']:
        if col in df.columns and col in num_cols:
            vals = {g: df.loc[df[gender_col] == g, col].dropna().values for g in groups}
            if len(groups) >= 2:
                # only compare first two groups with t-test for quick checks
                g0, g1 = groups[:2]
                try:
                    tstat, pval = stats.ttest_ind(vals[g0], vals[g1], equal_var=False, nan_policy='omit')
                except Exception:
                    tstat, pval = np.nan, np.nan
                stats_summary.append({'metric': col, 'group0': g0, 'group1': g1, 'tstat': float(tstat) if not np.isnan(tstat) else None, 'pval': float(pval) if not np.isnan(pval) else None})

# Chi-square example: Counseling_Service_Use vs gender
chi_summary = None
if gender_col is not None and 'Counseling_Service_Use' in df.columns:
    ct = pd.crosstab(df[gender_col], df['Counseling_Service_Use'])
    try:
        chi2, p, dof, ex = stats.chi2_contingency(ct.fillna(0))
        chi_summary = {'chi2': float(chi2), 'pval': float(p), 'dof': int(dof)}
        ct.to_csv(os.path.join(OUT_DIR, 'crosstab_gender_counseling.csv'))
    except Exception:
        chi_summary = None

# PCA on numeric features (after dropping NaNs rows subset)
# Disabled when SKIP_PCA is True to avoid producing PCA CSVs/images
pca_result_path = os.path.join(OUT_DIR, 'pca_components.csv')
if not SKIP_PCA and len(num_cols) >= 2:
    num_df = df[num_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)
    pca = PCA(n_components=min(5, scaled.shape[1]))
    comps = pca.fit_transform(scaled)
    pcs = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(comps.shape[1])])
    pcs.to_csv(pca_result_path, index=False)
    # plot first two components if present
    if pcs.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        plt.scatter(pcs['PC1'], pcs['PC2'], s=8, alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: PC1 vs PC2 (numeric features)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'pca_pc1_pc2.png'), dpi=150)
        plt.close()

# Outlier detection: flag rows with any numeric outside 1.5 IQR
outlier_flags = pd.DataFrame(index=df.index)
for c in num_cols:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df[c] < (q1 - 1.5 * iqr)) | (df[c] > (q3 + 1.5 * iqr))
    outlier_flags[f'outlier_{c}'] = outlier_mask
outlier_flags['any_outlier'] = outlier_flags.any(axis=1)
outlier_flags['any_outlier'] = outlier_flags.any(axis=1)
outlier_flags['any_outlier'].to_csv(os.path.join(OUT_DIR, 'outlier_flags.csv'))

# Save cleaned full augmented CSV (non-destructive basic cleaning)
cleaned = df.copy()
# Example basic imputation for small missing CGPA: fill with median
if 'CGPA' in cleaned.columns:
    cleaned['CGPA'] = pd.to_numeric(cleaned['CGPA'], errors='coerce')
    cleaned['CGPA'] = cleaned['CGPA'].fillna(cleaned['CGPA'].median())

cleaned_path = os.path.join(OUT_DIR, 'students_mental_health_survey_augmented_10000_cleaned.csv')
cleaned.to_csv(cleaned_path, index=False)

# Write stats summary JSON/text
import json
out_stats = {'shape': df.shape, 'num_columns': len(num_cols), 'cat_columns': len(cat_cols), 'numeric_sample_describe': num_desc.to_dict(), 'ttest_summary': stats_summary, 'chi2_summary': chi_summary}
with open(os.path.join(OUT_DIR, 'analysis_summary.json'), 'w') as f:
    json.dump(out_stats, f, indent=2, default=str)

print('EDA complete. Outputs saved under:', os.path.relpath(OUT_DIR, str(ROOT)))
