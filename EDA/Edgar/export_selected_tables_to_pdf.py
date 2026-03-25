#!/usr/bin/env python3
"""
Render selected tables (numeric describe and column lists) from the two survey CSVs
into a paginated PDF suitable for review. Saves to `outputs/selected_tables.pdf`.
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.expanduser('~/Desktop/ML')
ORIG = os.path.join(BASE, 'students_mental_health_survey_original.csv')
AUG = os.path.join(BASE, 'students_mental_health_survey_augmented_10000.csv')
OUT_DIR = os.path.join(BASE, 'outputs')
PDF_OUT = os.path.join(OUT_DIR, 'selected_tables.pdf')

os.makedirs(OUT_DIR, exist_ok=True)

def safe_read(path):
    if not os.path.exists(path):
        print('Missing file:', path)
        return None
    return pd.read_csv(path, low_memory=False)

fmt_float = lambda x: ("{:.3f}".format(x) if (isinstance(x, (int, float, np.floating, np.integer)) and not pd.isna(x)) else ("NaN" if pd.isna(x) else str(x)))

def render_dataframe_pages(pdf, df, title, rows_per_page=20):
    if df is None or df.shape[0] == 0:
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, f"{title}\n(No rows)", ha='center', va='center')
        pdf.savefig(fig); plt.close()
        return

    nrows = df.shape[0]
    cols = df.columns.tolist()
    for start in range(0, nrows, rows_per_page):
        sub = df.iloc[start:start+rows_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title(f"{title} — rows {start+1}-{min(start+rows_per_page, nrows)}", fontsize=12, pad=12)

        cell_text = []
        for row in sub.itertuples(index=False):
            cell_text.append([fmt_float(x) for x in row])

        table = ax.table(cellText=cell_text,
                         colLabels=sub.columns.tolist(),
                         rowLabels=sub.index.astype(str).tolist(),
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def render_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.title(title, fontsize=14, pad=12)
    # print lines with a fixed vertical spacing
    y = 0.9
    for line in lines:
        plt.text(0.02, y, line, fontsize=10, va='top')
        y -= 0.035
        if y < 0.05:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            y = 0.9
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    df_orig = safe_read(ORIG)
    df_aug = safe_read(AUG)

    if df_orig is None and df_aug is None:
        print('No input files found. Exiting.')
        return

    # Compute numeric describe tables (transposed) like in the notebook
    desc_orig = df_orig.select_dtypes(include=[np.number]).describe().T if df_orig is not None else None
    desc_aug = df_aug.select_dtypes(include=[np.number]).describe().T if df_aug is not None else None

    cols_orig = list(df_orig.columns) if df_orig is not None else []
    cols_aug = list(df_aug.columns) if df_aug is not None else []

    def missing_summary(df):
        if df is None:
            return None
        miss = df.isnull().sum()
        pct = (miss / max(1, len(df))) * 100
        out = pd.DataFrame({'missing_count': miss, 'missing_pct': pct})
        return out.sort_values('missing_pct', ascending=False)

    miss_orig = missing_summary(df_orig)
    miss_aug = missing_summary(df_aug)

    with PdfPages(PDF_OUT) as pdf:
        # title page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.66, 'Comparing Original vs Augmented Datasets', ha='center', fontsize=18, weight='bold')
        plt.text(0.5, 0.58, 'Selected tables for Student Mental Health EDA', ha='center', fontsize=12)
        plt.text(0.5, 0.48, 'Includes: numeric describe() (transposed) and missing-value summaries for each dataset', ha='center', fontsize=10)
        pdf.savefig(fig); plt.close()

        # original describe
        render_dataframe_pages(pdf, desc_orig, 'Original dataset: numeric describe().T', rows_per_page=20)

        # augmented describe
        render_dataframe_pages(pdf, desc_aug, 'Augmented dataset: numeric describe().T', rows_per_page=20)

        # missing-value summaries
        render_dataframe_pages(pdf, miss_orig, 'Original dataset: Missing values summary', rows_per_page=30)
        render_dataframe_pages(pdf, miss_aug, 'Augmented dataset: Missing values summary', rows_per_page=30)

    print('Saved PDF to', PDF_OUT)

if __name__ == '__main__':
    main()
