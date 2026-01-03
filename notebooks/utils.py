import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def stringify_consecutive_numbers(numbers: list[int]) -> str:
    runs = []
    _lst = []
    for _n in sorted(numbers):
        if not _lst:
            _lst.append(_n)
        elif _n == _lst[-1] + 1:
            _lst.append(_n)
        else:
            runs.append(_lst)
            _lst = [_n]
    runs.append(_lst)

    if len(runs) == 0:
        return "No numbers"
    
    strings = []
    for _lst in runs:
        if len(_lst) == 1:
            strings.append(f"{_lst[0]}")
        else:
            strings.append(f"{_lst[0]}-{_lst[-1]}")
    
    return ", ".join(strings)

def disp_columns(df: pd.DataFrame, title: str = None):
    dtype_dict = dict(df.dtypes)
    columns = dict()
    columns = dict()
    for _k, _v in dtype_dict.items():
        m = re.match(r"(.*[^\d])(\d+)", _k)
        if not m:
            columns[_k] = str(_v)
            continue
        else:
            _new_key = m.group(1) + "###"
            if _new_key not in columns:
                columns[_new_key] = defaultdict(list)
            columns[_new_key][str(_v)].append(int(m.group(2)))
            
    lines = []
    if title:
        lines.append(title)
        lines.append("\u203E"*len(title))
        
    for _k, _v in columns.items():
        if isinstance(_v, str):
            lines.append(f"- {_k}: {_v}")
        else:
            _k = re.sub(r"\#\#\#$", "", _k)
            if len(_v) == 1:
                _dtype = list(_v.keys())[0]
                _indices = _v[_dtype]
                lines.append(f"- {_k}<{stringify_consecutive_numbers(_indices)}>: {_dtype}")
            else:
                lines.append(f"- {_k}<N>")
                for _dtype, _indices in _v.items():
                    lines.append(f"  - N = {stringify_consecutive_numbers(_indices)}: {_dtype}")
    
    max_line_len = max(len(_l) for _l in lines) + 1
    
    print("\u250C" + "\u2500"*(max_line_len+1) + "\u2510")
    for _line in lines:
        print(f"\u2502 {_line:<{max_line_len}}\u2502")
    print("\u2514" + "\u2500"*(max_line_len+1) + "\u2518\n")
    
    return

def disp_distributions(dataframe: pd.DataFrame, exclude_cols=None):
    """Display table-style distribution plots for all columns in a DataFrame."""

    columns = dict(dataframe.dtypes)

    if exclude_cols:
        for col in exclude_cols:
            if col in columns:
                columns.pop(col)

    _nrows = len(columns) + 1  # +1 for header row
    _ncols = 4  # Column, # Unique, % NaN, Distribution

    fig, axes = plt.subplots(_nrows, _ncols, figsize=(10, 0.4 * _nrows),
                              gridspec_kw={'width_ratios': [2, 1, 1, 4]})

    # Handle single data row case
    if len(columns) == 1:
        axes = axes.reshape(2, 4)

    # Header row
    headers = ['Column', '# Unique', '% NaN', 'Distribution']
    for j, header in enumerate(headers):
        ax = axes[0, j]
        ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Data rows
    for i, (_col, _type) in enumerate(columns.items()):
        row = i + 1  # offset for header
        _series = dataframe[_col]
        n_unique = _series.nunique()
        pct_nan = _series.isna().mean() * 100

        # Column name
        axes[row, 0].text(0.5, 0.5, _col, ha='center', va='center', fontsize=9)
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_ylim(0, 1)
        axes[row, 0].axis('off')

        # # Unique
        axes[row, 1].text(0.5, 0.5, str(n_unique), ha='center', va='center', fontsize=9)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].axis('off')

        # % NaN
        axes[row, 2].text(0.5, 0.5, f"{pct_nan:.1f}%", ha='center', va='center', fontsize=9)
        axes[row, 2].set_xlim(0, 1)
        axes[row, 2].set_ylim(0, 1)
        axes[row, 2].axis('off')

        # Distribution plot
        ax = axes[row, 3]
        if _type == "object":
            # Categorical: proportional horizontal stacked bar
            counts = _series.value_counts(normalize=True)
            left = 0
            colors = plt.cm.tab10.colors
            for j, (val, prop) in enumerate(counts.items()):
                ax.barh(0, prop, left=left, height=0.8,
                       color=colors[j % len(colors)], edgecolor='none')
                left += prop
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
        else:
            # Numerical: bar plot if few unique values, otherwise histogram
            data = _series.dropna()
            if len(data) > 0:
                unique_vals = data.nunique()
                if unique_vals < 50:
                    counts = data.value_counts().sort_index()
                    heights = counts.values / counts.values.max()
                    heights = np.maximum(heights, 0.05)  # minimum visibility
                    ax.bar(counts.index, heights, color='steelblue', edgecolor='none')
                else:
                    hist_counts, bin_edges = np.histogram(data, bins=50)
                    heights = hist_counts / hist_counts.max()
                    heights = np.where(hist_counts > 0, np.maximum(heights, 0.05), 0)
                    bin_widths = bin_edges[1] - bin_edges[0]
                    ax.bar(bin_edges[:-1], heights, width=bin_widths, align='edge',
                           color='steelblue', edgecolor='none')
                ax.set_xlim(data.min(), data.max())
                ax.set_ylim(0, 1.05)

        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()
    