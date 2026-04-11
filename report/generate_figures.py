"""
Generate all figures for the thesis.
Run from project root: python -m report.generate_figures
Output: report/figures/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
from pathlib import Path
from collections import Counter

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titlepad': 14,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS    = Path('results')
SCENES_DIR = Path('results/scene_graphs')
OUT        = Path('report/figures')
OUT.mkdir(parents=True, exist_ok=True)


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def load_per_question(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def savefig(name):
    plt.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close()
    print(f'  {name} saved.')


# ── Figure 1: GT tok accuracy across datasets ────────────────────
def fig_gt_accuracy():
    datasets = {
        'CLEVR': load_summary(RESULTS / 'clevr_val_gt_n1000_summary.json'),
        'CoGenT': load_summary(RESULTS / 'cogent_b_gt_n5000_summary.json'),
        'CLEVR-Humans': load_summary(RESULTS / 'clevr_humans_gt_summary.json'),
        'Super-CLEVR': load_summary(RESULTS / 'superclevr_gt_n5000_summary.json'),
    }

    labels = list(datasets.keys())
    accs   = [d['accuracy'] * 100 for d in datasets.values()]
    colors = ['#4C72B0', '#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color=colors, edgecolor='white', linewidth=1.2, width=0.5)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 118)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('GT Tok Accuracy by Dataset (Reasoning Only)', pad=16)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    savefig('fig1_gt_accuracy')


# ── Figure 2: GT vs Detected tok on CLEVR ────────────────────────
def fig_gt_vs_detected():
    gt  = load_summary(RESULTS / 'clevr_val_gt_n1000_summary.json')
    det = load_summary(RESULTS / 'clevr_detected_summary.json')

    categories     = ['count', 'exist', 'query', 'compare', 'other']
    labels_pretty  = ['Count', 'Exist', 'Query', 'Compare', 'Other']

    gt_vals  = [gt['accuracy_by_category'].get(c, 0) * 100  for c in categories]
    det_vals = [det['accuracy_by_category'].get(c, 0) * 100 for c in categories]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, gt_vals,  w, label='GT tok',      color='#4C72B0', edgecolor='white')
    b2 = ax.bar(x + w/2, det_vals, w, label='Detected tok', color='#DD8452', edgecolor='white')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_pretty)
    ax.set_ylim(0, 125)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('GT tok vs Detected tok — CLEVR (RQ1: Perception Bottleneck)', pad=16)
    ax.legend(loc='upper right')

    plt.tight_layout()
    savefig('fig2_gt_vs_detected')


# ── Figure 3: Error breakdown for detected tok ────────────────────
def fig_error_breakdown():
    det = load_summary(RESULTS / 'clevr_detected_summary.json')

    total     = det['total']
    correct   = det['correct']
    exec_err  = int(det['execution_error_rate'] * total)
    nl_fail   = total - int(det['valid_program_rate'] * total)
    wrong_exec = max(total - correct - exec_err - nl_fail, 0)

    sizes  = [correct, exec_err, nl_fail, wrong_exec]
    labels = ['Correct', 'Execution error\n(perception)', 'NL2DSL failed', 'Wrong answer']
    colors = ['#55A868', '#C44E52', '#8172B2', '#CCB974']

    filtered = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    sizes, labels, colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(10)

    legend_labels = [f'{l} ({s:,})' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)

    ax.set_title('Error Breakdown — CLEVR Detected Tok', pad=16)
    plt.tight_layout()
    savefig('fig3_error_breakdown')


# ── Figure 4: Category accuracy heatmap ──────────────────────────
def fig_category_heatmap():
    summaries = {
        'CLEVR (GT)':         load_summary(RESULTS / 'clevr_val_gt_n1000_summary.json'),
        'CLEVR (Detected)':   load_summary(RESULTS / 'clevr_detected_summary.json'),
        'CoGenT (GT)':        load_summary(RESULTS / 'cogent_b_gt_n5000_summary.json'),
        'CLEVR-Humans (GT)':  load_summary(RESULTS / 'clevr_humans_gt_summary.json'),
        'Super-CLEVR (GT)':   load_summary(RESULTS / 'superclevr_gt_n5000_summary.json'),
    }

    categories    = ['count', 'exist', 'query', 'compare', 'other']
    cat_labels    = ['Count', 'Exist', 'Query', 'Compare', 'Other']
    dataset_labels = list(summaries.keys())

    data = np.array([
        [s['accuracy_by_category'].get(c, float('nan')) * 100 for c in categories]
        for s in summaries.values()
    ])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(cat_labels)))
    ax.set_xticklabels(cat_labels)
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(len(dataset_labels)):
        for j in range(len(cat_labels)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if val < 40 or val > 85 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Accuracy (%)')
    ax.set_title('Accuracy by Question Category and Dataset', pad=16)

    plt.tight_layout()
    savefig('fig4_category_heatmap')


# ── Figure 5: Valid program rate and execution error rate ─────────
def fig_pipeline_rates():
    summaries = {
        'CLEVR\n(GT)':         load_summary(RESULTS / 'clevr_val_gt_n1000_summary.json'),
        'CLEVR\n(Detected)':   load_summary(RESULTS / 'clevr_detected_summary.json'),
        'CoGenT\n(GT)':        load_summary(RESULTS / 'cogent_b_gt_n5000_summary.json'),
        'CLEVR-Humans\n(GT)':  load_summary(RESULTS / 'clevr_humans_gt_summary.json'),
        'Super-CLEVR\n(GT)':   load_summary(RESULTS / 'superclevr_gt_n5000_summary.json'),
    }

    labels   = list(summaries.keys())
    valid_pr = [s['valid_program_rate'] * 100 for s in summaries.values()]
    exec_err = [s['execution_error_rate'] * 100 for s in summaries.values()]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, valid_pr, w, label='Valid program rate', color='#4C72B0', edgecolor='white')
    ax.bar(x + w/2, exec_err, w, label='Execution error rate', color='#C44E52', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel('%')
    ax.set_title('NL2DSL Valid Program Rate vs Execution Error Rate', pad=16)
    ax.legend(loc='upper right')

    plt.tight_layout()
    savefig('fig5_pipeline_rates')


# ── Figure 6: Color confusion matrix (GT vs Detected) ────────────
def fig_color_confusion():
    import json as _json
    gt_path = Path('datasets/clevr/scenes/CLEVR_val_scenes.json')
    if not gt_path.exists() or not SCENES_DIR.exists():
        print('  fig6 skipped: GT scenes or scene_graphs not found.')
        return

    with open(gt_path) as f:
        gt_scenes = {s['image_filename'].replace('.png', ''): s
                     for s in _json.load(f)['scenes']}

    COLORS = ['red', 'blue', 'green', 'yellow', 'cyan', 'purple', 'brown', 'gray']
    conf = np.zeros((len(COLORS), len(COLORS)), dtype=int)

    for sg_file in SCENES_DIR.glob('clevr_*.json'):
        img_id = sg_file.stem.replace('clevr_', '')
        if img_id not in gt_scenes:
            continue
        gt_objs  = gt_scenes[img_id]['objects']
        with open(sg_file) as f:
            det_objs = _json.load(f)['objects']
        for det in det_objs:
            dcx, dcy = det['pixel_coords'][0], det['pixel_coords'][1]
            best = min(gt_objs, key=lambda g: (g['pixel_coords'][0]-dcx)**2 + (g['pixel_coords'][1]-dcy)**2)
            gt_c  = best['color']
            det_c = det.get('color')
            if gt_c in COLORS and det_c in COLORS:
                conf[COLORS.index(gt_c), COLORS.index(det_c)] += 1

    # Normalize rows
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = np.where(row_sums > 0, conf / row_sums, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(COLORS)))
    ax.set_yticks(range(len(COLORS)))
    ax.set_xticklabels(COLORS, rotation=45, ha='right')
    ax.set_yticklabels(COLORS)
    ax.set_xlabel('Detected color')
    ax.set_ylabel('GT color')
    ax.set_title('Color Classification Confusion Matrix\n(GT vs CLIP-detected)', pad=16)

    for i in range(len(COLORS)):
        for j in range(len(COLORS)):
            val = conf_norm[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.0f}%' if val >= 0.1 else '',
                        ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction')
    plt.tight_layout()
    savefig('fig6_color_confusion')


# ── Figure 7: Avg detected vs GT objects per scene ───────────────
def fig_object_count_dist():
    gt_path = Path('datasets/clevr/scenes/CLEVR_val_scenes.json')
    if not gt_path.exists() or not SCENES_DIR.exists():
        print('  fig7 skipped.')
        return

    with open(gt_path) as f:
        gt_scenes = {s['image_filename'].replace('.png', ''): s
                     for s in json.load(f)['scenes']}

    # Group by GT count: avg detected
    by_gt = {}
    for sg_file in SCENES_DIR.glob('clevr_*.json'):
        img_id = sg_file.stem.replace('clevr_', '')
        if img_id not in gt_scenes:
            continue
        gt_n = len(gt_scenes[img_id]['objects'])
        with open(sg_file) as f:
            det_n = len(json.load(f)['objects'])
        by_gt.setdefault(gt_n, []).append(det_n)

    gt_counts  = sorted(by_gt.keys())
    avg_det    = [np.mean(by_gt[n]) for n in gt_counts]
    std_det    = [np.std(by_gt[n])  for n in gt_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gt_counts, gt_counts,  'o--', color='#4C72B0', label='GT (perfect)', linewidth=1.5)
    ax.errorbar(gt_counts, avg_det, yerr=std_det, fmt='s-',
                color='#DD8452', label='YOLO detected (avg ± std)',
                capsize=4, linewidth=1.5)

    ax.set_xlabel('GT object count per scene')
    ax.set_ylabel('Number of objects')
    ax.set_title('GT vs YOLO-Detected Object Count per Scene', pad=16)
    ax.legend()
    ax.set_xticks(gt_counts)
    plt.tight_layout()
    savefig('fig7_object_count_dist')


# ── Figure 8: Accuracy vs number of objects in scene ─────────────
def fig_accuracy_vs_objects():
    gt_path = Path('datasets/clevr/scenes/CLEVR_val_scenes.json')
    pq_path = RESULTS / 'clevr_detected_per_question.jsonl'
    if not gt_path.exists() or not pq_path.exists():
        print('  fig8 skipped.')
        return

    with open(gt_path) as f:
        gt_scenes = {s['image_filename'].replace('.png', ''): s
                     for s in json.load(f)['scenes']}

    results = load_per_question(pq_path)

    by_count = {}
    for r in results:
        img_id = r.get('image_id', '')
        if img_id not in gt_scenes:
            continue
        n = len(gt_scenes[img_id]['objects'])
        by_count.setdefault(n, []).append(r['correct'])

    counts = sorted(by_count.keys())
    accs   = [np.mean(by_count[c]) * 100 for c in counts]
    ns     = [len(by_count[c]) for c in counts]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars = ax1.bar(counts, accs, color='#C44E52', alpha=0.8, edgecolor='white', width=0.6)
    for bar, val in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=9, color='#C44E52')

    ax1.set_xlabel('Number of objects in scene (GT)')
    ax1.set_ylabel('Accuracy (%)', color='#C44E52')
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax1.set_ylim(0, 45)
    ax1.set_xticks(counts)

    ax2 = ax1.twinx()
    ax2.plot(counts, ns, 'o--', color='#aaaaaa', linewidth=1.5, markersize=5)
    ax2.set_ylabel('Number of questions', color='#aaaaaa')
    ax2.tick_params(axis='y', labelcolor='#aaaaaa')
    ax2.spines['right'].set_visible(True)

    ax1.set_title('Detected Tok Accuracy vs Scene Object Count', pad=16)
    patch1 = mpatches.Patch(color='#C44E52', label='Accuracy (%)')
    patch2 = mpatches.Patch(color='#aaaaaa', label='Question count')
    ax1.legend(handles=[patch1, patch2], loc='upper right')

    plt.tight_layout()
    savefig('fig8_accuracy_vs_objects')


# ── Figure 9: Attribute accuracy summary (color/shape/material) ───
def fig_material_confusion():
    gt_path = Path('datasets/clevr/scenes/CLEVR_val_scenes.json')
    if not gt_path.exists() or not SCENES_DIR.exists():
        print('  fig9 skipped.')
        return

    with open(gt_path) as f:
        gt_scenes = {s['image_filename'].replace('.png', ''): s
                     for s in json.load(f)['scenes']}

    color_c = shape_c = mat_c = total = 0

    for sg_file in SCENES_DIR.glob('clevr_*.json'):
        img_id = sg_file.stem.replace('clevr_', '')
        if img_id not in gt_scenes:
            continue
        gt_objs = gt_scenes[img_id]['objects']
        with open(sg_file) as f:
            det_objs = json.load(f)['objects']
        for det in det_objs:
            dcx, dcy = det['pixel_coords'][0], det['pixel_coords'][1]
            best = min(gt_objs, key=lambda g: (g['pixel_coords'][0]-dcx)**2 + (g['pixel_coords'][1]-dcy)**2)
            if det.get('color')    == best['color']:    color_c += 1
            if det.get('shape')    == best['shape']:    shape_c += 1
            if det.get('material') == best['material']: mat_c   += 1
            total += 1

    attrs  = ['Color', 'Shape', 'Material']
    accs   = [color_c/total*100, shape_c/total*100, mat_c/total*100]
    colors = ['#4C72B0', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(attrs, accs, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'CLIP Attribute Classification Accuracy\n(n={total} matched objects)', pad=16)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    savefig('fig9_attribute_accuracy')


# ── Figure 10: Execution error breakdown ─────────────────────────
def fig_execution_errors():
    pq_path = RESULTS / 'clevr_detected_per_question.jsonl'
    if not pq_path.exists():
        print('  fig10 skipped.')
        return

    results = load_per_question(pq_path)
    errors  = [r['execution_error'] for r in results
               if r.get('execution_error') not in (None, 'image_not_found')]

    got0  = sum(1 for e in errors if 'got 0' in e)
    gotN  = len(errors) - got0

    labels = ['unique(): no object found\n(filter returned 0 matches)',
              'unique(): multiple objects\n(filter returned >1 match)']
    values = [got0, gotN]
    colors = ['#C44E52', '#DD8452']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.4)

    for bar, val in zip(bars, values):
        pct = val / len(errors) * 100
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{val:,}  ({pct:.1f}%)', va='center', fontsize=11)

    ax.set_xlabel('Number of errors')
    ax.set_title(f'Execution Error Types — CLEVR Detected Tok\n(total: {len(errors):,} errors out of {len(results):,} questions)', pad=16)
    ax.set_xlim(0, max(values) * 1.35)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    savefig('fig10_execution_errors')


if __name__ == '__main__':
    print('Generating figures...')
    fig_gt_accuracy()
    fig_gt_vs_detected()
    fig_error_breakdown()
    fig_category_heatmap()
    fig_pipeline_rates()
    fig_color_confusion()
    fig_object_count_dist()
    fig_accuracy_vs_objects()
    fig_material_confusion()
    fig_execution_errors()
    print(f'\nAll figures saved to {OUT.absolute()}')
