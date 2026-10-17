import sys, os, json, time, shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

BASE     = '/kaggle/input/datasets/bogdandjukic/neurosymbolic-vqa-eval-test'
CODE_DIR = f'{BASE}/neurosymbolic-vqa'
JSONL    = f'{BASE}/clevr_val_eval.jsonl'
IMAGES   = f'{BASE}/images/images'

N_IMAGES = 500
MATCH_PX = 60

sys.path.insert(0, CODE_DIR)

sg_cache = globals().get('sg_cache', {})
_use_cache = isinstance(sg_cache, dict) and bool(sg_cache)

if _use_cache:
    print(f'[A] Koristim postojeci sg_cache ({len(sg_cache)} slika).')

    _src = globals().get('all_samples', globals().get('samples', []))

    _gt_by_img = {}
    for _s in _src:
        _img = _s.get('image_path', '')
        if _img not in _gt_by_img:
            _gt_by_img[_img] = _s

    if not _gt_by_img:
        print('  Ucitavam GT iz JSONL...')
        with open(JSONL, encoding='utf-8') as _f:
            for _line in _f:
                _s = json.loads(_line)
                _img = _s.get('image_path', '')
                if _img not in _gt_by_img:
                    _gt_by_img[_img] = _s

    eval_pairs = []
    for img_path, det_sg in sg_cache.items():
        if det_sg is None:
            continue
        gt_entry = _gt_by_img.get(img_path)
        if gt_entry is None:
            fname = Path(img_path).name
            for k, v in _gt_by_img.items():
                if Path(k).name == fname:
                    gt_entry = v
                    break
        if gt_entry:
            eval_pairs.append((gt_entry['scene_graph'], det_sg, img_path))

    print(f'  Matched GT+det parova: {len(eval_pairs)}')

else:
    print('[B] sg_cache nije nadjen — gradim scene grafove od nule...')
    work_code = '/kaggle/working/code'
    shutil.copytree(CODE_DIR, work_code, dirs_exist_ok=True)
    sys.path.insert(0, work_code)
    os.chdir(work_code)
    from scene_graph.builder import SceneGraphBuilder
    _builder = SceneGraphBuilder()

    _seen, _raw = set(), []
    with open(JSONL, encoding='utf-8') as _f:
        for _line in _f:
            _s = json.loads(_line)
            _img = _s.get('image_path', '')
            _local = f'{IMAGES}/{Path(_img).name}'
            if _img not in _seen and Path(_local).exists():
                _s['local_path'] = _local
                _seen.add(_img)
                _raw.append(_s)
            if len(_raw) >= N_IMAGES:
                break
    print(f'  Slika za analizu: {len(_raw)}')

    eval_pairs = []
    _t0 = time.time()
    for _i, _s in enumerate(_raw):
        _local = _s.get('local_path', _s.get('image_path', ''))
        try:
            _det_sg = _builder.build(_local)
            eval_pairs.append((_s['scene_graph'], _det_sg, _s['image_path']))
        except Exception as _e:
            print(f'  Greska {Path(_local).name}: {_e}')
        if (_i + 1) % 50 == 0:
            print(f'  [{_i+1}/{len(_raw)}] {time.time()-_t0:.0f}s')

    print(f'  Izgradeno {len(eval_pairs)} parova.')


def px_center(obj):
    c = obj.get('pixel_coords', [0, 0])
    return np.array([c[0], c[1]], dtype=float)

def match_objects(gt_objs, det_objs, max_px=MATCH_PX):
    if not gt_objs or not det_objs:
        return []
    used = set(); pairs = []
    for gi, go in enumerate(gt_objs):
        best_di, best_d = None, float('inf')
        for di, do_ in enumerate(det_objs):
            if di in used:
                continue
            d = np.linalg.norm(px_center(go) - px_center(do_))
            if d < best_d:
                best_d, best_di = d, di
        if best_di is not None and best_d <= max_px:
            pairs.append((gi, best_di))
            used.add(best_di)
    return pairs


ATTRS = ('color', 'shape', 'material', 'size')
DIRS  = ('left', 'right', 'front', 'behind')

det_total_gt = det_total_det = det_matched = 0
attr_correct = {a: Counter() for a in ATTRS}
attr_total   = {a: Counter() for a in ATTRS}
confusion    = {a: defaultdict(Counter) for a in ATTRS}
rel_correct  = Counter(); rel_total = Counter()
img_stats    = []

for gt_sg, det_sg, img_path in eval_pairs:
    gt_objs  = gt_sg.get('objects', [])
    det_objs = det_sg.get('objects', [])
    n_gt  = len(gt_objs); n_det = len(det_objs)
    pairs = match_objects(gt_objs, det_objs)
    n_match = len(pairs)

    det_total_gt  += n_gt
    det_total_det += n_det
    det_matched   += n_match
    img_stats.append({'n_gt': n_gt, 'n_det': n_det, 'n_match': n_match, 'img': img_path})

    for gi, di in pairs:
        go, do_ = gt_objs[gi], det_objs[di]
        for attr in ATTRS:
            gt_val   = str(go.get(attr, '')).lower()
            pred_val = str(do_.get(attr, '')).lower()
            if not gt_val:
                continue
            attr_total[attr][gt_val] += 1
            if pred_val == gt_val:
                attr_correct[attr][gt_val] += 1
            confusion[attr][pred_val][gt_val] += 1

    gt_rels  = gt_sg.get('relationships', {})
    det_rels = det_sg.get('relationships', {})
    gt2det   = {gi: di for gi, di in pairs}
    for direction in DIRS:
        gt_dir  = gt_rels.get(direction, [])
        det_dir = det_rels.get(direction, [])
        for gi, di in pairs:
            if gi >= len(gt_dir):
                continue
            gt_nb  = {gt2det[x] for x in gt_dir[gi] if x in gt2det}
            det_nb = set(det_dir[di]) if di < len(det_dir) else set()
            rel_total[direction] += 1
            if gt_nb == det_nb:
                rel_correct[direction] += 1

print(f'Analiza gotova. {len(img_stats)} slika.')


n_imgs    = len(img_stats)
recall    = det_matched / det_total_gt   if det_total_gt  else 0
precision = det_matched / det_total_det  if det_total_det else 0
f1        = 2*recall*precision / (recall+precision) if recall+precision else 0
count_exact = sum(1 for s in img_stats if s['n_gt'] == s['n_det'])

SEP = '=' * 64

print(f'\n{SEP}')
print(f'PERCEPTION BREAKDOWN — {n_imgs} slika  (MATCH_PX={MATCH_PX}px)')
print(SEP)

print('\n[1] DETEKCIJA OBJEKATA')
print(f'  Ukupno GT obj:      {det_total_gt}  (avg {det_total_gt/max(n_imgs,1):.1f}/slici)')
print(f'  Ukupno detektovano: {det_total_det}  (avg {det_total_det/max(n_imgs,1):.1f}/slici)')
print(f'  Matched parova:     {det_matched}')
print(f'  Object recall:      {recall:.1%}   <- % GT obj koji su NADJENI')
print(f'  Object precision:   {precision:.1%}   <- % det obj koji su tacni')
print(f'  F1:                 {f1:.1%}')
print(f'  Tacan broj obj:     {count_exact}/{n_imgs} = {count_exact/n_imgs:.1%}')
missed = det_total_gt - det_matched
fp     = det_total_det - det_matched
print(f'\n  Po slici prosecno:')
print(f'    Propustenih GT (false neg): {missed/max(n_imgs,1):.2f}')
print(f'    Laznih pozitiva (false pos): {fp/max(n_imgs,1):.2f}')

print('\n[2] TACNOST ATRIBUTA (samo matched objekti)')
attr_summary = {}
for attr in ATTRS:
    tot = sum(attr_total[attr].values())
    cor = sum(attr_correct[attr].values())
    acc = cor / tot if tot else 0
    attr_summary[attr] = (cor, tot, acc)
    bar = '█' * round(acc * 20) + '░' * (20 - round(acc * 20))
    print(f'  {attr:<10} [{bar}] {cor}/{tot} = {acc:.1%}')

print('\n[3] CONFUSION PO VREDNOSTI')
for attr in ATTRS:
    print(f'\n  {attr.upper()}:')
    for gt_val in sorted(attr_total[attr].keys()):
        tot = attr_total[attr][gt_val]
        if tot == 0:
            continue
        cor = attr_correct[attr][gt_val]
        acc = cor / tot
        wrong_as = [(p, confusion[attr][p].get(gt_val, 0))
                    for p in confusion[attr]
                    if p != gt_val and confusion[attr][p].get(gt_val, 0) > 0]
        wrong_as.sort(key=lambda x: -x[1])
        w_str = '  -> prediktovano kao: ' + ', '.join(f'{p}({c})' for p, c in wrong_as[:3]) if wrong_as else ''
        flag = ' !!!' if acc < 0.65 else ''
        print(f'    GT={gt_val:<12} {acc:.0%} ({cor}/{tot}){w_str}{flag}')

print('\n[4] PROSTORNE RELACIJE')
for d in DIRS:
    t = rel_total[d]; c = rel_correct[d]
    print(f'  {d:<8} {c}/{t} = {c/max(t,1):.1%}')
total_rel_c = sum(rel_correct.values())
total_rel_t = sum(rel_total.values())
print(f'  UKUPNO   {total_rel_c}/{total_rel_t} = {total_rel_c/max(total_rel_t,1):.1%}')

print('\n[5] SLIKE SA NAJLOSIJOM DETEKCIJOM (bottom 8)')
worst_imgs = sorted(img_stats, key=lambda s: s['n_match'] / max(s['n_gt'], 1))[:8]
for s in worst_imgs:
    r = s['n_match'] / max(s['n_gt'], 1)
    print(f'  {Path(s["img"]).name}  GT={s["n_gt"]} det={s["n_det"]} match={s["n_match"]} recall={r:.0%}')

total_attr_c = sum(v[0] for v in attr_summary.values())
total_attr_t = sum(v[1] for v in attr_summary.values())
total_rel_acc = total_rel_c / max(total_rel_t, 1)

print(f'\n{SEP}')
print('SUMMARY ZA THESIS:')
print(f'  Object recall:      {recall:.1%}   ({(1-recall)*100:.0f}% GT obj NIJE pronadjeno)')
print(f'  Object precision:   {precision:.1%}')
print(f'  Object F1:          {f1:.1%}')
print(f'  Attr. accuracy:     {total_attr_c/max(total_attr_t,1):.1%}  (samo pronadjeni objekti)')
for attr in ATTRS:
    cor, tot, acc = attr_summary[attr]
    print(f'    {attr:<12} {acc:.1%}  ({cor}/{tot})')
print(f'  Relation accuracy:  {total_rel_acc:.1%}')
best_attr  = max(attr_summary, key=lambda a: attr_summary[a][2])
worst_attr = min(attr_summary, key=lambda a: attr_summary[a][2])
print(f'\n  Najlaksi atribut:  {best_attr} ({attr_summary[best_attr][2]:.1%})')
print(f'  Najtezi atribut:   {worst_attr} ({attr_summary[worst_attr][2]:.1%})')
print(f'\n  Kljucni nalaz: Recall {recall:.1%} = {(1-recall)*100:.0f}% GT obj propusteno')
print(f'  -> kada unique() trazi 1 obj a detector vrati 0, program pada')
print(f'  -> to direktno objasnjava ~55% exec error rate u E2E evaluaciji')
print(SEP)

_out = {
    'n_images': n_imgs, 'match_px': MATCH_PX,
    'detection': {
        'recall': round(recall, 4), 'precision': round(precision, 4),
        'f1': round(f1, 4),
        'count_exact_pct': round(count_exact / max(n_imgs, 1), 4),
    },
    'attributes': {
        attr: {'accuracy': round(acc, 4), 'correct': cor, 'total': tot}
        for attr, (cor, tot, acc) in attr_summary.items()
    },
    'relations': {
        d: {'accuracy': round(rel_correct[d] / max(rel_total[d], 1), 4),
            'correct': rel_correct[d], 'total': rel_total[d]}
        for d in DIRS
    },
}
_out_path = '/kaggle/working/perception_breakdown.json'
with open(_out_path, 'w') as _f:
    json.dump(_out, _f, indent=2)
print(f'\nRezultati sacuvani: {_out_path}')
