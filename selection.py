"""Review and property-based selection on disjoint material roles."""
import numpy as np
from metrics import auc, both_poles
METHODS = ['S_text', 'S_head', 'F_text', 'F_head', 'G_text', 'G_head', 'T_text', 'T_head']
OLD_METHODS = ['raw_prompt', 'raw_label', 'fashion_prompt', 'fashion_label', 'general_prompt', 'general_label', 'tactile_prompt', 'tactile_label']
POLICIES = ['calibration8', 'calibration16', 'review_pooled', 'review_within', 'fixed_S_text', 'fixed_F_text']
TASKS = [('SML_roughness', 'SML', 1, 0), ('SML_thickness', 'SML', 2, 1), ('EG_thickness', 'EG', 2, 1)]

def source_selectors(dev, scores, tolerance):
    y, mask, cats = (dev['y'], dev['mask'].astype(bool), dev['category'].astype(str))
    out = {}
    rows = []
    for axis in (1, 2):
        use = mask[:, axis]
        yy = y[use, axis]
        if not np.isin(yy, [0, 1]).all():
            raise ValueError('Observed review labels must be binary')
        cc = cats[use]
        pooled, within = ([], [])
        for m, method in enumerate(METHODS):
            p_runs, w_runs = ([], [])
            for r in range(3):
                ss = scores[m, r, use, axis]
                p_runs.append(auc(yy, ss))
                numer = 0.0
                denom = 0
                for c in sorted(np.unique(cc)):
                    pick = cc == c
                    count = int(np.sum(yy[pick] == 1) * np.sum(yy[pick] == 0))
                    if count:
                        numer += count * auc(yy[pick], ss[pick])
                        denom += count
                if not denom:
                    raise ValueError('No within-category positive-negative pair')
                w_runs.append(numer / denom)
            p, w = (float(np.mean(p_runs)), float(np.mean(w_runs)))
            pooled.append(p)
            within.append(w)
            rows.append(dict(axis=axis, method=method, pooled_auc=p, within_pair_auc=w))
        p, w = (np.asarray(pooled), np.asarray(within))
        out[axis] = {'review_pooled': int(first_max(p, tolerance)), 'review_within': int(first_max(w, tolerance)), 'pooled_auc': p, 'within_auc': w}
    return (out, rows)

def first_max(values, tolerance=0.0):
    values = np.asarray(values)
    return np.argmax(values >= values.max(axis=-1, keepdims=True) - tolerance, axis=-1)

def make_design(physical, cfg, fixed):
    boot_rng = np.random.default_rng(cfg['bootstrap_seed'])
    designs, arrays, rows, selected_ids = ({}, {}, [], {})
    for dataset in ('SML', 'EG'):
        ids = physical[f'{dataset}_id'].astype(str)
        groups = physical[f'{dataset}_group_id'].astype(str)
        cats = physical[f'{dataset}_category'].astype(str)
        eligible = physical[f'{dataset}_old_eligible'].astype(bool)
        assert len(set(ids)) == len(ids), f'Nonunique {dataset} IDs'
        assert len(set(groups)) == len(groups), f'Nonunique {dataset} material groups'
        designs[dataset] = []
        selected_ids[dataset] = {}
        for ci, category in enumerate(sorted(fixed['datasets'][dataset]['categories'])):
            indices = np.flatnonzero(eligible & (cats == category))
            indices = indices[np.argsort(ids[indices], kind='stable')]
            record = dict(dataset=dataset, category=category, old_eligible_count=len(indices), included=bool(len(indices) >= cfg['min_category_n']))
            if not record['included']:
                rows.append(record)
                continue
            locked = fixed['datasets'][dataset]['categories'][category]
            test = np.asarray(locked['test_indices'], dtype=np.int32)
            reservoir = np.asarray(locked['reservoir_indices'], dtype=np.int32)
            assert len(test) == cfg['test_per_category']
            assert set(test) | set(reservoir) == set(indices)
            assert len(indices) == locked['eligible_n']
            assert ids[test].tolist() == locked['test_ids']
            assert ids[reservoir].tolist() == locked['reservoir_ids']
            assert len(reservoir) >= max(cfg['calibration_budgets'])
            assert not set(groups[test]) & set(groups[reservoir])
            plans = np.asarray(locked['nested_subset_positions'], dtype=np.int32)
            assert plans.shape == (cfg['n_calibration_plans'], max(cfg['calibration_budgets']))
            assert np.all((plans >= 0) & (plans < len(reservoir)))
            assert all((len(set(row)) == len(row) for row in plans))
            boot_test = boot_rng.integers(0, len(test), size=(cfg['bootstrap_draws'], len(test)), dtype=np.int32)
            boot_cal = boot_rng.integers(0, len(reservoir), size=(cfg['bootstrap_draws'], len(reservoir)), dtype=np.int32)
            d = dict(category=category, test=test, reservoir=reservoir, plans=plans, boot_test=boot_test, boot_cal=boot_cal)
            designs[dataset].append(d)
            prefix = f'{dataset}_category{ci}'
            for key in ('test', 'reservoir', 'plans', 'boot_test', 'boot_cal'):
                arrays[f'{prefix}_{key}'] = d[key]
            selected_ids[dataset][category] = dict(test=ids[test].tolist(), reservoir=ids[reservoir].tolist(), storage_prefix=prefix)
            rows.append({**record, 'test_count': len(test), 'reservoir_count': len(reservoir)})
        if not designs[dataset]:
            raise ValueError(f'No eligible categories in {dataset}')
    return (designs, arrays, rows, selected_ids)

def gather_scores(scores, indices):
    gathered = scores[:, :, indices]
    return np.moveaxis(gathered, (0, 1), (-3, -2))

def task_metrics(physical, reordered, designs, task, cfg, batch=None):
    _, dataset, axis, truth_col = task
    scores = -reordered[dataset][..., axis]
    truth = physical[f'{dataset}_truth'][:, truth_col]
    if batch is None:
        shape = ()
    else:
        shape = (batch.stop - batch.start,)
    test = np.zeros(shape + (8,))
    cal = {b: np.zeros(shape + (cfg['n_calibration_plans'], 8)) for b in cfg['calibration_budgets']}
    degeneracy = dict(test=0, cal8=0, cal16=0)
    distinct = {b: [] for b in cfg['calibration_budgets']}
    for d in designs[dataset]:
        if batch is None:
            test_ix = d['test']
            reservoir_ix = d['reservoir']
            cal_ix = reservoir_ix[d['plans']]
        else:
            test_ix = d['test'][d['boot_test'][batch]]
            reservoir_ix = d['reservoir'][d['boot_cal'][batch]]
            cal_ix = reservoir_ix[:, d['plans']]
        test_truth = truth[test_ix]
        test += both_poles(gather_scores(scores, test_ix), test_truth, 5)
        degeneracy['test'] += int(np.sum(np.max(test_truth, -1) == np.min(test_truth, -1)))
        for budget in cfg['calibration_budgets']:
            ix = cal_ix[..., :budget]
            sorted_ix = np.sort(ix, axis=-1)
            distinct[budget].append(1 + np.sum(sorted_ix[..., 1:] != sorted_ix[..., :-1], axis=-1))
            yy = truth[ix]
            cal[budget] += both_poles(gather_scores(scores, ix), yy, 5)
            degeneracy[f'cal{budget}'] += int(np.sum(np.max(yy, -1) == np.min(yy, -1)))
    k = len(designs[dataset])
    test /= k
    for budget in cal:
        cal[budget] /= k
    return (test, cal, degeneracy, {b: np.sum(distinct[b], axis=0) for b in distinct})

def policy_values(test, cal, selectors, axis, cfg):
    shape = test.shape[:-1]
    nplans = cfg['n_calibration_plans']
    choices = np.empty(shape + (len(POLICIES), nplans), dtype=np.int16)
    choices[..., 0, :] = first_max(cal[8], cfg['selection_tolerance'])
    choices[..., 1, :] = first_max(cal[16], cfg['selection_tolerance'])
    choices[..., 2, :] = selectors[axis]['review_pooled']
    choices[..., 3, :] = selectors[axis]['review_within']
    choices[..., 4, :] = 0
    choices[..., 5, :] = 2
    values = np.take_along_axis(np.broadcast_to(test[..., None, :], shape + (len(POLICIES), 8)), choices, axis=-1).mean(axis=-1)
    regrets = test.max(axis=-1)[..., None] - values
    return (values, regrets, choices)

def ci(values, coverage):
    a = (1.0 - coverage) / 2
    return np.quantile(values, [a, 1 - a]).tolist()
