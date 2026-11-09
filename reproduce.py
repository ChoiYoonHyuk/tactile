"""Reproduce selection points or paired-bootstrap contrasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import selection as engine

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'selection.py'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/selection'))
    parser.add_argument('--bootstrap-draws', type=int, choices=(0, 5000), default=0)
    parser.add_argument('--check-expected', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    integrity = read_json(ROOT / 'checksums.json')
    inputs = [ENGINE, ROOT/'settings.json', ROOT/'data/split_plans59.json',
              ROOT/'data/dev_predictions59.npz', ROOT/'data/physical_predictions59.npz']
    for path in inputs:
        if digest(path) != integrity[path.relative_to(ROOT).as_posix()]['sha256']:
            raise ValueError('Packaged input differs: ' + str(path))
    protocol = read_json(ROOT/'settings.json')
    partition = protocol['partition']
    cfg = dict(partition_seed=partition['split_seed'], calibration_seed=partition['subset_seed'],
               bootstrap_seed=partition['bootstrap_seed'], n_calibration_plans=partition['n_subsets'],
               bootstrap_draws=args.bootstrap_draws, test_per_category=partition['test_per_category'],
               calibration_budgets=partition['budgets'], min_category_n=36,
               batch_size=10, selection_tolerance=1e-12)
    if protocol['methods'] != engine.METHODS or protocol['legacy_methods'] != engine.OLD_METHODS:
        raise ValueError('Candidate order differs from the original study')
    with np.load(ROOT/'data/dev_predictions59.npz', allow_pickle=False) as archive:
        dev = {name: archive[name] for name in archive.files}
    with np.load(ROOT/'data/physical_predictions59.npz', allow_pickle=False) as archive:
        physical = {name: archive[name] for name in archive.files}
    dids, pids = dev['methods'].astype(str).tolist(), physical['methods'].astype(str).tolist()
    source_scores = dev['scores'][[dids.index(m) for m in engine.OLD_METHODS]].astype(np.float64)
    reordered = {ds: physical[ds+'_scores'][[pids.index(m) for m in engine.OLD_METHODS]].astype(np.float64)
                 for ds in ('SML', 'EG')}
    selectors, review_rows = engine.source_selectors(dev, source_scores, cfg['selection_tolerance'])
    fixed = read_json(ROOT/'data/split_plans59.json')
    designs, _, _, _ = engine.make_design(physical, cfg, fixed)
    point_values, point_regrets = [], []
    point_rows, candidate_rows, frequency_rows = [], [], []
    for task in engine.TASKS:
        test, calibration, _, _ = engine.task_metrics(physical, reordered, designs, task, cfg)
        values, regrets, choices = engine.policy_values(test, calibration, selectors, task[2], cfg)
        point_values.append(values)
        point_regrets.append(regrets)
        for i, policy in enumerate(engine.POLICIES):
            point_rows.append(dict(task=task[0], policy=policy, ndcg5=float(values[i]), regret=float(regrets[i])))
            for m, method in enumerate(engine.METHODS):
                frequency_rows.append(dict(task=task[0], policy=policy, method=method,
                                           selected_count=int(np.sum(choices[i] == m))))
        for m, method in enumerate(engine.METHODS):
            candidate_rows.append(dict(task=task[0], method=method, ndcg5=float(test[m])))
    point_values, point_regrets = np.asarray(point_values), np.asarray(point_regrets)
    for i, policy in enumerate(engine.POLICIES):
        point_rows.append(dict(task='three_task_macro', policy=policy,
                               ndcg5=float(point_values[:, i].mean()), regret=float(point_regrets[:, i].mean())))
    if args.check_expected:
        with (ROOT/'expected/policy_results59.csv').open(encoding='utf-8', newline='') as stream:
            expected = {(row['task'], row['policy']): row for row in csv.DictReader(stream)}
        for row in point_rows:
            ref = expected[row['task'], row['policy']]
            np.testing.assert_allclose([row['ndcg5'], row['regret']], [float(ref['ndcg5']), float(ref['regret'])], rtol=0, atol=1e-12)
        with (ROOT/'expected/selection_frequencies59.csv').open(encoding='utf-8', newline='') as stream:
            expected_counts = {(r['task'], r['policy'], r['method']): int(r['selected_count']) for r in csv.DictReader(stream)}
        if any(r['selected_count'] != expected_counts[r['task'], r['policy'], r['method']] for r in frequency_rows):
            raise ValueError('Selection frequency differs from the original study')

    contrasts = []
    if args.bootstrap_draws:
        bootstrap = np.empty((args.bootstrap_draws, len(engine.TASKS), len(engine.POLICIES)), dtype=float)
        for lo in range(0, args.bootstrap_draws, cfg['batch_size']):
            batch = slice(lo, min(lo + cfg['batch_size'], args.bootstrap_draws))
            for ti, task in enumerate(engine.TASKS):
                test, cal, _, _ = engine.task_metrics(physical, reordered, designs, task, cfg, batch)
                bootstrap[batch, ti] = engine.policy_values(test, cal, selectors, task[2], cfg)[0]
            if batch.stop % 100 == 0:
                print(f'Bootstrap {batch.stop}/{args.bootstrap_draws}', flush=True)
        scopes = [(task[0], point_values[ti], bootstrap[:, ti]) for ti, task in enumerate(engine.TASKS)]
        scopes.append(('three_task_macro', point_values.mean(axis=0), bootstrap.mean(axis=1)))
        for scope, values, draws in scopes:
            for left in (0, 1):
                for right in range(2, 6):
                    coverage = .9875 if scope == 'three_task_macro' and left == 0 else .95
                    low, high = engine.ci(draws[:, left] - draws[:, right], coverage)
                    contrasts.append(dict(scope=scope, left=engine.POLICIES[left], right=engine.POLICIES[right],
                                          difference=float(values[left] - values[right]), coverage=coverage,
                                          ci_low=low, ci_high=high))
        if args.check_expected:
            with (ROOT/'expected/contrasts59.csv').open(encoding='utf-8', newline='') as stream:
                expected = {(r['scope'], r['left'], r['right']): r for r in csv.DictReader(stream)}
            for row in contrasts:
                ref = expected[row['scope'], row['left'], row['right']]
                np.testing.assert_allclose([row[k] for k in ('difference', 'coverage', 'ci_low', 'ci_high')],
                                           [float(ref[k]) for k in ('difference', 'coverage', 'ci_low', 'ci_high')], rtol=0, atol=1e-10)

    output = args.output.resolve()
    if output == ROOT or ROOT/'data' in output.parents or ROOT/'data' == output or ROOT/'expected' in output.parents:
        raise ValueError('Output must not replace packaged inputs')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'policy_values.csv', point_rows)
    write_csv(output/'candidate_test.csv', candidate_rows)
    write_csv(output/'selection_frequencies.csv', frequency_rows)
    write_csv(output/'review_candidate_scores.csv', review_rows)
    if contrasts:
        write_csv(output/'contrasts.csv', contrasts)
    result = dict(point_rows=len(point_rows), frequency_rows=len(frequency_rows),
                  bootstrap_draws=args.bootstrap_draws, confidence_intervals_computed=bool(args.bootstrap_draws),
                  expected_values_checked=args.check_expected, selection_sha256=digest(ENGINE),
                  seconds=round(time.monotonic()-start, 3), output=str(output))
    with (output/'run_info.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
