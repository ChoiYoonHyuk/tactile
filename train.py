"""Train review adapters or linear heads from frozen image features."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from metrics import auc
from models import AXES, ReportClassifier, ResidualAdapter, masked_bce, review_loss

SEEDS = (20261102, 20261103, 20261104)


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def sign_weights(labels, mask):
    weights = np.ones((3, 2), np.float32)
    for axis in range(3):
        counts = np.array([np.sum(mask[:, axis] & (labels[:, axis] == y)) for y in (0, 1)], float)
        if counts.min() < 2:
            raise ValueError('Training requires at least two labels per class and axis')
        value = np.minimum(counts.sum() / (2 * counts), 20)
        weights[axis] = value / ((value * counts).sum() / counts.sum())
    return weights


def predict(model, values, head=False):
    model.eval()
    with torch.inference_mode():
        chunks = [model(torch.as_tensor(values[i:i + 512], dtype=torch.float32))
                  for i in range(0, len(values), 512)]
    return np.concatenate([(x[0] if head else x).numpy() for x in chunks])


def macro_auc(labels, scores):
    values = []
    for axis in range(3):
        valid = np.isfinite(labels[:, axis])
        y = labels[valid, axis]
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError('Development labels require both classes on every axis')
        values.append(auc(y, scores[valid, axis]))
    return float(np.mean(values))


def load_features(path, kind):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    images, split = unit(data['images']), data['split'].astype(str)
    if images.ndim != 2 or not np.isfinite(images).all() or np.any(np.linalg.norm(images, axis=1) == 0):
        raise ValueError('Image features must be finite nonzero vectors')
    train, dev = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'dev')
    if not len(train) or not len(dev) or not set(split) <= {'train', 'dev', 'test'}:
        raise ValueError('Nonempty train/dev partitions are required')
    groups = data['group_id'].astype(str)
    for first, second in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if set(groups[split == first]) & set(groups[split == second]):
            raise ValueError('Image groups must remain disjoint across partitions')
    a, b = np.asarray(data['A_labels'], np.float32), np.asarray(data['B_labels'], np.float32)
    if a.shape != (len(images), 3) or b.shape != a.shape:
        raise ValueError('Expected three attribute labels per product')
    if not np.isnan(a[split != 'train']).all() or not np.isnan(b[train]).all():
        raise ValueError('Training uses author group A; held-out labels use group B')
    for labels in (a, b):
        if not np.isin(labels[np.isfinite(labels)], (0, 1)).all():
            raise ValueError('Observed labels must be binary')
    data.update(images=images, groups=groups, train=train, dev=dev, a=a, b=b)
    if kind != 'head':
        text = unit(data[kind])
        if text.shape != images.shape or not np.isfinite(text).all() or not np.all(np.linalg.norm(text[train], axis=1) > 0):
            raise ValueError('Every training image requires its matched text feature')
        if np.any(text[split != 'train']):
            raise ValueError('Held-out review text must be absent')
        keys = data.get(kind + '_text_key')
        if keys is None:
            keys = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in text]
        data.update(text=text, text_keys=np.asarray(keys), directions=(unit(data['prompt_vectors'])[:, 0] - unit(data['prompt_vectors'])[:, 1]))
    return data


def fit(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = load_features(args.features, args.kind)
    images, train, dev = data['images'], data['train'], data['dev']
    feature_sha = hashlib.sha256(args.features.read_bytes()).hexdigest()
    mask = np.isfinite(data['a'])
    y = np.where(mask, data['a'], 0).astype(np.float32)
    weights = torch.as_tensor(sign_weights(y[train], mask[train]))
    targets, known = torch.as_tensor(y), torch.as_tensor(mask)
    upstream = None
    if args.adapter:
        if args.kind != 'head':
            raise ValueError('--adapter is only used when fitting a head')
        upstream = json.loads((args.adapter / 'selection.json').read_text(encoding='utf-8'))
        if upstream['features_sha256'] != feature_sha or upstream['seeds'] != list(args.seeds):
            raise ValueError('Adapter and head must use the same feature input and runs')
        if upstream['kind'] not in ('general', 'tactile'):
            raise ValueError('Expected a review-supervised adapter')
    models, optimizers, inputs = [], [], []
    for seed in args.seeds:
        transformed = images
        if upstream:
            adapter = ResidualAdapter(images.shape[1])
            state = torch.load(args.adapter / f'seed{seed}.pt', map_location='cpu', weights_only=True)
            adapter.load_state_dict(state['state_dict'])
            transformed = np.zeros_like(images)
            rows = np.r_[train, dev]
            transformed[rows] = predict(adapter, images[rows])
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ReportClassifier(images.shape[1], 'linear') if args.kind == 'head' else ResidualAdapter(images.shape[1])
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4))
        inputs.append(transformed)
    args.output.mkdir(parents=True, exist_ok=False)
    history, best, states = [], float('-inf'), None
    for epoch in range(1, args.epochs + 1):
        scores = []
        for seed, model, optimizer, image in zip(args.seeds, models, optimizers, inputs):
            model.train()
            order = np.random.default_rng(seed + 1009 * epoch).permutation(train)
            for start in range(0, len(order), 128):
                rows = order[start:start + 128]
                optimizer.zero_grad(set_to_none=True)
                output = model(torch.as_tensor(image[rows]))
                if args.kind == 'head':
                    loss = masked_bce(output[0], targets[rows], known[rows], weights)
                else:
                    groups, keys = data['groups'][rows], data['text_keys'][rows]
                    positive = torch.as_tensor((groups[:, None] == groups[None, :]) | (keys[:, None] == keys[None, :]))
                    loss = review_loss(output, torch.as_tensor(data['text'][rows]), 'infonce', positive)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
            prediction = predict(model, image[dev], args.kind == 'head')
            if args.kind != 'head':
                prediction = prediction @ data['directions'].T
            scores.append(macro_auc(data['b'][dev], prediction))
        mean = float(np.mean(scores))
        history.append(dict(epoch=epoch, run_auroc=scores, mean_auroc=mean))
        if mean > best:
            best, selected_epoch = mean, epoch
            states = [copy.deepcopy(model.state_dict()) for model in models]
        print(json.dumps(history[-1]), flush=True)
    for seed, state in zip(args.seeds, states):
        torch.save(dict(state_dict=state, kind=args.kind, seed=seed, epoch=selected_epoch, dimension=images.shape[1]), args.output / f'seed{seed}.pt')
    result = dict(kind=args.kind, adapter=str(args.adapter.resolve()) if args.adapter else None,
                  seeds=list(args.seeds), selected_epoch=selected_epoch, mean_development_auroc=best,
                  dimension=images.shape[1], features_sha256=feature_sha, axes=list(AXES), history=history)
    (args.output / 'selection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('general', 'tactile', 'head'), required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs=3, default=SEEDS)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != 3:
        parser.error('Use a positive epoch count and three distinct runs')
    fit(args)
