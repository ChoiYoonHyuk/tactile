# Tactile Retrieval

Compact code for **Review-Based Model Selection for Tactile Fashion Retrieval**.

## Quick start

Use Python 3.10 or later. From this directory:

```bash
python -m pip install -r requirements.txt
python -B reproduce.py --output outputs/selection --check-expected
```

The supplied frozen predictions reproduce Table 2 point estimates and selection frequencies on CPU. The default run computes no confidence intervals. For the original 5,000 paired bootstrap draws:

```bash
python -B reproduce.py --output outputs/bootstrap --bootstrap-draws 5000 --check-expected
```

Each run requires a new output directory. Results include policy values, candidate scores, selection frequencies, and review-validation scores. Bootstrap runs also write contrast intervals. `--check-expected` compares results with the supplied reference CSV files.

## Core files

| File | Purpose |
|---|---|
| `labels.py` | Full-sentence review evidence and attribute-label resolution |
| `models.py` | Residual adapters, linear heads, and training losses |
| `train.py` | Review supervision and head fitting from frozen features |
| `metrics.py` | AUROC, tie-aware nDCG, category averages, and Spearman correlation |
| `selection.py` | Review/property selectors and paired material resampling |
| `reproduce.py` | Table 2 reproduction and reference checks |

`data/` contains the existing development predictions, material predictions, and selection/test assignments. `expected/` contains reference outputs. `settings.json` records selection settings, and `checksums.json` verifies the reproduction inputs.

The numerical kernels and evidence rules follow the experiment implementation. Download scripts, server-specific paths, archival checks, and unused experimental branches are omitted.

## Training

Training requires PyTorch and the original frozen-feature NPZ files, which are separate inputs:

```bash
python -m pip install -r requirements-training.txt
python -B train.py --features source_features41.npz --kind general --output outputs/general
python -B train.py --features source_features41.npz --kind head --adapter outputs/general --output outputs/general_head
```

Use `--kind tactile` for tactile-review supervision. Omit `--adapter` to fit a head on the original image features. Supplying FashionCLIP features fits the FashionCLIP head with the same recipe. Defaults are three runs, 30 epochs, AdamW, learning rate 0.001, weight decay 0.0001, and batches of 128. Mean development macro AUROC selects one common epoch. Outputs are `selection.json` and three `seed*.pt` checkpoints.

| Feature key | Shape and meaning |
|---|---|
| `images` | `[N, D]` frozen image features |
| `split`, `group_id` | `[N]` partition and image-group identifiers |
| `A_labels`, `B_labels` | `[N, 3]` labels, with NaN for unknown values |
| `general`, `tactile` | `[N, D]` pooled training-review features |
| `prompt_vectors` | `[3, 2, D]` descriptor prototypes for adapter selection |

Image groups must be disjoint across partitions. Build features with separate author groups: TRAIN uses A labels and development uses B labels. Review features outside TRAIN must be zero. The text targets retain the original window/review/author averaging. Head-only training needs neither review-feature matrix nor descriptor prototypes.

## Labels and scores

Axes are `soft_stiff`, `smooth_rough`, and `thin_thick`. Scores increase toward soft, smooth, and thin. Negate the smooth/thin score when ranking increasing roughness/thickness. Unknown mentions and explicit negation never imply the opposite descriptor.

```python
from labels import extract_evidence, resolve_axis

evidence = extract_evidence("The fabric is soft and smooth.")
votes = {"author_1": {(r["term"], r["polarity"]) for r in evidence}}
label = resolve_axis(votes, ("soft", "stiff"))
```

The eight candidates are original SigLIP2 (S), original FashionCLIP (F), and general/tactile-review SigLIP2 adapters (G/T), each with text or head readout. Text readouts use fixed descriptor prompts. Each fitted run is evaluated separately before averaging metrics.

## Evaluation

Selection uses the requested attribute's development AUROC or independent property evidence. Property budgets are 8/16 materials per category, totaling 16/32 for SML and 80/160 for EG. Test sets contain 40 SML and 200 EG materials. The two SML tasks share material identities and resampling.

nDCG uses linear midrank relevance, expected gain over entire score ties, both query directions, and equal category weights. Regret is the gap from the best candidate on the same test materials. The 100 nested selection subsets share a finite reservoir.

Table 1 metrics can be applied to the corresponding held-out predictions through `metrics.py`. Raw images, review corpora, encoder weights, original training-feature files, and Table 1 prediction arrays are external inputs. The supplied arrays support the Table 2 reproduction command above.

## Validation

The compact implementation reproduces all 24 policy rows and 144 selection-frequency rows from the reference results. Numerical kernels and review extraction are checked against the experiment code. Adapter, raw-head, and adapted-head training are checked with synthetic features. Full bootstrap confidence intervals are available through the optional command; the 5,000-draw analysis is not rerun during packaging.
