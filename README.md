# Compositional Image Retrieval with CLIP — Dynamic & Hybrid Conditioning

![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Backbone](https://img.shields.io/badge/backbone-CLIP%20ViT--B%2F32%20(frozen)-16a085)
![Dataset](https://img.shields.io/badge/dataset-CelebA-orange)
![Stats](https://img.shields.io/badge/eval-paired%20bootstrap-8a2be2)

Given a **reference face** and a set of **positive and negative textual conditions**
(e.g. `+Eyeglasses, -Red_Hair`), retrieve images that **satisfy all the conditions while
preserving the identity** of the reference. CLIP ViT-B/32 (HuggingFace) is kept completely
frozen; the project studies how the query vector should be built from the image and text
embeddings.

Built for the **Deep Learning** course at the University of Trento (2026).

## Results

The zero-shot baseline of the assignment is the latent-space sum `q = normalize(r + c)`, where
`c` adds a bipolar text direction for each `T+` condition and subtracts one for each `T-`
condition. We replace it with a **learned projection head** that reads the pair
*(reference, conditions)* and outputs a 512-d displacement vector:

```
baseline:      q = normalize( r + c )
contribution:  q = normalize( r + γ · MLP([r ; c]) )      →   collapses to  W_r r + W_c c + b
```

| Official benchmark (13 queries, CelebA test) | Recall@10 | vs baseline (paired, 95% CI) |
|---|---|---|
| reference image only, no conditions | 7.62% | −8.30pp |
| zero-shot baseline (correctly implemented) | 15.92% | — |
| **learned head, collapsed to two matrices + bias** | **37.03%** | **+21.11pp** [+12.87, +29.75], p < 0.0001, 12/13 queries won |
| bagging ensemble (12 models) | 37.35% | +21.43pp [+13.22, +30.11] |

On the 101-query development set the head gives **+6.95pp** Recall@10 (19.91% → 26.86%),
replicated on 5 seeds, all p < 0.0001. Some other findings:

- the learned map is **almost linear** (R² ≈ 0.99): two matrices and a bias keep 93% of the gain,
  with the same inference cost as the baseline;
- **64% of the gain** comes from conditioning on the reference image, not from correcting the
  text directions;
- the method is limited to the **closed vocabulary** of the training attributes;
- twelve alternative hypotheses (per-condition dynamic weighting, cross-attention fusion, text
  direction repair, attribute filters, …) were **measured and excluded** with oracle upper
  bounds and paired bootstrap tests.

## Evaluation protocol

With only 13 official queries, differences below about 6.8pp can't be resolved, and every
effect at stake was smaller than that. So all methods are developed and compared on a
**101-query development set** built on the validation split. Results are reported as a
**paired difference with a hierarchical bootstrap CI** (`src/devset.py`, `bootstrap_paired`),
never as two means side by side. The official benchmark was spent only on methods declared in
advance, together with a written prediction.

## Repository layout

```
notebooks/12_submission.ipynb      the final report: full runnable code + text, trains the head from scratch
notebooks/11_baseline_e_stato.ipynb  baseline, evaluation protocol and early exclusions
src/benchmark.py                   official benchmark: parsing, ground truth, Recall/Precision@K
src/devset.py                      development set, hierarchical & paired bootstrap
src/extract_embeddings.py          CLIP image-embedding extraction for the CelebA splits
src/phi_dynamic.py                 learned per-condition weighting (negative result)
contributo-candidato/              the contribution: multi-seed runs, linear collapse, interpretability, checkpoint
scripts/                           training (Colab), benchmark evaluation, oracle ceilings, filters
esperimenti/                       characterization experiments with their JSON results
celeba_evaluation.json             official benchmark (13 distinct queries)
celeba_devset_valid.json           development set (101 queries)
```

Most of the internal notes and comments are in Italian. The submission notebook is in English.

## Quick start

```bash
conda create --name clip_project python=3.12 && conda activate clip_project
pip install -r requirements.txt
```

The easiest path is to open **`notebooks/12_submission.ipynb` on Colab**: it downloads the
precomputed CLIP embeddings and the few images it needs from the public materials repo
([filipporoseghini-cyber/dl-project-2026-materials](https://github.com/filipporoseghini-cyber/dl-project-2026-materials))
and runs end to end in a few minutes.

To recompute the embeddings yourself, download CelebA (images, `list_attr_celeba.txt`,
`list_eval_partition.txt`), set `DATA_ROOT` and `SPLIT` in `src/extract_embeddings.py` and run
it once per split. The output goes to `embeddings/celeba_{split}_clip_vitb32.pt`.

> **Data notice.** CelebA images and the extracted embeddings (~460 MB) are **not** included
> in this repository. CelebA is released for non-commercial research only. Get it from the
> [official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Authors

Team project by **Nilo Netti** ([@ermlne](https://github.com/ermlne)),
**Arianna Schiavi** ([@ariannaschiavi-AIS](https://github.com/ariannaschiavi-AIS)) and
**Filippo Roseghini** ([@filipporoseghini-cyber](https://github.com/filipporoseghini-cyber)).

## License

MIT, see [LICENSE](LICENSE). CelebA is subject to its own license.
