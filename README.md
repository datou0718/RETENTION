# RETENTION

Official implementation of **RETENTION: Resource-Efficient Tree-Based Ensemble Model Acceleration with Content-Addressable Memory**, published in *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)*, 2026.

Authors: **Yi-Chun Liao**, Chieh-Lin Tsai, Yuan-Hao Chang, Camélia Slimani, Jalil Boukhobza, and Tei-Wei Kuo

[Paper](https://doi.org/10.1109/TCAD.2026.3691288) · [arXiv](https://arxiv.org/abs/2506.05994) · [Purity Threshold Pruning](https://github.com/Datou0718/Purity-Threshold-Pruning)

RETENTION reduces TCAM capacity requirements through model pruning and optimized data placement. This repository estimates CAM array counts for trained Random Forest and XGBoost classifiers and includes an example integrating purity threshold pruning.

## Installation

Tested with Python 3.12. Run from the repository root:

```bash
git clone https://github.com/Datou0718/RETENTION.git
cd RETENTION
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

```bash
python example.py
```

The example trains Random Forest and XGBoost on Iris, prunes the Random Forest, and prints CAM counts for all four mapping strategies using `64 × 64` arrays. It also reports Random Forest test accuracy before and after pruning. Results vary with the random training split.

## Usage

Pass a fitted `RandomForestClassifier` with `model_type="RF"`, or a tree-based `XGBClassifier` with numeric splits and `model_type="XGB"`:

```python
from scripts import RetentionModel

mapper = RetentionModel(
    model=trained_model,
    model_type="RF",
    cam_size=64,         # 64 rows × 64 columns
    strategy="SPC",
)
mapper.run_retention()  # Prints the CAM count; returns None.
```

To retrieve the count programmatically:

```python
mapper.extract_paths()
num_cams = int(mapper.data_placement_optimization())
```

Create a new mapper for each strategy. `cam_size` must be a positive integer; for SPC, it must accommodate the maximum number of distinct conditions in any root-to-leaf path.

## Mapping strategies

| Strategy | Description |
| --- | --- |
| `ODR` | **Occurrence-Based Double Reordering:** reorder conditions and paths, then discard arrays containing only don't-care cells. |
| `SPC` | **Similarity-Based Path Clustering:** group paths with shared conditions into individual arrays. |
| `NUM` | **Naive Unified Mapping:** map all trees together. Baseline for ODR. |
| `NIM` | **Naive Independent Mapping:** map each tree separately. Baseline for SPC. |

## Citation

```bibtex
@ARTICLE{11511658,
  author={Liao, Yi-Chun and Tsai, Chieh-Lin and Chang, Yuan-Hao and Slimani, Camélia and Boukhobza, Jalil and Kuo, Tei-Wei},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={RETENTION: Resource-Efficient Tree-Based Ensemble Model Acceleration with Content-Addressable Memory},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Trees (botanical);Vegetation;Cams;Computer aided manufacturing;Memory;Redundancy;Termination of employment;Computers;Machine learning},
  doi={10.1109/TCAD.2026.3691288}}
```
