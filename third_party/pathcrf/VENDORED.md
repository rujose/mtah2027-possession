# PathCRF, vendored

- **Source:** github.com/hyunsungkim-ds/pathcrf, commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`.
- **Paper:** Kim H, Lee K, Seo S, Ko S-K, Yoon J, Park C. *PathCRF: Ball-Free Soccer Event
  Detection via Possession Path Inference from Player Trajectories*. KDD 2026.
  doi:10.1145/3770855.3818152.
- **Licence:** Mozilla Public License 2.0 ([LICENSE](LICENSE)). These files stay under MPL-2.0.

## What is here

Copied **unchanged**:
- `models/` (all model files);
- `set_transformer/attention.py`, `blocks.py` and `model.py`;
- `datatools/config.py`;
- `saved/140/args.json`, the settings of the model we use (its "dynamic sparse CRF", fold 6).

**Added (not in the original):** empty `datatools/__init__.py` and `set_transformer/__init__.py`. The
original relies on running from its own folder; these let us import it from ours. No original file
was edited.

**Not copied:** training, data preparation, evaluation and the notebook. We call the model directly
from `possession/pathcrf_adapter.py`.

## The weights

`saved/140/model/state_dict_best.pt` (19.3 MB) isn't in our git history. `possession/pathcrf_adapter.py`
downloads it from the commit above and checks its SHA-256:

    c82235a17416354b5544f85721ca850304765737a2d9d5435dcdefd52a0d3700

It's loaded with `torch.load(..., weights_only=True)`: plain tensors only, no code.

## What it was trained on

The DFL / IDSSE matches J03WMX, J03WN1, J03WOH, J03WOY and J03WPY. J03WQQ was the validation match
(it chose the checkpoint), and J03WR9 is the held-out test match: its `train.py` sorts the match
files and `cv_fold_idx` 6 picks the 7th. **So on our bench, only J03WR9 and Metrica's 3 games are
unseen.**
