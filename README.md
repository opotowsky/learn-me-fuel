# learn-me-fuel
Repository for various machine learning implementations for me research.

Given a very specific setup of training data:
1. .pkl files are all located in a separate directory: `prep-pkl`
2. Update the location of the desired .pkl file in `learnme.py` (or relevant notebook)
3. Run: `./learnme.py` with the following optional arguments (they indicate True if present, False if not): 1. `--track_preds` (`-tp`), 2. `--err_n_scores` (`-es`), 3. `--learn_curves` (`-lc`), 4. `--valid_curves` (`-vc`), and 5. `--test_compare` (`-tc`)
4. cleanup and store results; update results paths in plotme.py
5. Graph results via: `./plotme.py`
