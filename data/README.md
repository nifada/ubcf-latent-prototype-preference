# Data

## Source dataset
The ratings come from the **MovieLens 100K** dataset (943 users, 1,682 items, 100,000 ratings) published by GroupLens Research, University of Minnesota: <https://grouplens.org/datasets/movielens/100k/>

Please cite: F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19:1–19:19, 2015.

## Folds (created by the authors)
The five train/test folds in `folds/` were created by the authors from the ratings of MovieLens 100K. They are **not** the folds distributed by GroupLens (`u1.base` … `u5.test`). The files contain user, item and rating only (`user::item::rating`); timestamps were not used.

| File | Content |
|---|---|
| `folds/training-fold{1..5}-rating10.txt` | 80,000 training ratings per fold |
| `folds/tes-fold{1..5}-rating10.txt` | 20,000 test ratings per fold |

"Given-10" protocol: in every fold the 195–215 test users (207 on average) keep at least 10 ratings in the training set (10.5 on average) and have about 97 test ratings on average; 79% of them have at least 20 test items. All other users contribute all of their ratings to the training set. Recommendations are generated and evaluated only for the test users of each fold.

Split procedure: The folds were generated from the 100,000 ratings of MovieLens 100K (`u.data`, converted to `user::item::rating` format) with a custom splitting tool (given-10 protocol). For each fold, users with more than 10 ratings were drawn at random without replacement; for each drawn user, randomly chosen ratings were moved to the test file until only 10 remained in training. Drawing stopped when the test file reached 20% of all ratings (20,000 ratings, 195–215 test users per fold), so the last user drawn in each fold keeps more than 10 training ratings. The five folds were drawn independently with an unseeded random generator, so their test users overlap (674 distinct test users). The released files are the exact splits used in the experiments; the splitting tool itself is not part of this repository.

## Terms of use
The fold files are derived from MovieLens 100K and are shared under the GroupLens terms (use for research, attribution, no commercial use or redistribution of the original dataset without permission; see the link above). The code of this repository is released under the MIT licence (`../LICENSE`).
