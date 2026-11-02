# Enhancing User-Based Collaborative Filtering Through Latent Representation, Prototype-Based Clustering, and Preference-Aware Similarity

Code, data folds and saved results of the paper published in the *Journal of Advances in Information Technology* (JAIT, 2026).

**Authors:** Noor Ifada, Rizki Ashuri Pratama, Yunus Safi’i, Maulidhan Ady Nugraha

The paper compares configurations of a sequential user-based collaborative filtering (UBCF) framework on MovieLens 100K:

1. **latent representation** of the users: SVD, PCA or NMF;
2. **prototype-based clustering** of the latent users: Self-Organizing Map (SOM) or Fuzzy C-Means (FCM);
3. **preference-aware similarity** inside each cluster: ITR or IPWR, followed by neighbour selection and rating prediction.

## Repository layout

```
ubcf.py                 all methods as functions (data, latent projection, clustering, similarity,
                        prediction, metrics, baselines)
run_experiments.py      command line: run any method on the five folds
significance_tests.py   paired t-tests with Benjamini-Hochberg correction (Tables IV, VI, VII)
colab_run.ipynb         the same run in Google Colab
data/folds/             the five given-10 train/test folds (see data/README.md)
results/                saved results of the paper
    per_fold_ndcg.csv       NDCG@1..100 of every method on every fold
    tables/                 mean and SD of NDCG, adjusted p-values of the paired t-tests
```

## Quick start

```bash
git clone https://github.com/nifada/ubcf-latent-prototype-preference.git
cd ubcf-latent-prototype-preference
pip install -r requirements.txt            # Python 3.10 or newer

python run_experiments.py --list                      # names of all methods
python run_experiments.py --methods ITR --folds 1     # first test, about 15 seconds
python run_experiments.py --methods PS-IPWR PS-ITR    # a proposed method on all five folds
python significance_tests.py                          # statistics from the saved per-fold results
```

`run_experiments.py` prints the mean over the folds (NDCG@5/10/15/20, Precision, MAE, rating coverage, share of
users without a list) and writes `outputs/summary.csv` and `outputs/per_fold_ndcg.csv`.
Useful options: `--folds 1 2`, `--h 10 20 30` (other neighbourhood sizes), `--fcm-seed 0` (repeatable FCM),
`--out-dir`. To run in Google Colab, open `colab_run.ipynb`.

## Methods and hyperparameters (Table II)

f = latent dimension, C = number of clusters, h = number of neighbours. The same values are in `ubcf.METHODS`.

| Method | Latent | Clustering | Similarity | f | C | h |
|---|---|---|---|---|---|---|
| `SF-ITR` / `SF-IPWR` | SVD | FCM | ITR / IPWR | 30 / 90 | 29 / 30 | 10 / 10 |
| `PF-ITR` / `PF-IPWR` | PCA | FCM | ITR / IPWR | 80 / 70 | 22 / 30 | 10 / 10 |
| `NF-ITR` / `NF-IPWR` | NMF | FCM | ITR / IPWR | 20 / 20 | 27 / 30 | 10 / 10 |
| `SS-ITR` / `SS-IPWR` | SVD | SOM | ITR / IPWR | 30 / 30 | 7 / 7 | 20 / 10 |
| `PS-ITR` / `PS-IPWR` | PCA | SOM | ITR / IPWR | 50 / 50 | 25 / 25 | 30 / 20 |
| `NS-ITR` / `NS-IPWR` | NMF | SOM | ITR / IPWR | 20 / 20 | 23 / 23 | 20 / 80 |
| `SA-ITR` / `SA-IPWR` | SVD | AHC | ITR / IPWR | 10 / 10 | 30 / 25 | 10 / 10 |
| `SA-PCC` | SVD | AHC | PCC | 20 | 25 | 10 |
| `ITR`, `IPWR` | – | – | ITR / IPWR (no latent projection, no clustering) | – | – | 10 |

Other settings: SOM with a one-dimensional map of C neurons (`sklearn-som`, 1000 iterations, seed 1);
FCM with fuzzifier 2, tolerance 1e-4, at most 100 iterations, restarted while a cluster has fewer than 5 users;
AHC with single linkage and correlation distance; PCA with the ARPACK solver; NMF with random initialisation (seed 0).

## Where to find each step in `ubcf.py`

| Step | Functions |
|---|---|
| Data | `load_fold` |
| Latent representation | `latent_svd`, `latent_pca`, `latent_nmf` |
| Clustering | `cluster_som`, `cluster_fcm`, `cluster_ahc` |
| User similarity | `similarity_pcc`, `similarity_itr`, `similarity_ipwr` |
| Neighbours and prediction | `predict_ratings`, `predict_clustered` |
| Top-N list and metrics | `rank_items`, `ranking_metrics`, `rating_error`, `evaluate_predictions` |
| Methods | `run_ubcf_method` (all methods above) |
| One method, all folds | `evaluate_method`, `summarise` |

## Evaluation protocol

Each fold has 80,000 training and 20,000 test ratings; the 195–215 test users of a fold keep 10 ratings in the
training set (given-10). The candidates of a test user are the items it has not rated in training; every item of its
test ratings is relevant. Lists have 100 items; NDCG uses the ideal DCG over min(N, |relevant items|) positions.
A test user without a list scores 0 and is kept in the averages. A user who is alone in its AHC cluster has no neighbour and
gets no list. MAE is computed on the test ratings that received a prediction, and "rating coverage" is the share of test
ratings that did.

## Notes on reproducing the reported numbers

* **Deterministic methods.** `ITR`, `IPWR` and `SA-PCC` give the per-fold NDCG stored in `results/per_fold_ndcg.csv` exactly.
* **FCM** starts from random memberships and is not seeded (as in the paper), so `SF-`, `PF-` and `NF-` methods differ a
  little from run to run. Use `--fcm-seed` for repeatable runs.
* **Items with equal predicted rating.** Many items share exactly the same predicted rating, so the order among tied items
  changes the metrics a little. By default `ubcf.py` lists tied items by ascending item id for every method
  (`--tie-break item_id`). The authors' notebooks of the SOM, FCM and AHC methods used numpy's default sort; `--tie-break numpy`
  imitates that order, but numpy does not guarantee it across versions and CPUs.
* **SOM** needs `sklearn-som`; the paper used version 1.1.0.
* **Other models in `results/`.** The saved per-fold NDCG of the other baselines of the paper (UBCF+PCC, ItemKNN, Siamese, TopPop, SLIM, EASE) is kept only so that `significance_tests.py` reproduces Tables III–VII; their code is not part of this repository.

## Data

The folds were created by the authors from MovieLens 100K; see `data/README.md` for the source, the citation and the terms of use.

## Citation

If you use this repository, please cite the paper and the MovieLens dataset.

```bibtex
@article{ifada2026ubcf,
  author  = {Ifada, Noor and Pratama, Rizki Ashuri and Safi'i, Yunus and Nugraha, Maulidhan Ady},
  title   = {Enhancing User-Based Collaborative Filtering Through Latent Representation, Prototype-Based Clustering, and Preference-Aware Similarity},
  journal = {Journal of Advances in Information Technology},
  year    = {2026}
}
```

## Licence

Code: MIT licence (`LICENSE`). Data: MovieLens 100K, derived folds under the GroupLens terms (`data/README.md`).
