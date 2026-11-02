"""
User-based collaborative filtering with latent representation, prototype-based
clustering and preference-aware similarity.

Companion code of the manuscript
    "Enhancing User-Based Collaborative Filtering Through Latent Representation,
     Prototype-Based Clustering, and Preference-Aware Similarity" (JAIT).

Everything lives in this one module.  Every method of the paper is a function:

    1. Data                 load_fold
    2. Latent projection    latent_svd, latent_pca, latent_nmf
    3. Clustering           cluster_som, cluster_fcm, cluster_ahc
    4. User similarity      similarity_pcc, similarity_itr, similarity_ipwr
    5. Prediction           predict_ratings
    6. Ranking and metrics  rank_items, ranking_metrics, rating_error
    7. Methods              run_ubcf_method
    8. Experiment driver    evaluate_method

Conventions
-----------
* Rating matrices are pandas DataFrames: rows = users, columns = items, 0 = not rated.
* "Given-10" protocol: every test user keeps 10 ratings in the training matrix;
  the remaining ratings of that user are in the test matrix.
* A method recommends, for each test user, the items the user has NOT rated in
  the training matrix, ranked by predicted rating.
* All ratings of the test file count as relevant (any test rating >= 1).
"""
import os
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "folds")
N_FOLDS = 5
TOP_N = 100          # length of the ranked list that is evaluated
CUTOFFS = (5, 10, 15, 20)


# ---------------------------------------------------------------------------
# Hyperparameters selected in the paper (Table II)
#   f = latent dimension, C = number of clusters, h = neighbourhood size
# ---------------------------------------------------------------------------
METHODS = {
    # name        latent  clustering similarity  f    C    h
    "SA-PCC":  dict(latent="SVD", clustering="AHC", similarity="PCC",  f=20, C=25, h=10),
    "SA-ITR":  dict(latent="SVD", clustering="AHC", similarity="ITR",  f=10, C=30, h=10),
    "SA-IPWR": dict(latent="SVD", clustering="AHC", similarity="IPWR", f=10, C=25, h=10),
    "SF-ITR":  dict(latent="SVD", clustering="FCM", similarity="ITR",  f=30, C=29, h=10),
    "SF-IPWR": dict(latent="SVD", clustering="FCM", similarity="IPWR", f=90, C=30, h=10),
    "PF-ITR":  dict(latent="PCA", clustering="FCM", similarity="ITR",  f=80, C=22, h=10),
    "PF-IPWR": dict(latent="PCA", clustering="FCM", similarity="IPWR", f=70, C=30, h=10),
    "NF-ITR":  dict(latent="NMF", clustering="FCM", similarity="ITR",  f=20, C=27, h=10),
    "NF-IPWR": dict(latent="NMF", clustering="FCM", similarity="IPWR", f=20, C=30, h=10),
    "SS-ITR":  dict(latent="SVD", clustering="SOM", similarity="ITR",  f=30, C=7,  h=20),
    "SS-IPWR": dict(latent="SVD", clustering="SOM", similarity="IPWR", f=30, C=7,  h=10),
    "PS-ITR":  dict(latent="PCA", clustering="SOM", similarity="ITR",  f=50, C=25, h=30),
    "PS-IPWR": dict(latent="PCA", clustering="SOM", similarity="IPWR", f=50, C=25, h=20),
    "NS-ITR":  dict(latent="NMF", clustering="SOM", similarity="ITR",  f=20, C=23, h=20),
    "NS-IPWR": dict(latent="NMF", clustering="SOM", similarity="IPWR", f=20, C=23, h=80),
    # Standalone similarity measures: no latent projection, no clustering (one cluster).
    "ITR":      dict(latent=None, clustering=None, similarity="ITR",  f=None, C=None, h=10),
    "IPWR":     dict(latent=None, clustering=None, similarity="IPWR", f=None, C=None, h=10),
}

# Clustering settings that are shared by all runs of the paper.
SOM_LEARNING_RATE = 0.5      # value used in the runs behind the reported results
SOM_MAX_ITER = 1000
SOM_SEED = 1
FCM_FUZZIFIER = 2.0
FCM_MAX_ITER = 100
FCM_TOLERANCE = 1e-4
FCM_MIN_MEMBERS = 5          # FCM is restarted while a cluster has fewer users


# ===========================================================================
# 1. DATA
# ===========================================================================
def load_fold(fold, data_dir=DATA_DIR):
    """Return (train, test) rating matrices of one fold (users x items, 0 = not rated)."""
    columns = dict(sep="::", names=["user_id", "item_id", "rating"],
                   usecols=range(3), engine="python")

    def read(name):
        path = os.path.join(data_dir, name.format(fold=fold))
        table = pd.read_csv(path, **columns)
        return table.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

    return read("training-fold{fold}-rating10.txt"), read("tes-fold{fold}-rating10.txt")


# ===========================================================================
# 2. LATENT REPRESENTATION  (users x items  ->  users x f)
# ===========================================================================
def latent_svd(R, f):
    """Left singular vectors U[:, :f] of the rating matrix (singular values are not used)."""
    u, _, _ = np.linalg.svd(R, full_matrices=True)
    return u[:, :f]


def latent_pca(R, f):
    """Principal-component scores (scikit-learn PCA, ARPACK solver)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=f, svd_solver="arpack", random_state=0).fit_transform(R)


def latent_nmf(R, f):
    """User factors W of a non-negative factorisation R ~ W H (scikit-learn NMF)."""
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return NMF(n_components=f, init="random", random_state=0).fit_transform(R)


LATENT = {"SVD": latent_svd, "PCA": latent_pca, "NMF": latent_nmf}


# ===========================================================================
# 3. CLUSTERING  (users x f  ->  one cluster label per user)
# ===========================================================================
def cluster_som(X, n_clusters, seed=SOM_SEED):
    """Self-Organizing Map with a one-dimensional map of `n_clusters` neurons (sklearn-som)."""
    try:
        from sklearn_som.som import SOM
    except ImportError as error:
        raise ImportError("The SOM methods need the sklearn-som package: pip install sklearn-som") from error
    som = SOM(m=n_clusters, n=1, dim=X.shape[1], lr=SOM_LEARNING_RATE,
              max_iter=SOM_MAX_ITER, random_state=seed)
    return np.asarray(som.fit_predict(np.asarray(X, dtype=float)))


def _fcm_once(X, n_clusters, rng):
    """One run of Fuzzy C-Means from random memberships; returns hard labels (arg-max)."""
    w = FCM_FUZZIFIER
    membership = rng.rand(len(X), n_clusters)
    membership /= membership.sum(axis=1, keepdims=True)
    previous = 0.0
    for iteration in range(1, FCM_MAX_ITER + 1):
        weights = membership ** w
        centres = (weights.T @ X) / weights.sum(axis=0)[:, None]
        dist2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        objective = (dist2 * weights).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = dist2 ** (-1.0 / (w - 1.0))
        membership = inverse / inverse.sum(axis=1, keepdims=True)
        if abs(objective - previous) < FCM_TOLERANCE:
            break
        previous = objective
    return membership.argmax(axis=1)


def cluster_fcm(X, n_clusters, seed=None, max_restarts=200):
    """Fuzzy C-Means with hard assignment.

    As in the paper, FCM starts from random memberships (no seed) and is run again
    while any cluster has fewer than 5 users.  Pass `seed` for repeatable runs.
    """
    X = np.asarray(X, dtype=float)
    for attempt in range(max_restarts):
        rng = np.random.RandomState(None if seed is None else seed + attempt)
        labels = _fcm_once(X, n_clusters, rng)
        sizes = np.bincount(labels, minlength=n_clusters)
        if sizes.min() >= FCM_MIN_MEMBERS:
            break
    return labels


def cluster_ahc(X, n_clusters):
    """Agglomerative hierarchical clustering, single linkage, correlation distance."""
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="correlation",
                                    linkage="single")
    return model.fit_predict(np.asarray(X, dtype=float))


CLUSTERING = {"SOM": cluster_som, "FCM": cluster_fcm, "AHC": cluster_ahc}


# ===========================================================================
# 4. USER SIMILARITY  (ratings of the users of one cluster -> users x users)
# ===========================================================================
def _user_statistics(R):
    """Mean rating of each user, mean-centred ratings, std. deviation, mean rating of each item."""
    rated = R != 0
    n_rated = rated.sum(axis=1)
    mean_user = np.where(n_rated > 0, R.sum(axis=1) / np.maximum(n_rated, 1), 0.0)
    centred = np.where(rated, R - mean_user[:, None], 0.0)
    # Population std. deviation of the mean-centred ratings; the divisor is the largest
    # number of ratings of any user in the matrix (the length of the padded item sets).
    std = np.sqrt((centred ** 2).sum(axis=1) / n_rated.max())
    n_raters = rated.sum(axis=0)
    mean_item = np.where(n_raters > 0, R.sum(axis=0) / np.maximum(n_raters, 1), 0.0)
    return mean_user, centred, std, mean_item


def similarity_pcc(R):
    """Pearson correlation over the items rated by both users (-10 if there is none)."""
    rated = (R != 0).astype(float)
    centred = _user_statistics(R)[1]
    numerator = centred @ centred.T
    squares = (centred ** 2) @ rated.T            # sum of u's squared deviations on co-rated items
    denominator = np.sqrt(squares * squares.T)
    S = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), -10.0)
    np.fill_diagonal(S, 1.0)
    return S


def similarity_itr(R):
    """ITR = Triangle' x URP  (diagonal 0.5)."""
    mean_user, _, std, _ = _user_statistics(R)
    norm = np.sqrt((R ** 2).sum(axis=1))
    n = len(R)
    S = np.zeros((n, n))
    for u in range(n - 1):
        distance = np.sqrt(((R[u] - R[u + 1:]) ** 2).sum(axis=1))
        triangle = 1 - distance / (norm[u] + norm[u + 1:])
        x = np.abs(mean_user[u] - mean_user[u + 1:]) * np.abs(std[u] - std[u + 1:])
        urp = 1 - 1 / (1 + np.exp(-x))
        S[u, u + 1:] = triangle * urp
    S = S + S.T
    np.fill_diagonal(S, 0.5)
    return S


def similarity_ipwr(R):
    """IPWR = 0.5 * RPB + 0.5 * IPCC  (diagonal 1)."""
    mean_user, _, std, mean_item = _user_statistics(R)
    adjusted = R * (mean_user[:, None] - mean_item[None, :])   # r*mean_user - r*mean_item
    cross = adjusted @ adjusted.T                              # co-rated items only (zeros elsewhere)
    norm = np.sqrt((adjusted ** 2).sum(axis=1))
    n = len(R)
    S = np.zeros((n, n))
    with np.errstate(all="ignore"):
        for u in range(n - 1):
            x = np.abs(mean_user[u] - mean_user[u + 1:]) * np.abs(std[u] - std[u + 1:])
            rpb = np.cos(x)
            ipcc = cross[u, u + 1:] / (norm[u] * norm[u + 1:])
            S[u, u + 1:] = 0.5 * rpb + 0.5 * ipcc
    S = S + S.T
    np.fill_diagonal(S, 1.0)
    return S


SIMILARITY = {"PCC": similarity_pcc, "ITR": similarity_itr, "IPWR": similarity_ipwr}


# ===========================================================================
# 5. RATING PREDICTION
# ===========================================================================
def predict_ratings(S, weights, rated, centred, mean_user, target, hs, min_sim=-1.0):
    """Predict the unrated items of one user from its h most similar raters.

    S, weights : similarity rows used to choose neighbours / to weight them (users x users)
    rated      : boolean rating-indicator matrix of the candidate neighbours
    centred    : mean-centred ratings of the candidate neighbours
    target     : row index of the user to predict
    hs         : list of neighbourhood sizes h; one prediction vector is returned per h

    For every item the neighbours are the h most similar users (other than the target)
    that rated the item and whose similarity is at least `min_sim`; the prediction is
        r_hat(u, i) = mean_u + sum_v w(u, v) (r_vi - mean_v) / sum_v |w(u, v)|
    and NaN if the item has no neighbour.
    Returns (items, {h: predictions}) where `items` are the column positions of the
    items the target has not rated.
    """
    similarity = S[target].copy()
    similarity[target] = -np.inf
    order = np.argsort(-similarity, kind="stable")
    order = order[(order != target) & (similarity[order] >= min_sim)]   # NaN similarities drop out here
    items = np.flatnonzero(~rated[target])
    out = {h: np.full(len(items), np.nan) for h in hs}
    if len(order) == 0:
        return items, out
    w = weights[target, order]
    sub = rated[order][:, items]                                # neighbours x candidate items
    rank = np.cumsum(sub, axis=0)                               # rank of a neighbour among raters
    weighted = w[:, None] * centred[order][:, items]
    for h in hs:
        keep = (sub & (rank <= h)).astype(float)                # the first h raters of each item
        numerator = (weighted * keep).sum(axis=0)
        denominator = np.abs(w) @ keep
        valid = denominator > 0
        out[h][valid] = mean_user[target] + numerator[valid] / denominator[valid]
    return items, out


def predict_clustered(train, test_users, labels, similarity_fn, hs, min_sim=-1.0):
    """User-based predictions where neighbours come from the target's own cluster only.

    A single cluster (labels all equal) gives the standalone similarity methods.
    A user who is alone in its cluster has no neighbours and gets no list.
    Returns {h: {user_id: (items, predictions)}} with `items` as column positions.
    """
    R = train.values.astype(float)
    rated = R != 0
    mean_user, centred, _, _ = _user_statistics(R)
    row_of = {u: k for k, u in enumerate(train.index)}
    wanted = {row_of[u] for u in test_users if u in row_of}
    result = {h: {} for h in hs}
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        targets = [k for k in members if k in wanted]
        if not targets:
            continue
        if len(members) == 1:
            S = None
        else:
            S = similarity_fn(R[members])
        local = {k: j for j, k in enumerate(members)}
        for k in targets:
            if S is None:
                items = np.flatnonzero(~rated[k])
                preds = {h: np.full(len(items), np.nan) for h in hs}
            else:
                items, preds = predict_ratings(
                    S, S, rated[members], centred[members], mean_user[members],
                    local[k], hs, min_sim)
            for h in hs:
                result[h][train.index[k]] = (items, preds[h])
    return result


# ===========================================================================
# 6. RANKING AND METRICS
# ===========================================================================
def rank_items(items, predictions, n=TOP_N, tie_break="item_id"):
    """Top-n item positions by predicted rating; items without prediction go last (-1).

    Many items share exactly the same predicted rating (for example items rated by a single
    neighbour), so the order among tied items matters for the metrics:
      tie_break="item_id"  tied items are listed by ascending item id (deterministic, default);
      tie_break="numpy"    reversed ascending numpy.argsort (the order of the authors' notebooks;
                           it depends on the numpy version and the CPU).
    """
    valid = ~np.isnan(predictions)
    if tie_break == "numpy":
        order = np.flip(predictions[valid].argsort())
    else:
        order = np.argsort(-predictions[valid], kind="stable")
    top = list(items[valid][order][:n])
    return top + [-1] * (n - len(top))


def ranking_metrics(top_lists, relevant, n=TOP_N):
    """Per-user precision, recall, AP and NDCG at every cut-off 1..n.

    top_lists : list of ranked item-position lists (-1 = empty slot)
    relevant  : list of sets with the relevant item positions of each user
    Returns a dict of arrays (users x n), precision/recall in percent.
    """
    ranks = np.arange(1, n + 1)
    gain = 1.0 / np.log2(1 + ranks)
    precision, recall, ndcg, ap = [], [], [], []
    for top, rel in zip(top_lists, relevant):
        hit = np.array([item in rel for item in top], dtype=float)
        hits = np.cumsum(hit)
        precision.append(100 * hits / ranks)
        recall.append(100 * hits / len(rel) if rel else np.zeros(n))
        dcg = np.cumsum(gain * hit)
        ideal = np.cumsum(gain * (ranks <= len(rel)))     # IDCG(N) sums up to min(N, |relevant|)
        ndcg.append(np.where(ideal > 0, dcg / np.where(ideal > 0, ideal, 1), 0.0))
        ap.append((precision[-1] * hit).sum() / len(rel) if rel else 0.0)
    return dict(precision=np.array(precision), recall=np.array(recall),
                ndcg=np.array(ndcg), ap=np.array(ap))


def rating_error(predicted, test, clip=None):
    """MAE over the test ratings that received a prediction, and the share of test ratings scored.

    predicted : {user_id: {item_position: predicted rating}} (NaN predictions are skipped)
    test      : test matrix (users x items)
    clip      : None or (low, high) to clip predictions to the rating scale
    """
    errors, total = [], 0
    for u, row in zip(test.index, test.values):
        positions = np.flatnonzero(row)
        total += len(positions)
        known = predicted.get(u, {})
        for j in positions:
            p = known.get(j, np.nan)
            if not np.isnan(p):
                if clip:
                    p = min(max(p, clip[0]), clip[1])
                errors.append(abs(p - row[j]))
    mae = float(np.mean(errors)) if errors else float("nan")
    return mae, len(errors) / max(total, 1)


# ===========================================================================
# 7. METHODS
#    Each function returns {h: {user_id: (item_positions, predictions)}} for the
#    users of the test set; `evaluate_predictions` turns that into metrics.
# ===========================================================================
def run_ubcf_method(name, train, test_users, hs=None, fcm_seed=None):
    """Any user-based method: standalone ITR / IPWR, SA-*, and the proposed S?-*, P?-*, N?-* methods.

    Steps: latent projection -> clustering -> similarity inside each cluster ->
    h nearest neighbours -> rating prediction.
    """
    cfg = METHODS[name]
    hs = hs or [cfg["h"]]
    if cfg["clustering"] is None:                       # standalone similarity: one cluster
        labels = np.zeros(len(train), dtype=int)
    else:
        X = LATENT[cfg["latent"]](train.values.astype(float), cfg["f"])
        if cfg["clustering"] == "FCM":
            labels = cluster_fcm(X, cfg["C"], seed=fcm_seed)
        else:
            labels = CLUSTERING[cfg["clustering"]](X, cfg["C"])
    # A neighbour needs a similarity of at least -1; this only removes users for which the
    # similarity is undefined (for PCC: no co-rated item, coded -10).
    return predict_clustered(train, test_users, labels, SIMILARITY[cfg["similarity"]], hs, min_sim=-1.0)


# ===========================================================================
# 8. EVALUATION OF ONE METHOD ON ONE FOLD
# ===========================================================================
def evaluate_predictions(predictions, train, test, tie_break="item_id"):
    """Ranking metrics and rating error of the predictions of one cut-off h on one fold.

    Every test user counts in the averages; a user without a list scores 0.
    """
    item_ids = train.columns.values
    top_lists, relevant, no_list = [], [], 0
    scored = {}
    for u, row in zip(test.index, test.values):
        relevant.append({i for i in test.columns[row != 0]})             # item ids
        if u in predictions:
            items, preds = predictions[u]
            top = rank_items(items, preds, tie_break=tie_break)
            top_lists.append([item_ids[j] if j >= 0 else -1 for j in top])
            scored[u] = {test.columns.get_loc(item_ids[j]): p
                         for j, p in zip(items, preds)
                         if item_ids[j] in test.columns and not np.isnan(p)}
            no_list += int(np.isnan(preds).all())
        else:
            top_lists.append([-1] * TOP_N)
            no_list += 1
    metrics = ranking_metrics(top_lists, relevant)
    mae, coverage = rating_error(scored, test)
    mae_clipped, _ = rating_error(scored, test, clip=(1, 5))
    return dict(ndcg_curve=metrics["ndcg"].mean(axis=0), precision_curve=metrics["precision"].mean(axis=0),
                recall_curve=metrics["recall"].mean(axis=0), map=metrics["ap"].mean(),
                mae=mae, mae_clipped=mae_clipped, rating_coverage=coverage,
                no_list_share=no_list / len(test))


def evaluate_method(name, folds=range(1, N_FOLDS + 1), data_dir=DATA_DIR, hs=None,
                    fcm_seed=None, tie_break="item_id", verbose=True):
    """Run one method on the given folds.  Returns {h: [metrics of fold 1, fold 2, ...]}.

    name        : a key of METHODS
    hs          : neighbourhood sizes to evaluate (default: the h of Table II)
    tie_break   : order of items with equal predicted rating, see rank_items
    """
    results = {}
    for fold in folds:
        train, test = load_fold(fold, data_dir)
        users = list(test.index)
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r}")
        predictions = run_ubcf_method(name, train, users, hs, fcm_seed)
        for h, per_user in predictions.items():
            results.setdefault(h, []).append(evaluate_predictions(per_user, train, test, tie_break))
        if verbose:
            first = next(iter(predictions))
            last = results[first][-1]
            print(f"  {name} fold {fold} (h={first}): NDCG@20 = {last['ndcg_curve'][19]:.4f}, "
                  f"no list for {100 * last['no_list_share']:.1f}% of users", flush=True)
    return results


def summarise(results):
    """Mean over folds of the headline numbers for every h (a list of dict rows)."""
    rows = []
    for h, per_fold in results.items():
        row = {"h": h}
        for n in CUTOFFS:
            row[f"NDCG@{n}"] = np.mean([r["ndcg_curve"][n - 1] for r in per_fold])
        row["P@10 (%)"] = np.mean([r["precision_curve"][9] for r in per_fold])
        row["P@20 (%)"] = np.mean([r["precision_curve"][19] for r in per_fold])
        row["MAE"] = np.mean([r["mae"] for r in per_fold])
        row["MAE clipped"] = np.mean([r["mae_clipped"] for r in per_fold])
        row["rating coverage (%)"] = 100 * np.mean([r["rating_coverage"] for r in per_fold])
        row["no list (%)"] = 100 * np.mean([r["no_list_share"] for r in per_fold])
        rows.append(row)
    return rows
