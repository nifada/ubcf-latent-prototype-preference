"""
Paired t-tests with Benjamini-Hochberg adjustment (Tables IV, VI and VII of the paper).

Input : a CSV with the NDCG@1..100 of every method and fold
        (default results/per_fold_ndcg.csv, or the outputs/per_fold_ndcg.csv of run_experiments.py).
        Columns: method, fold, N=1 ... N=100.
Output: results/tables/ndcg_all_methods.csv, significance_proposed_vs_baselines.csv,
        significance_vs_reference_baselines.csv and significance_ablation_contrasts.csv.

    python significance_tests.py
    python significance_tests.py --ndcg outputs/per_fold_ndcg.csv --out-dir my_tables
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

CUTOFFS = (5, 10, 15, 20)
LATENT = {"S": "SVD", "P": "PCA", "N": "NMF"}      # first letter of a proposed method
CLUSTER = {"F": "FCM", "S": "SOM"}                 # second letter
PROPOSED = [f"{a}{b}-{s}" for s in ("ITR", "IPWR") for a in "SPN" for b in "FS"]
BASELINES = ["SA-PCC", "ITR", "IPWR", "UBCF+PCC", "ItemKNN", "Siamese"]


def load_ndcg(path):
    """Return {method: array (folds x 100)} of NDCG@1..100."""
    table = pd.read_csv(path)
    if "h" in table:                                # outputs of run_experiments.py: keep one h per method
        table = table.drop_duplicates(["method", "fold"], keep="first")
    columns = [f"N={n}" for n in range(1, 101)]
    return {m: g.sort_values("fold")[columns].to_numpy() for m, g in table.groupby("method")}


def benjamini_hochberg(p):
    """Adjusted p-values of the Benjamini-Hochberg step-up procedure."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        running = min(running, p[i] * n / rank)
        adjusted[i] = running
    return adjusted


def paired_test(ndcg, a, b, cutoff):
    x, y = ndcg[a][:, cutoff - 1], ndcg[b][:, cutoff - 1]
    return float((x - y).mean()), float(stats.ttest_rel(x, y).pvalue)


def test_family(ndcg, pairs, cutoffs):
    """Test every (a, b) pair at every cut-off and adjust the whole family together."""
    rows = []
    for a, b in pairs:
        for n in cutoffs:
            diff, p = paired_test(ndcg, a, b, n)
            rows.append(dict(a=a, b=b, cutoff=n, mean_diff=diff, p_raw=p))
    for row, q in zip(rows, benjamini_hochberg([r["p_raw"] for r in rows])):
        row["p_BH_adjusted"] = float(q)
    return pd.DataFrame(rows)


def ablation_contrasts():
    """(group, label, a, b): the contrasts of Tables VII (latent, clustering, similarity)."""
    contrasts = []
    for sim in ("ITR", "IPWR"):
        for c, cl in (("S", "SOM"), ("F", "FCM")):
            name = lambda latent: f"{latent}{c}-{sim}"
            contrasts += [("latent", f"PCA vs SVD ({cl}, {sim})", name("P"), name("S")),
                          ("latent", f"PCA vs NMF ({cl}, {sim})", name("P"), name("N")),
                          ("latent", f"NMF vs SVD ({cl}, {sim})", name("N"), name("S"))]
    for sim in ("ITR", "IPWR"):
        contrasts += [("cluster", f"FCM vs AHC (SVD, {sim})", f"SF-{sim}", f"SA-{sim}"),
                      ("cluster", f"SOM vs AHC (SVD, {sim})", f"SS-{sim}", f"SA-{sim}")]
        for latent, nm in LATENT.items():
            contrasts.append(("cluster", f"SOM vs FCM ({nm}, {sim})", f"{latent}S-{sim}", f"{latent}F-{sim}"))
    for latent, nm in LATENT.items():
        for c, cl in CLUSTER.items():
            contrasts.append(("similarity", f"IPWR vs ITR ({nm}+{cl})", f"{latent}{c}-IPWR", f"{latent}{c}-ITR"))
    contrasts += [("similarity", "IPWR vs ITR (SVD+AHC)", "SA-IPWR", "SA-ITR"),
                  ("similarity", "IPWR vs ITR (no latent, no clustering)", "IPWR", "ITR")]
    return contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndcg", default="results/per_fold_ndcg.csv")
    p.add_argument("--out-dir", default="results/tables")
    args = p.parse_args()
    ndcg = load_ndcg(args.ndcg)
    os.makedirs(args.out_dir, exist_ok=True)

    # Mean and standard deviation over the folds (Table III)
    rows = []
    for method, a in ndcg.items():
        row = {"method": method}
        for n in CUTOFFS:
            row[f"NDCG@{n} mean"], row[f"NDCG@{n} sd"] = a[:, n - 1].mean(), a[:, n - 1].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(args.out_dir, "ndcg_all_methods.csv"), index=False)

    # Table IV: every proposed method against the six baselines (one family of 288 tests)
    main_family = test_family(ndcg, [(m, b) for m in PROPOSED for b in BASELINES], CUTOFFS)
    main_family.rename(columns={"a": "proposed", "b": "baseline", "cutoff": "cutoff"}).to_csv(
        os.path.join(args.out_dir, "significance_proposed_vs_baselines.csv"), index=False)

    # Table VI: the two best methods against the reference baselines TopPop and SLIM
    if all(m in ndcg for m in ("TopPop", "SLIM")):
        reference = test_family(ndcg, [(m, b) for m in ("PS-IPWR", "PS-ITR") for b in ("TopPop", "SLIM")], CUTOFFS)
        reference.to_csv(os.path.join(args.out_dir, "significance_vs_reference_baselines.csv"), index=False)

    # Table VII: ablation contrasts at N = 10 and 20
    contrasts = ablation_contrasts()
    rows = []
    for group, label, a, b in contrasts:
        for n in (10, 20):
            diff, pval = paired_test(ndcg, a, b, n)
            rows.append(dict(group=group, contrast=label, a=a, b=b, cutoff=n, mean_diff=diff, p_raw=pval))
    ablation = pd.DataFrame(rows)
    ablation["p_BH_adjusted"] = benjamini_hochberg(ablation["p_raw"].to_numpy())
    ablation.to_csv(os.path.join(args.out_dir, "significance_ablation_contrasts.csv"), index=False)
    print(f"Tables written to {args.out_dir}/ ({len(main_family)} + {len(ablation)} tests)")


if __name__ == "__main__":
    main()
