"""
Run methods of the paper on the five given-10 folds in data/folds.

Examples
--------
    python run_experiments.py --list                         # all method names
    python run_experiments.py --methods ITR IPWR             # standalone similarity measures
    python run_experiments.py --methods SA-PCC SA-ITR SA-IPWR
    python run_experiments.py --methods PS-IPWR --folds 1    # one fold only (faster)
    python run_experiments.py --methods PS-IPWR --h 10 20 30 # try several neighbourhood sizes

Results are printed and written to outputs/summary.csv (means over the folds) and
outputs/per_fold_ndcg.csv (NDCG@1..100 of every fold, the input of significance_tests.py).
"""
import argparse
import os

import numpy as np
import pandas as pd

import ubcf


def parse_args():
    all_methods = list(ubcf.METHODS)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", metavar="METHOD", help="methods to run (see --list)")
    p.add_argument("--list", action="store_true", help="print the available methods and exit")
    p.add_argument("--folds", nargs="+", type=int, default=list(range(1, ubcf.N_FOLDS + 1)),
                   help="folds to run (default: 1 2 3 4 5)")
    p.add_argument("--h", nargs="+", type=int, default=None,
                   help="neighbourhood sizes h (default: the value of Table II)")
    p.add_argument("--data-dir", default=ubcf.DATA_DIR, help="folder with the fold files")
    p.add_argument("--out-dir", default="outputs", help="folder for the result files")
    p.add_argument("--fcm-seed", type=int, default=None,
                   help="seed for Fuzzy C-Means (default: none, as in the paper; set it for repeatable runs)")
    p.add_argument("--tie-break", choices=["item_id", "numpy"], default="item_id",
                   help="order of items with equal predicted rating (see ubcf.rank_items)")
    args = p.parse_args()
    if args.list or not args.methods:
        print("Available methods:\n  " + "\n  ".join(all_methods))
        raise SystemExit(0)
    unknown = [m for m in args.methods if m not in all_methods]
    if unknown:
        p.error(f"unknown method(s): {unknown}; use --list")
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary, curves = [], []
    for name in args.methods:
        print(f"Running {name} on folds {args.folds} ...", flush=True)
        results = ubcf.evaluate_method(name, args.folds, args.data_dir, args.h, args.fcm_seed,
                                       args.tie_break)
        for row in ubcf.summarise(results):
            summary.append({"method": name, **row})
        for h, per_fold in results.items():
            for fold, r in zip(args.folds, per_fold):
                curves.append({"method": name, "h": h, "fold": fold,
                               **{f"N={n}": v for n, v in enumerate(r["ndcg_curve"], start=1)}})
    table = pd.DataFrame(summary)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(args.out_dir, "per_fold_ndcg.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\nMean over folds:\n", table.to_string(index=False))
    print(f"\nWritten to {args.out_dir}/summary.csv and {args.out_dir}/per_fold_ndcg.csv")


if __name__ == "__main__":
    main()
