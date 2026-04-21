"""
Comprehensive evaluation script.

Evaluates all four approaches against PeerRead ground truth:
  1. Monolithic baseline     (monolithic_prompts/)
  2. Dimension agents        (dimension_agent_prompts/)
  3. Two-stage agents        (two_stage_agent_prompts/)
  4. Debate agents           (debate_agent_prompts/)

Metrics per dimension:
  - Pearson r        (linear correlation)
  - Spearman rho     (rank-order correlation, robust to outliers)
  - MAE              (mean absolute error; measures score calibration)
  - Bias             (mean LLM - mean GT; positive = LLM overestimates)

Accept/Reject prediction (where GT labels are available):
  - Accuracy, Precision, Recall, F1

Extra for debate approach:
  - Debate rate (% of dimensions that triggered the referee)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import re

# ---------------------------------------------------------------------------
# Dimension lists per venue (matching ground truth availability)
# ---------------------------------------------------------------------------

VENUE_DIMENSIONS = {
    "acl_2017": [
        "RECOMMENDATION", "SUBSTANCE", "APPROPRIATENESS",
        "MEANINGFUL_COMPARISON", "CLARITY", "REVIEWER_CONFIDENCE",
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "IMPACT",
    ],
    "conll_2016": [
        "RECOMMENDATION", "SUBSTANCE", "APPROPRIATENESS",
        "MEANINGFUL_COMPARISON", "CLARITY", "REVIEWER_CONFIDENCE",
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "IMPACT", "REPLICABILITY",
    ],
    "iclr_2017": [
        "RECOMMENDATION", "REVIEWER_CONFIDENCE", "CLARITY",
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY",
    ],
}

# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------

def extract_numeric_score(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.match(r"(-?\d+(?:\.\d+)?)", val.strip())
        if m:
            return float(m.group(1))
    return None


def get_ground_truth_scores(venue, peerread_dir, dimension):
    """Average human reviewer scores per paper for one dimension."""
    gt_scores = {}
    reviews_dir = Path(peerread_dir) / venue / "test" / "reviews"
    if not reviews_dir.exists():
        return gt_scores
    for f in reviews_dir.glob("*.json"):
        with open(f) as fin:
            data = json.load(fin)
        scores = [
            v for r in data.get("reviews", [])
            if (v := extract_numeric_score(r.get(dimension))) is not None
        ]
        if scores:
            gt_scores[f.stem] = np.mean(scores)
    return gt_scores


def get_llm_scores(venue, llm_dir, dimension):
    """LLM score per paper for one dimension."""
    llm_scores = {}
    reviews_dir = Path(llm_dir) / venue / "test"
    if not reviews_dir.exists():
        return llm_scores
    for f in reviews_dir.glob("*_review.json"):
        paper_id = f.stem.replace(".pdf_review", "")
        with open(f) as fin:
            data = json.load(fin)
        score = extract_numeric_score(data.get(dimension))
        if score is not None:
            llm_scores[paper_id] = score
    return llm_scores


# ---------------------------------------------------------------------------
# Core correlation / metrics
# ---------------------------------------------------------------------------

def correlate_dimension(venue, peerread_dir, llm_dir, dimension):
    """
    Compute Pearson r, Spearman rho, MAE, and bias for one dimension.

    Returns (pearson_r, pearson_p, spearman_r, spearman_p, mae, bias, n) or None.
    """
    gt_scores = get_ground_truth_scores(venue, peerread_dir, dimension)
    llm_scores = get_llm_scores(venue, llm_dir, dimension)

    common = sorted(set(gt_scores) & set(llm_scores))
    if len(common) < 2:
        return None

    gt = np.array([gt_scores[k] for k in common])
    llm = np.array([llm_scores[k] for k in common])

    if np.std(gt) == 0 or np.std(llm) == 0:
        return None

    pearson_r, pearson_p = pearsonr(gt, llm)
    spearman_r, spearman_p = spearmanr(gt, llm)
    mae = float(np.mean(np.abs(llm - gt)))
    bias = float(np.mean(llm - gt))

    return pearson_r, pearson_p, spearman_r, spearman_p, mae, bias, len(common)


# ---------------------------------------------------------------------------
# Accept / reject prediction
# ---------------------------------------------------------------------------

def get_ground_truth_decisions(venue, peerread_dir):
    decisions = {}
    reviews_dir = Path(peerread_dir) / venue / "test" / "reviews"
    if not reviews_dir.exists():
        return decisions

    if venue == "iclr_2017":
        for f in reviews_dir.glob("*.json"):
            data = json.load(open(f))
            acc = data.get("accepted")
            if acc is not None:
                decisions[f.stem] = bool(acc)

    elif venue == "acl_2017":
        accepted_file = Path(peerread_dir) / "acl_accepted.txt"
        if accepted_file.exists():
            accepted_titles = set(accepted_file.read_text().splitlines())
            for f in reviews_dir.glob("*.json"):
                data = json.load(open(f))
                decisions[f.stem] = data.get("title", "") in accepted_titles

    return decisions


def evaluate_accept_reject(venue, peerread_dir, llm_dir, threshold=5):
    gt_decisions = get_ground_truth_decisions(venue, peerread_dir)
    llm_scores = get_llm_scores(venue, llm_dir, "RECOMMENDATION")

    common = sorted(set(gt_decisions) & set(llm_scores))
    if len(common) < 2:
        return None

    y_true = [int(gt_decisions[k]) for k in common]
    y_pred = [int(llm_scores[k] >= threshold) for k in common]
    n = len(common)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, n, y_true, y_pred, common


# ---------------------------------------------------------------------------
# Debate-specific stats
# ---------------------------------------------------------------------------

def evaluate_augmentation_stats(venue, llm_dir):
    """Report lit-survey augmentation coverage for an approach's output directory."""
    reviews_dir = Path(llm_dir) / venue / "test"
    if not reviews_dir.exists():
        return None

    total = surveys_ok = 0
    pool_sizes = []
    augmented_dims_seen = set()

    for f in reviews_dir.glob("*_review.json"):
        data = json.load(open(f))
        stats = data.get("_augmentation_stats")
        if stats is None:
            return None  # not a lit-augmented output
        total += 1
        if stats.get("survey_found"):
            surveys_ok += 1
            pool_sizes.append(stats.get("survey_pool_size", 0))
        augmented_dims_seen.update(stats.get("augmented_dims", []))

    if total == 0:
        return None

    return {
        "total_papers":       total,
        "papers_with_survey": surveys_ok,
        "coverage_rate":      surveys_ok / total,
        "mean_pool_size":     float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "augmented_dims":     sorted(augmented_dims_seen),
    }


def evaluate_debate_stats(venue, debate_dir):
    """Report how often the referee was triggered across all papers in a venue."""
    reviews_dir = Path(debate_dir) / venue / "test"
    if not reviews_dir.exists():
        return None

    total_dims, debated_dims = 0, 0
    paper_debate_rates = []

    for f in reviews_dir.glob("*_review.json"):
        data = json.load(open(f))
        stats = data.get("_debate_stats")
        if stats:
            total_dims  += stats.get("total_dimensions", 0)
            debated_dims += stats.get("dimensions_debated", 0)
            paper_debate_rates.append(stats.get("debate_rate", 0))

    if total_dims == 0:
        return None

    return {
        "total_dimensions_evaluated": total_dims,
        "dimensions_triggered_debate": debated_dims,
        "overall_debate_rate": debated_dims / total_dims,
        "mean_paper_debate_rate": float(np.mean(paper_debate_rates)) if paper_debate_rates else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-approach evaluation
# ---------------------------------------------------------------------------

def evaluate_approach(approach_name, llm_dir, peerread_dir, venues, show_debate=False):
    """
    Evaluate one approach across venues.

    Returns dict: {venue -> [(dim, pearson_r, pearson_p, spearman_r, spearman_p, mae, bias, n), ...]}
    """
    print(f"\n{'#'*72}")
    print(f"# {approach_name}")
    print(f"# Output dir : {llm_dir}")
    print(f"{'#'*72}")

    all_results = {}

    for venue in venues:
        print(f"\n{'='*60}")
        print(f"  Venue: {venue}")
        print(f"{'='*60}")

        dimensions = VENUE_DIMENSIONS.get(venue, [])
        print(f"\n  [Score Alignment]")
        header = f"  {'Dimension':30s}  {'Pearson r':>10s}  {'Spearman ρ':>10s}  {'MAE':>6s}  {'Bias':>6s}  n"
        print(header)
        print("  " + "-" * (len(header) - 2))

        results = []
        for dim in dimensions:
            res = correlate_dimension(venue, peerread_dir, llm_dir, dim)
            if res is None:
                print(f"  {dim:30s}  -- insufficient data")
            else:
                pr, pp, sr, sp, mae, bias, n = res
                print(
                    f"  {dim:30s}  {pr:+.3f}{'*' if pp<0.05 else ' ':1s}       "
                    f"{sr:+.3f}{'*' if sp<0.05 else ' ':1s}       "
                    f"{mae:5.2f}  {bias:+5.2f}  {n}"
                )
                results.append((dim, pr, pp, sr, sp, mae, bias, n))

        if results:
            avg_p  = np.mean([r[1] for r in results])
            avg_sp = np.mean([r[3] for r in results])
            avg_mae = np.mean([r[5] for r in results])
            avg_bias = np.mean([r[6] for r in results])
            print(
                f"\n  {'AVERAGE':30s}  {avg_p:+.3f}            "
                f"{avg_sp:+.3f}            "
                f"{avg_mae:5.2f}  {avg_bias:+5.2f}"
            )
        all_results[venue] = results

        # Accept / reject
        ar = evaluate_accept_reject(venue, peerread_dir, llm_dir)
        if ar is not None:
            acc, prec, rec, f1, n, y_true, y_pred, common = ar
            print(f"\n  [Accept/Reject Prediction]  (RECOMMENDATION >= 5 → Accept)")
            print(f"  Accuracy  {acc:.3f}   Precision {prec:.3f}   Recall {rec:.3f}   F1 {f1:.3f}   n={n}")
            print(f"  (accepted={sum(y_true)}, rejected={n-sum(y_true)})")

            print(f"\n  Per-paper details:")
            gt_rec  = get_ground_truth_scores(venue, peerread_dir, "RECOMMENDATION")
            llm_rec = get_llm_scores(venue, llm_dir, "RECOMMENDATION")
            for k in common:
                idx = common.index(k)
                gt_dec  = "accept" if y_true[idx] else "reject"
                llm_dec = "accept" if y_pred[idx] else "reject"
                gt_s    = f"GT={gt_rec[k]:.1f}"    if k in gt_rec  else "GT=N/A"
                llm_s   = f"LLM={llm_rec[k]:.1f}" if k in llm_rec else "LLM=N/A"
                tag     = "OK" if gt_dec == llm_dec else "MISS"
                print(f"    {k}: {gt_s} {llm_s}  GT→{gt_dec}  LLM→{llm_dec}  [{tag}]")

        # Debate-specific stats
        if show_debate:
            stats = evaluate_debate_stats(venue, llm_dir)
            if stats:
                print(f"\n  [Debate Statistics]")
                print(f"  Referee triggered: {stats['dimensions_triggered_debate']} / "
                      f"{stats['total_dimensions_evaluated']} dim-evaluations "
                      f"({stats['overall_debate_rate']:.1%})")
                print(f"  Mean per-paper debate rate: {stats['mean_paper_debate_rate']:.1%}")

        # Lit-augmented-specific stats
        aug_stats = evaluate_augmentation_stats(venue, llm_dir)
        if aug_stats:
            print(f"\n  [Literature Survey Augmentation Coverage]")
            print(f"  Papers with survey: {aug_stats['papers_with_survey']} / "
                  f"{aug_stats['total_papers']} ({aug_stats['coverage_rate']:.1%})")
            print(f"  Mean survey pool size: {aug_stats['mean_pool_size']:.1f} papers")
            print(f"  Augmented dims per paper: {', '.join(aug_stats['augmented_dims'])}")

    return all_results


# ---------------------------------------------------------------------------
# Cross-approach comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(approach_results: dict, venues):
    """
    approach_results: {approach_label -> {venue -> [(dim, pearson_r, pearson_p, spearman_r, ...), ...]}}
    """
    labels = list(approach_results.keys())
    col_w = 13  # width per approach column

    print(f"\n{'#'*72}")
    print(f"# COMPARISON: {' vs '.join(labels)}")
    print(f"{'#'*72}")

    for venue in venues:
        print(f"\n  Venue: {venue}")

        # Header
        dim_col = 30
        hdr = f"  {'Dimension':^{dim_col}s}"
        for label in labels:
            hdr += f"  {'Pearson':>{col_w}s}"
        for label in labels:
            hdr += f"  {'Spearman':>{col_w}s}"
        for label in labels:
            hdr += f"  {'MAE':>{col_w}s}"
        print(hdr)

        sub = f"  {' '*dim_col}"
        for label in labels:
            sub += f"  {label[:col_w]:>{col_w}s}"
        for label in labels:
            sub += f"  {label[:col_w]:>{col_w}s}"
        for label in labels:
            sub += f"  {label[:col_w]:>{col_w}s}"
        print(sub)
        print("  " + "-" * (dim_col + len(labels) * 3 * (col_w + 2)))

        # Build lookup: label -> dim -> (pearson_r, spearman_r, mae)
        lookup = {}
        for label in labels:
            results = approach_results[label].get(venue, [])
            lookup[label] = {
                dim: (pr, sr, mae)
                for dim, pr, pp, sr, sp, mae, bias, n in results
            }

        all_dims = sorted({
            dim
            for label in labels
            for dim in lookup[label]
        })

        for dim in all_dims:
            row = f"  {dim:<{dim_col}s}"
            for label in labels:
                val = lookup[label].get(dim)
                row += f"  {(f'{val[0]:+.3f}' if val else 'N/A'):>{col_w}s}"
            for label in labels:
                val = lookup[label].get(dim)
                row += f"  {(f'{val[1]:+.3f}' if val else 'N/A'):>{col_w}s}"
            for label in labels:
                val = lookup[label].get(dim)
                row += f"  {(f'{val[2]:.3f}' if val else 'N/A'):>{col_w}s}"
            print(row)

        # Averages row
        row = f"  {'AVERAGE':<{dim_col}s}"
        for label in labels:
            vals = [v[0] for v in lookup[label].values()]
            row += f"  {(f'{np.mean(vals):+.3f}' if vals else 'N/A'):>{col_w}s}"
        for label in labels:
            vals = [v[1] for v in lookup[label].values()]
            row += f"  {(f'{np.mean(vals):+.3f}' if vals else 'N/A'):>{col_w}s}"
        for label in labels:
            vals = [v[2] for v in lookup[label].values()]
            row += f"  {(f'{np.mean(vals):.3f}' if vals else 'N/A'):>{col_w}s}"
        print(f"  {'-'*(dim_col + len(labels)*3*(col_w+2))}")
        print(row)


def print_grand_summary(approach_results: dict, venues):
    """Single table: one row per approach, macro-averaged across all venues and dimensions."""
    print(f"\n{'#'*72}")
    print(f"# GRAND SUMMARY  (macro-averaged across all venues & dimensions)")
    print(f"{'#'*72}")
    print(f"\n  {'Approach':<28s}  {'Pearson r':>10s}  {'Spearman ρ':>10s}  {'MAE':>6s}  {'Bias':>6s}")
    print("  " + "-" * 68)

    for label, venue_results in approach_results.items():
        all_p, all_sp, all_mae, all_bias = [], [], [], []
        for venue, results in venue_results.items():
            for dim, pr, pp, sr, sp, mae, bias, n in results:
                all_p.append(pr)
                all_sp.append(sr)
                all_mae.append(mae)
                all_bias.append(bias)
        if all_p:
            print(
                f"  {label:<28s}  {np.mean(all_p):+.3f}        "
                f"{np.mean(all_sp):+.3f}        "
                f"{np.mean(all_mae):5.2f}  {np.mean(all_bias):+5.2f}"
            )
        else:
            print(f"  {label:<28s}  -- no data")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_dataset_stats(peerread_dir, venues, approach_dirs):
    """Print a summary of the evaluation dataset before running metrics."""
    print(f"\n{'#'*72}")
    print(f"# EVALUATION DATASET OVERVIEW")
    print(f"{'#'*72}")

    total_papers = 0
    total_reviews = 0

    for venue in venues:
        papers_dir  = Path(peerread_dir) / venue / "test" / "parsed_pdfs"
        reviews_dir = Path(peerread_dir) / venue / "test" / "reviews"

        n_papers  = len(list(papers_dir.glob("*.json")))  if papers_dir.exists()  else 0
        n_reviews = len(list(reviews_dir.glob("*.json"))) if reviews_dir.exists() else 0

        # Count papers that have at least one numeric score for any dimension
        dims = VENUE_DIMENSIONS.get(venue, [])
        papers_with_gt = set()
        if reviews_dir.exists():
            for f in reviews_dir.glob("*.json"):
                data = json.load(open(f))
                for r in data.get("reviews", []):
                    if any(extract_numeric_score(r.get(d)) is not None for d in dims):
                        papers_with_gt.add(f.stem)
                        break

        # Accept/reject labels
        decisions = get_ground_truth_decisions(venue, peerread_dir)

        # Per-approach output counts
        approach_counts = {}
        for label, out_dir in approach_dirs.items():
            out_path = Path(out_dir) / venue / "test"
            n = len(list(out_path.glob("*_review.json"))) if out_path.exists() else 0
            approach_counts[label] = n

        print(f"\n  Venue : {venue}")
        print(f"  {'Papers in test split':<35s} {n_papers}")
        print(f"  {'Review files (human)':<35s} {n_reviews}")
        print(f"  {'Papers with ≥1 numeric GT score':<35s} {len(papers_with_gt)}")
        print(f"  {'Papers with accept/reject label':<35s} {len(decisions)}")
        print(f"  {'Dimensions evaluated':<35s} {', '.join(dims)}")
        print(f"\n  LLM output coverage:")
        for label, count in approach_counts.items():
            bar = "█" * count + "░" * max(0, n_papers - count)
            pct = f"{count/n_papers:.0%}" if n_papers else "N/A"
            print(f"    {label:<18s} {count:>3}/{n_papers}  ({pct})")

        total_papers  += n_papers
        total_reviews += n_reviews

    print(f"\n  {'─'*50}")
    print(f"  Total papers across all venues : {total_papers}")
    print(f"  Total review files             : {total_reviews}")


def main():
    peerread_dir   = "PeerRead/data"
    venues         = ["acl_2017", "conll_2016", "iclr_2017"]

    approach_dirs = {
        "Monolithic"   : "monolithic_prompts",
        "Single-Agent" : "single_agent_prompts",
        "Dim-Agents"   : "dimension_agent_prompts",
        "Lit-Augmented": "lit_augmented_agent_prompts",
        "Two-Stage"    : "two_stage_agent_prompts",
        "Debate"       : "debate_agent_prompts",
    }

    print_dataset_stats(peerread_dir, venues, approach_dirs)

    approach_results = {}

    for label, out_dir in approach_dirs.items():
        if not Path(out_dir).exists():
            print(f"\n[SKIP] {label}: output directory '{out_dir}' not found. "
                  f"Run the corresponding script first.")
            continue
        is_debate = (label == "Debate")
        approach_results[label] = evaluate_approach(
            label, out_dir, peerread_dir, venues, show_debate=is_debate
        )

    if len(approach_results) >= 2:
        print_comparison_table(approach_results, venues)

    print_grand_summary(approach_results, venues)

    print(f"\n  * = p < 0.05")


if __name__ == "__main__":
    main()
