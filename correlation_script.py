import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import re

# Numeric dimensions to correlate per venue
# ACL and CoNLL share the same schema; ICLR has fewer numeric fields in ground truth
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


def extract_numeric_score(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.match(r"(-?\d+(?:\.\d+)?)", val.strip())
        if m:
            return float(m.group(1))
    return None


def get_ground_truth_scores(venue, peerread_dir, dimension):
    """
    For each paper, compute the average reviewer score for `dimension`.
    For ICLR: use only annotated (scored) reviews (skip comment-only posts).
    Returns: dict {paper_id: avg_score}
    """
    gt_scores = {}
    reviews_dir = Path(peerread_dir) / venue / "test" / "reviews"
    if not reviews_dir.exists():
        return gt_scores
    for f in reviews_dir.glob("*.json"):
        with open(f) as fin:
            data = json.load(fin)
        paper_id = f.stem
        reviews = data.get("reviews", [])
        scores = []
        for r in reviews:
            val = extract_numeric_score(r.get(dimension))
            if val is not None:
                scores.append(val)
        if scores:
            gt_scores[paper_id] = np.mean(scores)
    return gt_scores


def get_llm_scores(venue, llm_dir, dimension):
    """
    For each paper, get the LLM's score for `dimension` from the output JSON.
    Returns: dict {paper_id: score}
    """
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


def correlate_dimension(venue, peerread_dir, llm_dir, dimension):
    """Compute Pearson correlation for a single dimension. Returns (corr, p, n) or None."""
    gt_scores = get_ground_truth_scores(venue, peerread_dir, dimension)
    llm_scores = get_llm_scores(venue, llm_dir, dimension)

    common = set(gt_scores) & set(llm_scores)
    if len(common) < 2:
        return None

    gt = [gt_scores[k] for k in sorted(common)]
    llm = [llm_scores[k] for k in sorted(common)]

    # Check for zero variance (pearsonr will fail)
    if np.std(gt) == 0 or np.std(llm) == 0:
        return None

    corr, p = pearsonr(gt, llm)
    return corr, p, len(common)


def get_ground_truth_decisions(venue, peerread_dir):
    """
    Get accept/reject ground truth for each paper.
    ICLR: 'accepted' field in review JSON.
    ACL: title lookup against acl_accepted.txt.
    CoNLL: no labels available.
    Returns: dict {paper_id: bool}
    """
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
                title = data.get("title", "")
                decisions[f.stem] = title in accepted_titles

    return decisions


def evaluate_accept_reject(venue, peerread_dir, llm_dir, threshold=5):
    """
    Evaluate accept/reject prediction using LLM RECOMMENDATION score.
    Papers with RECOMMENDATION >= threshold are predicted as accepted.
    Returns (accuracy, precision, recall, f1, n) or None.
    """
    gt_decisions = get_ground_truth_decisions(venue, peerread_dir)
    llm_scores = get_llm_scores(venue, llm_dir, "RECOMMENDATION")

    common = sorted(set(gt_decisions) & set(llm_scores))
    if len(common) < 2:
        return None

    y_true = [int(gt_decisions[k]) for k in common]
    y_pred = [int(llm_scores[k] >= threshold) for k in common]
    n = len(common)

    # Accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    # Precision, Recall, F1 for accept (positive class)
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, n, y_true, y_pred, common


def evaluate_approach(approach_name, llm_dir, peerread_dir, venues):
    """Evaluate a single approach (monolithic or dimension agents) across all venues."""
    print(f"\n{'#'*70}")
    print(f"# {approach_name}")
    print(f"# LLM output dir: {llm_dir}")
    print(f"{'#'*70}")

    all_results = {}

    for venue in venues:
        print(f"\n{'='*60}")
        print(f"Venue: {venue}")
        print(f"{'='*60}")

        # --- Per-dimension Pearson correlation ---
        print("\n  [Score Alignment: Pearson Correlation]")
        dimensions = VENUE_DIMENSIONS.get(venue, [])
        results = []

        for dim in dimensions:
            result = correlate_dimension(venue, peerread_dir, llm_dir, dim)
            if result is None:
                print(f"  {dim:30s}  -- not enough data (< 2 overlapping papers with scores)")
            else:
                corr, p, n = result
                print(f"  {dim:30s}  r={corr:+.3f}  p={p:.3g}  n={n}")
                results.append((dim, corr, p, n))

        if results:
            avg_corr = np.mean([r[1] for r in results])
            print(f"\n  {'Average correlation':30s}  r={avg_corr:+.3f}  (across {len(results)} dimensions)")

        all_results[venue] = results

        # --- Accept/Reject prediction (only for monolithic which has RECOMMENDATION) ---
        ar_result = evaluate_accept_reject(venue, peerread_dir, llm_dir, threshold=5)
        if ar_result is not None:
            accuracy, precision, recall, f1, n, y_true, y_pred, common = ar_result
            print(f"\n  [Accept/Reject Prediction]")
            print(f"  Threshold: RECOMMENDATION >= 5 -> Accept")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1:        {f1:.3f}")
            print(f"  n={n} (accepted={sum(y_true)}, rejected={n - sum(y_true)})")

            print(f"\n  Per-paper details:")
            gt_rec = get_ground_truth_scores(venue, peerread_dir, "RECOMMENDATION")
            llm_rec = get_llm_scores(venue, llm_dir, "RECOMMENDATION")
            for k in common:
                gt_dec = "accept" if y_true[common.index(k)] else "reject"
                llm_dec = "accept" if y_pred[common.index(k)] else "reject"
                gt_score = f"GT_rec={gt_rec[k]:.2f}" if k in gt_rec else "GT_rec=N/A"
                llm_score = f"LLM_rec={llm_rec[k]:.2f}" if k in llm_rec else "LLM_rec=N/A"
                match = "OK" if gt_dec == llm_dec else "MISS"
                print(f"    Paper {k}: {gt_score}, {llm_score}, GT={gt_dec}, LLM={llm_dec} [{match}]")

    return all_results


def print_comparison_table(mono_results, dim_results, venues):
    """Print a side-by-side comparison table of monolithic vs dimension agents."""
    print(f"\n{'#'*70}")
    print(f"# COMPARISON: Monolithic vs Dimension Agents")
    print(f"{'#'*70}")

    for venue in venues:
        print(f"\n  Venue: {venue}")
        print(f"  {'Dimension':30s}  {'Monolithic':>12s}  {'Dim Agent':>12s}  {'Delta':>10s}")
        print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*10}")

        mono_map = {dim: corr for dim, corr, p, n in mono_results.get(venue, [])}
        dim_map = {dim: corr for dim, corr, p, n in dim_results.get(venue, [])}

        all_dims = sorted(set(list(mono_map.keys()) + list(dim_map.keys())))
        for dim in all_dims:
            m = mono_map.get(dim)
            d = dim_map.get(dim)
            m_str = f"{m:+.3f}" if m is not None else "N/A"
            d_str = f"{d:+.3f}" if d is not None else "N/A"
            if m is not None and d is not None:
                delta = d - m
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "—"
            print(f"  {dim:30s}  {m_str:>12s}  {d_str:>12s}  {delta_str:>10s}")

        # Averages
        mono_avg = np.mean(list(mono_map.values())) if mono_map else None
        dim_avg = np.mean(list(dim_map.values())) if dim_map else None
        m_avg_str = f"{mono_avg:+.3f}" if mono_avg is not None else "N/A"
        d_avg_str = f"{dim_avg:+.3f}" if dim_avg is not None else "N/A"
        if mono_avg is not None and dim_avg is not None:
            delta_avg_str = f"{dim_avg - mono_avg:+.3f}"
        else:
            delta_avg_str = "—"
        print(f"  {'AVERAGE':30s}  {m_avg_str:>12s}  {d_avg_str:>12s}  {delta_avg_str:>10s}")


def main():
    peerread_dir = "PeerRead/data"
    monolithic_dir = "monolithic_prompts"
    dimension_dir = "dimension_agent_prompts"

    venues = ["acl_2017", "conll_2016", "iclr_2017"]

    # Evaluate both approaches
    mono_results = evaluate_approach("MONOLITHIC BASELINE", monolithic_dir, peerread_dir, venues)

    # Only run dimension agent eval if the output directory exists
    dim_results = {}
    if Path(dimension_dir).exists():
        dim_results = evaluate_approach("DIMENSION AGENTS", dimension_dir, peerread_dir, venues)
        print_comparison_table(mono_results, dim_results, venues)
    else:
        print(f"\nDimension agent output not found at '{dimension_dir}'. "
              f"Run dimension_agents.py first to generate outputs.")


if __name__ == "__main__":
    main()
