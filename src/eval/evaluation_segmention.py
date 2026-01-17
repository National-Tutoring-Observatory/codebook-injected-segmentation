import json
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from collections import Counter
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def load_json_results(path):
    """Loads segmentation results from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def load_csv_data(path):
    """Loads ground truth data from a CSV file."""
    return pd.read_csv(path)

def get_boundary_vector(num_utterances, boundary_indices):
    """
    Creates a binary vector where 1 indicates a boundary *after* that utterance.
    Length is num_utterances - 1 (internal boundaries).
    """
    if num_utterances <= 1:
        return []
    vec = [0] * (num_utterances - 1)
    for idx in boundary_indices:
        if 0 <= idx < len(vec):
            vec[idx] = 1
    return vec

def get_segments(num_utterances, boundary_indices):
    """
    Returns a list of segments, where each segment is a list of utterance indices.
    """
    sorted_boundaries = sorted([b for b in boundary_indices if 0 <= b < num_utterances - 1])
    segment_starts = [0] + [b + 1 for b in sorted_boundaries]
    segment_ends = [b + 1 for b in sorted_boundaries] + [num_utterances]
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        segments.append(list(range(start, end)))
    return segments

def get_majority_label(labels):
    """Returns the majority label from a list of labels, ignoring NaNs."""
    valid_labels = [l for l in labels if pd.notna(l) and str(l).strip() != '']
    if not valid_labels:
        return None
    return Counter(valid_labels).most_common(1)[0][0]

def calculate_segment_irr(segments, human_labels, ai_labels):
    """
    Calculates Cohen's Kappa at the segment level.
    Each segment gets a single label (majority vote) from Human and AI.
    """
    seg_human = []
    seg_ai = []
    
    for seg_indices in segments:
        h_labs = [human_labels[i] for i in seg_indices if i < len(human_labels)]
        a_labs = [ai_labels[i] for i in seg_indices if i < len(ai_labels)]
        
        h_maj = get_majority_label(h_labs)
        a_maj = get_majority_label(a_labs)
        
        # Only consider segments where both have a label? Or treat None as a label?
        # Standard IRR usually requires same label set. Let's treat None as 'None'.
        seg_human.append(str(h_maj))
        seg_ai.append(str(a_maj))
        
    return cohen_kappa_score(seg_human, seg_ai)

def calculate_entropy_metrics(segments, labels):
    """
    Calculates Entropy, Purity, and Normalized Entropy.
    """
    segment_entropies = []
    segment_purities = []
    all_valid_labels = [l for l in labels if pd.notna(l) and str(l).strip() != '']
    
    if not all_valid_labels:
        return 0.0, 0.0, 0.0
        
    # Global entropy
    global_counts = Counter(all_valid_labels)
    global_probs = [c/len(all_valid_labels) for c in global_counts.values()]
    global_ent = entropy(global_probs) if global_probs else 0.0
    
    for seg_indices in segments:
        seg_labels = [labels[i] for i in seg_indices if i < len(labels)]
        valid_seg_labels = [l for l in seg_labels if pd.notna(l) and str(l).strip() != '']
        
        if not valid_seg_labels:
            continue
            
        counts = Counter(valid_seg_labels)
        probs = [c/len(valid_seg_labels) for c in counts.values()]
        
        # Entropy
        ent = entropy(probs)
        segment_entropies.append(ent)
        
        # Purity
        most_common = counts.most_common(1)[0][1]
        purity = most_common / len(valid_seg_labels)
        segment_purities.append(purity)
        
    avg_entropy = np.mean(segment_entropies) if segment_entropies else 0.0
    avg_purity = np.mean(segment_purities) if segment_purities else 0.0
    norm_entropy = avg_entropy / global_ent if global_ent > 0 else 0.0
    
    return avg_entropy, avg_purity, norm_entropy

def calculate_js_divergence(segments, labels):
    """
    Calculates average JS divergence between adjacent segments.
    """
    js_divs = []
    
    # Get all unique labels to form probability vectors
    all_valid_labels = sorted(list(set([l for l in labels if pd.notna(l) and str(l).strip() != ''])))
    label_to_idx = {l: i for i, l in enumerate(all_valid_labels)}
    num_classes = len(all_valid_labels)
    
    if num_classes < 2:
        return 0.0
        
    prev_dist = None
    
    for seg_indices in segments:
        seg_labels = [labels[i] for i in seg_indices if i < len(labels)]
        valid_seg_labels = [l for l in seg_labels if pd.notna(l) and str(l).strip() != '']
        
        if not valid_seg_labels:
            current_dist = None
        else:
            counts = Counter(valid_seg_labels)
            dist = np.zeros(num_classes)
            for l, c in counts.items():
                dist[label_to_idx[l]] = c
            dist = dist / np.sum(dist)
            current_dist = dist
            
        if prev_dist is not None and current_dist is not None:
            # JS Divergence
            js = jensenshannon(prev_dist, current_dist)
            # JS distance is returned by scipy, square it for divergence? 
            # Usually JS divergence is the square of the metric.
            # But "JS distance" is often what people mean. Let's return the metric (0-1).
            js_divs.append(js)
            
        prev_dist = current_dist
        
    return np.mean(js_divs) if js_divs else 0.0

def calculate_segment_human_ai_js_divergence(segments, human_labels, ai_labels, include_none=False):
    """
    Calculates average JS divergence between Human and AI label distributions within each segment.
    """
    js_divs = []
    
    # Get all unique labels to form probability vectors
    # Combine both to ensure same support
    all_labels = human_labels + ai_labels
    if not include_none:
        valid_labels = sorted(list(set([l for l in all_labels if pd.notna(l) and str(l).strip() != ''])))
    else:
        # treat NaNs as a label if include_none is True
        valid_labels = sorted(list(set([str(l) if pd.notna(l) else "None" for l in all_labels])))

    if not valid_labels:
        return 0.0

    label_to_idx = {l: i for i, l in enumerate(valid_labels)}
    num_classes = len(valid_labels)
    
    if num_classes < 2:
        return 0.0

    for seg_indices in segments:
        h_labs = [human_labels[i] for i in seg_indices if i < len(human_labels)]
        a_labs = [ai_labels[i] for i in seg_indices if i < len(ai_labels)]
        
        # Filter/Process labels
        if not include_none:
            h_proc = [l for l in h_labs if pd.notna(l) and str(l).strip() != '']
            a_proc = [l for l in a_labs if pd.notna(l) and str(l).strip() != '']
        else:
            h_proc = [str(l) if pd.notna(l) else "None" for l in h_labs]
            a_proc = [str(l) if pd.notna(l) else "None" for l in a_labs]
            
        if not h_proc or not a_proc:
             continue

        h_counts = Counter(h_proc)
        a_counts = Counter(a_proc)
        
        h_dist = np.zeros(num_classes)
        a_dist = np.zeros(num_classes)
        
        for l, c in h_counts.items():
            if l in label_to_idx:
                h_dist[label_to_idx[l]] = c
        
        for l, c in a_counts.items():
            if l in label_to_idx:
                a_dist[label_to_idx[l]] = c
                
        # Normalize
        if np.sum(h_dist) > 0:
            h_dist = h_dist / np.sum(h_dist)
        if np.sum(a_dist) > 0:
            a_dist = a_dist / np.sum(a_dist)
            
        # JS Divergence
        js = jensenshannon(h_dist, a_dist)
        js_divs.append(js)
            
    return np.mean(js_divs) if js_divs else 0.0

def calculate_boundary_change_rate(boundary_indices, labels):
    """
    Calculates % of boundaries where label[i] != label[i+1].
    """
    if not boundary_indices:
        return 0.0
        
    changes = 0
    total = 0
    
    for b in boundary_indices:
        if b < len(labels) - 1:
            l1 = labels[b]
            l2 = labels[b+1]
            
            s1 = str(l1) if pd.notna(l1) else ""
            s2 = str(l2) if pd.notna(l2) else ""
            
            if s1 != s2:
                changes += 1
            total += 1
            
    return changes / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Compare and Evaluate Segmentation Results")
    parser.add_argument("--file1", required=True, help="Path to first JSON results file (Reference/Model 1)")
    parser.add_argument("--file2", required=True, help="Path to second JSON results file (Hypothesis/Model 2)")
    parser.add_argument("--csv", required=True, help="Path to Ground Truth CSV")
    parser.add_argument("--output", default="evaluation_report.txt", help="Output report path")
    
    parser.add_argument("--include_none", action="store_true", help="Include 'None' labels in JS divergence calculation")
    
    args = parser.parse_args()
    
    json1 = load_json_results(args.file1)
    json2 = load_json_results(args.file2)
    df = load_csv_data(args.csv)
    
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())
    common_keys = list(keys1.intersection(keys2))
    
    report = []
    report.append("Segmentation Evaluation Report (Advanced Metrics)")
    report.append("===============================================")
    report.append(f"File 1: {os.path.basename(args.file1)}")
    report.append(f"File 2: {os.path.basename(args.file2)}")
    report.append(f"CSV: {os.path.basename(args.csv)}")
    report.append(f"Common Dialogues: {len(common_keys)}")
    report.append(f"Include None in JS Div: {args.include_none}")
    report.append("")
    
    # Metrics Storage
    # Dimension 1: IRR
    utterance_human_labels = []
    utterance_ai_labels = []
    
    # Global Segment Labels for IRR
    seg1_human_all = []
    seg1_ai_all = []
    seg2_human_all = []
    seg2_ai_all = []
    
    # Dimension 2: Cluster Quality (Using Human Labels)
    metrics1 = {'entropy': [], 'purity': [], 'norm_entropy': [], 'js_div': [], 'change_rate': [], 'human_ai_js': []}
    metrics2 = {'entropy': [], 'purity': [], 'norm_entropy': [], 'js_div': [], 'change_rate': [], 'human_ai_js': []}
    
    for filename in common_keys:
        try:
            uuid = filename.replace("dialogue_", "").replace(".txt", "")
            if 'session_id' in df.columns:
                 dialogue_df = df[df['session_id'] == uuid]
            else:
                 dialogue_df = pd.DataFrame()
        except:
            continue
            
        if dialogue_df.empty:
            continue
            
        if 'MessageSequence' in dialogue_df.columns:
            dialogue_df = dialogue_df.sort_values('MessageSequence')
            
        human_labels = dialogue_df['Human'].tolist() if 'Human' in dialogue_df.columns else []
        ai_labels = dialogue_df['AI'].tolist() if 'AI' in dialogue_df.columns else []
        
        # Accumulate for Utterance IRR
        utterance_human_labels.extend([str(l) for l in human_labels])
        utterance_ai_labels.extend([str(l) for l in ai_labels])
        
        num_utts = len(dialogue_df)
        b1 = json1[filename].get('boundary_indices', [])
        b2 = json2[filename].get('boundary_indices', [])
        
        # Get Segments
        segs1 = get_segments(num_utts, b1)
        segs2 = get_segments(num_utts, b2)
        
        # Dimension 1: Accumulate Segment Labels
        def accumulate_seg_labels(segments, h_labels, a_labels, dest_h, dest_a):
            for seg_indices in segments:
                h_labs = [h_labels[i] for i in seg_indices if i < len(h_labels)]
                a_labs = [a_labels[i] for i in seg_indices if i < len(a_labels)]
                h_maj = get_majority_label(h_labs)
                a_maj = get_majority_label(a_labs)
                if h_maj and a_maj:
                    dest_h.append(str(h_maj))
                    dest_a.append(str(a_maj))

        if human_labels and ai_labels:
            accumulate_seg_labels(segs1, human_labels, ai_labels, seg1_human_all, seg1_ai_all)
            accumulate_seg_labels(segs2, human_labels, ai_labels, seg2_human_all, seg2_ai_all)
            
        # Dimension 2: Cluster Quality (Human Labels)
        if human_labels:
            e1, p1, ne1 = calculate_entropy_metrics(segs1, human_labels)
            js1 = calculate_js_divergence(segs1, human_labels)
            cr1 = calculate_boundary_change_rate(b1, human_labels)
            
            metrics1['entropy'].append(e1)
            metrics1['purity'].append(p1)
            metrics1['norm_entropy'].append(ne1)
            metrics1['js_div'].append(js1)
            metrics1['change_rate'].append(cr1)
            
            e2, p2, ne2 = calculate_entropy_metrics(segs2, human_labels)
            js2 = calculate_js_divergence(segs2, human_labels)
            cr2 = calculate_boundary_change_rate(b2, human_labels)
            
            metrics2['entropy'].append(e2)
            metrics2['purity'].append(p2)
            metrics2['norm_entropy'].append(ne2)
            metrics2['js_div'].append(js2)
            metrics2['change_rate'].append(cr2)
            
            # New Metric: Human vs AI JS Divergence
            if ai_labels:
                hajs1 = calculate_segment_human_ai_js_divergence(segs1, human_labels, ai_labels, args.include_none)
                metrics1['human_ai_js'].append(hajs1)
                
                hajs2 = calculate_segment_human_ai_js_divergence(segs2, human_labels, ai_labels, args.include_none)
                metrics2['human_ai_js'].append(hajs2)

    # Report Generation
    def avg(l): return np.mean(l) if l else 0.0

    report.append("Dimension 1: Cluster Quality (Human Labels)")
    report.append("-------------------------------------------")
    report.append(f"{'Metric':<30} | {'File 1':<10} | {'File 2':<10}")
    report.append("-" * 56)
    report.append(f"{'Avg Segment Entropy (Low)':<30} | {avg(metrics1['entropy']):<10.4f} | {avg(metrics2['entropy']):<10.4f}")
    report.append(f"{'Avg Segment Purity (High)':<30} | {avg(metrics1['purity']):<10.4f} | {avg(metrics2['purity']):<10.4f}")
    report.append(f"{'Normalized Entropy (Low)':<30} | {avg(metrics1['norm_entropy']):<10.4f} | {avg(metrics2['norm_entropy']):<10.4f}")
    report.append(f"{'Avg JS Divergence (High)':<30} | {avg(metrics1['js_div']):<10.4f} | {avg(metrics2['js_div']):<10.4f}")
    report.append(f"{'Boundary Change Rate (High)':<30} | {avg(metrics1['change_rate']):<10.4f} | {avg(metrics2['change_rate']):<10.4f}")
    report.append(f"{'Human-AI JS Div (Low)':<30} | {avg(metrics1['human_ai_js']):<10.4f} | {avg(metrics2['human_ai_js']):<10.4f}")
    
    with open(args.output, 'w') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {args.output}")
    print("\n".join(report))

if __name__ == "__main__":
    main()
