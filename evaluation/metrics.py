import numpy as np

def IoU(tp, fp, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

def mIoU(classes_tp_fp_fn):
    ious = [IoU(tp, fp, fn) for tp, fp, fn in classes_tp_fp_fn]
    return sum(ious) / len(ious) if ious else 0.0

def weighted_f1(classes_tp_fp_fn_support):
    total = sum(s for _, _, _, s in classes_tp_fp_fn_support)
    if total == 0: return 0.0
    f1s = []
    for tp, fp, fn, support in classes_tp_fp_fn_support:
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1 * (support / total))
    return sum(f1s)

def Inst(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0

def Tool(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0

def ArgN(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0

def ArgV(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0

def Summ(rouge_l_f1):
    return rouge_l_f1 * 100

def TSR(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0

def HRR(correctly_rejected, total_unsolvable):
    return (correctly_rejected / total_unsolvable) * 100 if total_unsolvable > 0 else 0.0

def MGAR(all_subtasks_correct, total):
    return (all_subtasks_correct / total) * 100 if total > 0 else 0.0

def RSS(unsafe_segments, total_segments):
    return (unsafe_segments / total_segments) * 100 if total_segments > 0 else 0.0

def MTCS(consistent_pairs, total_pairs):
    return (consistent_pairs / total_pairs) * 100 if total_pairs > 0 else 0.0

def TLS(gt_calls, actual_calls):
    return gt_calls / actual_calls if actual_calls > 0 else 0.0

def CRR(correct_resolutions, total_conflicts):
    return (correct_resolutions / total_conflicts) * 100 if total_conflicts > 0 else 0.0

def DDF1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

def CCQ(retained_facts, total_facts):
    return retained_facts / total_facts if total_facts > 0 else 1.0

def CDF1(tp, fp, fn):
    return DDF1(tp, fp, fn)

def ReSR(safe_replans, total_detected_conflicts):
    return (safe_replans / total_detected_conflicts) * 100 if total_detected_conflicts > 0 else 0.0
