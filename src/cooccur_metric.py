import numpy as np
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import warnings


def conditional_t(y_pred, y_gt, y_gt_mask, thresh=0.5, avg=True):
    n_samples, n_classes = y_pred.shape
    
    assert y_pred.shape == y_gt.shape
    
    prec, rec = np.ones((n_classes, n_classes))*-1.0, np.ones((n_classes, n_classes))*-1.0
    maps = np.ones((n_classes, n_classes))*-1.0
    n_occurs = np.zeros((n_classes, n_classes))
    
    for c_j in range(n_classes):
        y_gt_sub = y_gt[y_gt_mask[:, c_j] == 1]  # contains the subset of samples where c_j exists
        y_pred_sub = y_pred[y_gt_mask[:, c_j] == 1]
        
        pr_j, re_j, f1_j, n_j = precision_recall_fscore_support(y_gt_sub, (y_pred_sub >= thresh).astype(np.uint8), average=None)
        map_j = average_precision_score(y_gt_sub, y_pred_sub, average=None)
        
        for c_i in range(n_classes):
            if c_i == c_j:
                continue
            if n_j[c_i] == 0:
                continue
            
            n_occurs[c_i, c_j] = n_j[c_i]
            if np.isnan(pr_j[c_i]) or np.isnan(re_j[c_i]):
                prec[c_i, c_j] = 0
                rec[c_i, c_j] = 0
            else:
                prec[c_i, c_j] = pr_j[c_i]
                rec[c_i, c_j] = re_j[c_i]
                
            if np.isnan(map_j[c_i]):
                maps[c_i, c_j] = 0
            else:
                maps[c_i, c_j] = map_j[c_i]

    if avg:
        return np.mean(prec[prec != -1]), np.mean(rec[rec != -1]), n_occurs, np.mean(maps[maps != -1])
    else:
        return prec, rec, n_occurs, maps


def avg_scores(score):
    return np.mean(score[score >= 0])*100


def get_f1(prec, rec):
    return 2*(prec*rec)/(prec+rec+1e-9)


def standard_metric(y_pred, y_gt, thresh=0.5):
    y_pred = np.concatenate(y_pred, 0)
    
    y_gt = np.concatenate(y_gt, 0)
    
    pr, re, _, n = precision_recall_fscore_support(y_gt, (y_pred >= thresh).astype(np.uint8), average=None)
    maps = average_precision_score(y_gt, y_pred, average=None)
    
    return pr, re, n, maps
    


def conditional_metric(y_pred, y_gt, t=0, thresh=0.5, avg=True):
    """
    y_pred is a list of un-thresholded predictions [(T1, C), (T2, C), ...]. Each element of the list is a different video, where the shape is (Time, #Classes).
    y_gt is a list of binary ground-truth labels [(T1, C), (T2, C), ...]. Each element of the list is a different video, where the shape is (Time, #Classes).
    t is an integer. If =0, measures in-timestep coocurrence. If >0, it measures conditional score of succeeding 
        actions (i.e. if c_i follows c_j). If <0 it measure conditional score of preceeding actions (i.e. if c_i preceeds c_j).
    thresh is a value in range (0, 1) which binarizes the predicted probabilities
    avg determines whether it returns a single score or class-wise scores
    
    Returns
    
    prec: the action-conditional precision score
    rec: the action-conditional recall score
    n_s: the number of samples for the pair of actions. Has shape (#Classes, #Classes).
    map: the action-conditional mAP score
    
    """
    y_pred = np.concatenate(y_pred, 0)
    
    if t == 0:
        y_gt = np.concatenate(y_gt, 0).astype(np.uint8)
        
        return conditional_t(y_pred, y_gt, y_gt, thresh, avg)
    else:    
        y_gt_mask = []
        for vid_y_gt in y_gt:
            
            if t > 0:  # looks at previous t time-steps
                cumsum = np.cumsum(vid_y_gt, 0)
                rolled = np.roll(cumsum, t, 0)
                
                rolled[:t] = 0
                n_in_last_t = cumsum-rolled
            else:  # looks at next 0-t time-steps
                vid_y_gt_flipped = np.flip(vid_y_gt, 0)
                
                cumsum = np.cumsum(vid_y_gt_flipped, 0)
                rolled = np.roll(cumsum, t, 0)
                
                rolled[:0-t] = 0
                n_in_last_t = cumsum-rolled
                
                n_in_last_t = np.flip(n_in_last_t, 0)
            
                
            n_in_last_t = np.clip(n_in_last_t, 0, 1)
            masked = n_in_last_t - vid_y_gt
            # 1: present before/after, but not in current
            # 0: present before/after and in current, or not present before/after and not in current
            # -1: not present before/after and in current
            
            masked = np.clip(masked, 0, 1)
            y_gt_mask.append(masked)
            
        y_gt = np.concatenate(y_gt, 0).astype(np.uint8)
        y_gt_mask = np.concatenate(y_gt_mask, 0).astype(np.uint8)
        
        return conditional_t(y_pred, y_gt, y_gt_mask, thresh, avg)
    
  


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    gt_labels = [np.random.randint(0, 2, (100, 5)), np.random.randint(0, 2, (200, 5))]
    pred_probs = [np.random.uniform(0, 1, (100, 5)), np.random.uniform(0, 1, (200, 5))]
    
    prec0, re0, ns0, map0 = conditional_metric(pred_probs, gt_labels, t=0, avg=True)
    fs0 = get_f1(prec0, re0) # action conditional f1-score
    
    print('Precision(c_i|c_j,0)=', prec0)
    print('Recall(c_i|c_j,0)=', re0)
    print('F1Score(c_i|c_j,0)=', map0)
    print('mAP(c_i|c_j,0)=', fs0)
    print()
    
    prec20, re20, ns20, map20 = conditional_metric(pred_probs, gt_labels, t=20, avg=True)
    fs20 = get_f1(prec20, re20) # action conditional f1-score

    print('Precision(c_i|c_j,20)=', prec20)
    print('Recall(c_i|c_j,20)=', re20)
    print('F1Score(c_i|c_j,20)=', map20)
    print('mAP(c_i|c_j,20)=', fs20)


