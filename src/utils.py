from sklearn.metrics import roc_auc_score

def weighted_auc(y_true, y_pred, wieghts):
  return roc_auc_score(y_true, y_pred, sample_weight=weights)