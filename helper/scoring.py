from sklearn.metrics import f1_score

def f1_score_overall(y_true, y_pred, best_threshold=0.5, only_overall=True):
    
    y_pred[y_pred > best_threshold] = 1
    y_pred[y_pred <= best_threshold] = 0    
    
    class_scores = f1_score(y_true, y_pred, labels=[0, 1], average=None)
    overall_score = f1_score(y_true, y_pred, labels=[0, 1], average='weighted')
    
    return(class_scores, overall_score)