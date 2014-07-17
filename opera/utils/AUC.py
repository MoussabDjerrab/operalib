import numpy as np

def AUC(xdata, ydata):
    """
    Given a list of x coordinates and a list of y coordinates, returns
    the area under the curve they define.
    """
    x = (np.roll(xdata, -1) - xdata)[:-1]
    y = (np.roll(ydata, -1) + ydata)[:-1]/2.
    return sum(map(lambda x, y: x*y, x, y))


def calc_auc_pr(pred,labels) : 
    """Compute AUROC and AUPR
        labels    : ([n] vector) true labels 0 ou 1
        pred      : ([n] vector) continuous values predicted by the classifier 
        AUTHOR    : C?line BROUARD
    """
    nb_pos = sum(labels == 1)
    nb_neg = sum(labels == 0)
    nb_tot = nb_pos + nb_neg
    #pred is sorted by decreasing order
    predf = np.sort(pred)*1.
    idx = np.argsort(pred)
    predf = predf[::-1]
    idx = idx[::-1]
    #labels is sorted according to idx
    labelsf = labels[idx]
    
    tp = np.cumsum(labelsf)*1.# true positive
    fp = (range(nb_tot) - tp)+1.# false positive
    flags = (np.diff(predf) != False) #identical thresholds are removed
    tpr = tp[flags] / nb_pos
    fpr = fp[flags] / nb_neg
    
    tpr = np.concatenate(([0],tpr,[1]))
    fpr = np.concatenate(([0],fpr,[1]))
    
    auc_roc = sum((fpr[1:]-fpr[:fpr.size-1])*(tpr[1:]+tpr[:tpr.size-1]))/2.;

    #Transform ROC curve points into PR space
    recall = tp / nb_pos
    precision = tp / (tp + fp)
    
    if (sum(np.diff(tp)>1) > 0) : 
        recall_fin = np.zeros(nb_pos)
        precision_fin = np.zeros(nb_pos)
        index = 0;
        #
        #Add in-between PR points
        #
        for i in range(tpr.size-1) : 
            recall_fin[index] = recall(i);
            precision_fin[index] = precision(i);
            index = index + 1
            if (tp[i+1] - tp[i] > 1) :     
                TPA = tp[i]
                FPA = fp[i]
                TPB = tp[i+1]
                FPB = fp[i+1]
                x = 1
                while (x < (TPB - TPA)) : 
                    recall_fin[index] = (TPA + x) / nb_pos
                    precision_fin[index] = (TPA + x) / (1.*TPA + x + FPA + (FPB - FPA)/(TPB - TPA) * x) 
                    x = x + 1
                    index = index + 1
        #
        recall_fin[index] = recall[recall.size-1]
        precision_fin[index] = precision[recall.size-1]
        recall_fin2 = recall_fin[:index]
        precision_fin2 = precision_fin[:index]
        recall_fin2 = np.concatenate(([0],recall_fin2,[1]))
        precision_fin2 = np.concatenate(([1],precision_fin2,[0]))
    else : 
        recall_fin2 = np.concatenate(([0],recall,[1]))
        precision_fin2 = np.concatenate(([1],precision,[0]))
    #   
    auc_pr =  sum((recall_fin2[1:] - recall_fin2[:recall_fin2.size-1])*(precision_fin2[1:] + precision_fin2[:precision_fin2.size-1]))/2
    return(auc_roc,auc_pr)