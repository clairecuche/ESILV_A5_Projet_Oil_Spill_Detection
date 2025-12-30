import numpy as np
import torch
from config import NUM_CLASSES_SEG

class SegmentationMetrics:
    """
    Calcule mIoU et mAcc selon les formules du paper LADOS (Section 4.4).
    Exclut le background (ID 0) du calcul du mIoU final.
    """
    def __init__(self, num_classes=NUM_CLASSES_SEG, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        if torch.is_tensor(pred): pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target): target = target.detach().cpu().numpy()
        
        pred, target = pred.flatten(), target.flatten()
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred, target = pred[mask], target[mask]
        
        # Optimisation : On utilise bincount pour la matrice de confusion (plus rapide que la boucle for)
        # Filtre les indices hors limites au cas où
        valid_indices = (target >= 0) & (target < self.num_classes) & \
                        (pred >= 0) & (pred < self.num_classes)
        
        category_indices = self.num_classes * target[valid_indices].astype(int) + \
                          pred[valid_indices].astype(int)
        
        self.confusion_matrix += np.bincount(
            category_indices, 
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
    
    def get_results(self):
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        
        # IoU
        denominator = TP + FP + FN
        iou = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), where=denominator != 0)
        
        # mIoU (Exclure Background ID 0 selon le paper)
        miou = np.nanmean(iou[1:])
        
        # Accuracy
        denominator_acc = TP + FN
        acc_per_class = np.divide(TP, denominator_acc, out=np.zeros_like(TP, dtype=float), where=denominator_acc != 0)
        macc = np.nanmean(acc_per_class)
        
        return {'class_iou': iou, 'mIoU': miou, 'mIoU_with_bg': np.nanmean(iou), 'mAcc': macc}