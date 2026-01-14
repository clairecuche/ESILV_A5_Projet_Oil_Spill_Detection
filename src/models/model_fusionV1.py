"""
Fusion de YOLOv11 et SegFormer pour la détection d'oil spills.

Stratégies implémentées:
1. Weighted Average (simple)
2. Class-Specific Fusion (liquides vs solides)
3. Confidence-Based Selection
4. Ensemble Voting

Basé sur le paper LADOS et le State of the Art.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loaders import get_dataloaders
from config import *
from metrics import SegmentationMetrics


class ModelFusion:
    """
    Classe pour fusionner les prédictions de YOLOv11 et SegFormer.
    
    Stratégies de fusion:
    - weighted_average: Moyenne pondérée des probabilités
    - class_specific: SegFormer pour liquides, YOLO pour solides
    - confidence_based: Sélection basée sur la confiance
    - voting: Vote majoritaire par pixel
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        segformer_model_path: str,
        device: str = DEVICE
    ):
        """
        Initialise les deux modèles.
        
        Args:
            yolo_model_path: Chemin vers best.pt de YOLO
            segformer_model_path: Chemin vers best_model.pt de SegFormer
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device)
        
        # 1. Chargement de YOLOv11
        print(f"🔄 Chargement de YOLOv11 depuis {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 2. Chargement de SegFormer
        print(f"🔄 Chargement de SegFormer depuis {segformer_model_path}...")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME_SEG, 
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Chargement des poids entraînés (gestion de l'état du dictionnaire)
        checkpoint = torch.load(segformer_model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # Nettoyage si le modèle a été sauvegardé avec 'torch.compile'
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.seg_model.load_state_dict(state_dict)
        self.seg_model.eval()
        
        # Processor pour SegFormer
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME_SEG)
        
        # Paramètres de dénormalisation
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Classes liquides vs solides (selon analyse de complémentarité)
        self.liquid_classes = [1, 2, 3]  # Oil, Emulsion, Sheen
        self.solid_classes = [4, 5]      # Ship, Oil-platform
        
        print("✅ Modèles chargés avec succès!")
    
    def predict_yolo(self, image: np.ndarray, conf: float = 0.3, iou: float = 0.45):
        """
        Prédiction YOLO (instance → semantic).
        
        Args:
            image: Image RGB uint8 (H, W, 3)
            conf: Seuil de confiance
            iou: Seuil NMS
            
        Returns:
            mask: Masque sémantique (H, W) avec classes 0-5
            probs: Probabilités par classe (H, W, 6)
        """
        H, W = image.shape[:2]

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Prédiction YOLO
        results = self.yolo_model.predict(
            image,
            verbose=False,
            device=self.device,
            imgsz=TARGET_SIZE[0],
            conf=conf,
            iou=iou
        )[0]
        
        # Initialiser masque et probabilités
        mask = np.zeros((H, W), dtype=np.int64)
        probs = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)
        probs[:, :, 0] = 1.0  # Background par défaut
        confidence_map = np.zeros((H, W), dtype=np.float32)
        
        if results.masks is None:
            return mask, probs
        
        # Convertir instances → semantic
        masks_data = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        for inst_mask, yolo_cls, conf in zip(masks_data, classes, confidences):
            # Resize et binariser
            inst_mask_resized = cv2.resize(inst_mask, (W, H), interpolation=cv2.INTER_LINEAR)
            binary_mask = inst_mask_resized > 0.5
            
            # Mapping YOLO (0-4) → GT (1-5)
            gt_cls = yolo_cls + 1
            
            # Mettre à jour là où la confiance est plus haute
            update_pixels = binary_mask & (conf > confidence_map)
            mask[update_pixels] = gt_cls
            
            # Probabilités: mettre 0 sur background, conf sur la classe
            probs[update_pixels, 0] = 0.0
            probs[update_pixels, gt_cls] = conf
            confidence_map[update_pixels] = conf
        
        # Renormaliser les probabilités
        probs_sum = probs.sum(axis=2, keepdims=True)
        probs_sum[probs_sum == 0] = 1.0  # Éviter division par 0
        probs = probs / probs_sum
        
        return mask, probs
    
    def predict_segformer(self, image_tensor: torch.Tensor):
        """
        Prédiction SegFormer.
        
        Args:
            image_tensor: Tensor normalisé (3, H, W)
            
        Returns:
            mask: Masque sémantique (H, W) avec classes 0-5
            probs: Probabilités par classe (H, W, 6)
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.seg_model(pixel_values=image_tensor.unsqueeze(0).to(self.device))
            logits = outputs.logits
            
            # Upsample vers taille originale
            logits_upsampled = F.interpolate(
                logits,
                size=image_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Softmax pour probabilités
            probs_tensor = F.softmax(logits_upsampled, dim=1)  # (1, 6, H, W)
            probs = probs_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 6)
            
            # Masque argmax
            mask = probs_tensor.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        
        return mask, probs
    
    # ========================================================================
    # STRATÉGIES DE FUSION
    # ========================================================================
    
    def fuse_weighted_average(
        self,
        yolo_probs: np.ndarray,
        seg_probs: np.ndarray,
        yolo_weight: float = 0.4,
        seg_weight: float = 0.6
    ) -> np.ndarray:
        """
        Fusion par moyenne pondérée des probabilités.
        
        Args:
            yolo_probs: Probabilités YOLO (H, W, 6)
            seg_probs: Probabilités SegFormer (H, W, 6)
            yolo_weight: Poids YOLO (défaut 0.4)
            seg_weight: Poids SegFormer (défaut 0.6)
            
        Returns:
            mask: Masque fusionné (H, W)
        """
        fused_probs = yolo_weight * yolo_probs + seg_weight * seg_probs
        mask = fused_probs.argmax(axis=2)
        return mask
    
    def fuse_class_specific(
        self,
        yolo_mask: np.ndarray,
        seg_mask: np.ndarray,
        yolo_probs: np.ndarray,
        seg_probs: np.ndarray
    ) -> np.ndarray:
        """
        Fusion basée sur la complémentarité:
        - SegFormer pour classes LIQUIDES (Oil, Emulsion, Sheen)
        - YOLO pour classes SOLIDES (Ship, Oil-platform)
        
        Justification (State of the Art):
        - SegFormer excelle sur liquides (IoU: Oil 71.76%, Emulsion 75.05%, Sheen 65.02%)
        - YOLO excelle sur solides (IoU: Ship 62.82%, Oil-platform 30.81%)
        
        Args:
            yolo_mask: Masque YOLO (H, W)
            seg_mask: Masque SegFormer (H, W)
            yolo_probs: Probabilités YOLO (H, W, 6)
            seg_probs: Probabilités SegFormer (H, W, 6)
            
        Returns:
            mask: Masque fusionné (H, W)
        """
        H, W = yolo_mask.shape
        fused_mask = np.zeros((H, W), dtype=np.int64)
        
        # Utiliser SegFormer pour les liquides
        for cls in self.liquid_classes:
            liquid_pixels = (seg_mask == cls)
            fused_mask[liquid_pixels] = cls
        
        # Utiliser YOLO pour les solides
        for cls in self.solid_classes:
            solid_pixels = (yolo_mask == cls)
            fused_mask[solid_pixels] = cls
        
        # Background: prendre le modèle le plus confiant
        bg_pixels = (fused_mask == 0)
        yolo_bg_conf = yolo_probs[bg_pixels, 0]
        seg_bg_conf = seg_probs[bg_pixels, 0]
        # Si aucun modèle ne prédit quelque chose, c'est du background
        
        return fused_mask
    
    def fuse_confidence_based(
        self,
        yolo_probs: np.ndarray,
        seg_probs: np.ndarray,
        threshold: float = 0.7
    ) -> np.ndarray:
        """
        Fusion basée sur la confiance:
        - Si un modèle est très confiant (> threshold), le prendre
        - Sinon, prendre la moyenne pondérée
        
        Args:
            yolo_probs: Probabilités YOLO (H, W, 6)
            seg_probs: Probabilités SegFormer (H, W, 6)
            threshold: Seuil de confiance haute
            
        Returns:
            mask: Masque fusionné (H, W)
        """
        H, W, C = yolo_probs.shape
        fused_mask = np.zeros((H, W), dtype=np.int64)
        
        # Confiance maximale de chaque modèle
        yolo_max_conf = yolo_probs.max(axis=2)
        seg_max_conf = seg_probs.max(axis=2)
        
        # YOLO très confiant
        yolo_confident = yolo_max_conf > threshold
        fused_mask[yolo_confident] = yolo_probs[yolo_confident].argmax(axis=1)
        
        # SegFormer très confiant
        seg_confident = (seg_max_conf > threshold) & ~yolo_confident
        fused_mask[seg_confident] = seg_probs[seg_confident].argmax(axis=1)
        
        # Ni l'un ni l'autre confiant: moyenne pondérée
        uncertain = ~yolo_confident & ~seg_confident
        fused_probs = 0.4 * yolo_probs[uncertain] + 0.6 * seg_probs[uncertain]
        fused_mask[uncertain] = fused_probs.argmax(axis=1)
        
        return fused_mask
    
    def fuse_voting(
        self,
        yolo_mask: np.ndarray,
        seg_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fusion par vote simple:
        - Si les deux modèles sont d'accord: prendre cette classe
        - Sinon: prendre SegFormer (car meilleur mIoU global)
        
        Args:
            yolo_mask: Masque YOLO (H, W)
            seg_mask: Masque SegFormer (H, W)
            
        Returns:
            mask: Masque fusionné (H, W)
        """
        # Accord
        agreement = (yolo_mask == seg_mask)
        fused_mask = np.where(agreement, yolo_mask, seg_mask)
        
        return fused_mask
    
    def fuse_hybrid_specialized(
    self,
    yolo_mask: np.ndarray,
    seg_mask: np.ndarray,
    yolo_probs: np.ndarray,
    seg_probs: np.ndarray
) -> np.ndarray:
        """
        Fusion Hybride Spécialisée : 
        - YOLO pour les classes SOLIDES (Ship, Platform)
        - SegFormer pour les classes LIQUIDES et BACKGROUND
        """
        # 1. On initialise le masque avec les prédictions de SegFormer
        # Cela garantit que les fluides et le background sont gérés par le meilleur modèle
        fused_mask = seg_mask.copy()
        
        # 2. On extrait la confiance de YOLO
        yolo_conf = yolo_probs.max(axis=2)
        
        # 3. Traitement spécifique des SOLIDES (Ship=4, Platform=5)
        # On utilise YOLO uniquement si sa confiance est suffisante (> 0.4)
        # pour éviter d'importer du bruit de détection
        for cls in [4, 5]:
            # On identifie les pixels où YOLO détecte un solide avec confiance
            yolo_solid_idx = (yolo_mask == cls) & (yolo_conf > 0.40)
            
            # On remplace les pixels du masque final par ceux de YOLO
            fused_mask[yolo_solid_idx] = cls
            
        # 4. Raffinement : Suppression des faux positifs de SegFormer
        # Si SegFormer voit un navire (4) mais que YOLO (le spécialiste) ne voit absolument rien
        # On considère que c'est une erreur de SegFormer et on remet en background
        for cls in [4, 5]:
            fp_risk = (seg_mask == cls) & (yolo_conf < 0.10)
            fused_mask[fp_risk] = 0
            
        return fused_mask

    def fuse_perfect_synergy(
    self,
    yolo_mask: np.ndarray,
    seg_mask: np.ndarray,
    yolo_probs: np.ndarray,
    seg_probs: np.ndarray
) -> np.ndarray:
        """
        La Fusion Parfaite : 
        - Liquides & Background : SegFormer (IOU > 70%)
        - Solides : Weighted Consensus (YOLO + SegFormer)
        - Sécurité : Filtrage des faux positifs YOLO sur le background.
        """
        # 1. Calcul du consensus pondéré (meilleure perf sur Ship/Platform)
        # On utilise vos poids optimaux (0.4 YOLO / 0.6 SegFormer)
        weighted_probs = (0.4 * yolo_probs) + (0.6 * seg_probs)
        weighted_mask = weighted_probs.argmax(axis=2)
        
        # 2. Initialisation avec SegFormer (Autorité sur le milieu marin)
        fused_mask = seg_mask.copy()
        
        # 3. Injection du consensus UNIQUEMENT pour les classes solides (4, 5)
        # On ne remplace SegFormer que si le consensus désigne un objet solide
        solid_consensus = np.isin(weighted_mask, [4, 5])
        fused_mask[solid_consensus] = weighted_mask[solid_consensus]
        
        # 4. Raffinement par la Confiance YOLO
        # YOLO est excellent pour délimiter les structures nettes.
        # Si YOLO est extrêmement sûr (> 0.8) d'un navire, il a priorité absolue.
        yolo_conf = yolo_probs.max(axis=2)
        ultra_conf_yolo = (yolo_mask == 4) & (yolo_conf > 0.8)
        fused_mask[ultra_conf_yolo] = 4

        # 5. Nettoyage des Faux Positifs (Security Gate)
        # Si SegFormer voit un navire mais que YOLO (le spécialiste) ne voit rien (< 0.05)
        # On considère que c'est du bruit de vagues détecté par SegFormer.
        fp_cleanup = (seg_mask == 4) & (yolo_conf < 0.05)
        fused_mask[fp_cleanup] = 0
        
        return fused_mask
    
    def fuse_adaptive_uncertainty(
    self,
    yolo_probs: np.ndarray,
    seg_probs: np.ndarray
) -> np.ndarray:
        """
        Fusion adaptative par incertitude :
        - Calcule l'entropie de chaque modèle (plus l'entropie est haute, moins le modèle est sûr).
        - Donne plus de poids au modèle le plus "certain" pour chaque pixel.
        - Applique un bonus de priorité aux objets solides pour YOLO.
        """
        # 1. Calcul de l'entropie (Incertitude) : -sum(p * log(p))
        yolo_uncertainty = -np.sum(yolo_probs * np.log(yolo_probs + 1e-10), axis=2)
        seg_uncertainty = -np.sum(seg_probs * np.log(seg_probs + 1e-10), axis=2)
        
        # 2. Inversion pour obtenir la "Certitude"
        yolo_certainty = 1.0 / (yolo_uncertainty + 1e-10)
        seg_certainty = 1.0 / (seg_uncertainty + 1e-10)
        
        # 3. Normalisation des poids de certitude
        total_certainty = yolo_certainty + seg_certainty
        w_yolo = yolo_certainty / total_certainty
        w_seg = seg_certainty / total_certainty
        
        # 4. Bonus de spécialisation (Expertise métier)
        # On booste YOLO sur les solides (4, 5) et SegFormer sur les liquides (1, 2, 3)
        yolo_expert_mask = np.isin(yolo_probs.argmax(axis=2), [4, 5])
        w_yolo[yolo_expert_mask] *= 1.5 # Boost YOLO sur Ship/Platform
        
        # 5. Fusion finale des probabilités
        fused_probs = (w_yolo[:,:,None] * yolo_probs) + (w_seg[:,:,None] * seg_probs)
        
        return fused_probs.argmax(axis=2)
                
    # ========================================================================
    # ÉVALUATION
    # ========================================================================
    
    def evaluate_all_strategies(self, test_loader=None, save_results: bool = True,num_samples: int = 350):
        """
        Évalue toutes les stratégies de fusion sur le test set.
        
        Args:
            test_loader: DataLoader test (si None, charge depuis get_dataloaders)
            save_results: Sauvegarder les résultats dans JSON
            
        Returns:
            results: Dict avec les résultats de chaque stratégie
        """
        if test_loader is None:
            test_loader = get_dataloaders()['test']
        
        print("\n" + "="*70)
        print("🔬 ÉVALUATION DES STRATÉGIES DE FUSION")
        print("="*70)
        
        # Métriques pour chaque stratégie
        strategies = {
            'yolo_only': SegmentationMetrics(NUM_CLASSES),
            'segformer_only': SegmentationMetrics(NUM_CLASSES),
            'weighted_average': SegmentationMetrics(NUM_CLASSES),
            'class_specific': SegmentationMetrics(NUM_CLASSES),
            'confidence_based': SegmentationMetrics(NUM_CLASSES),
            'voting': SegmentationMetrics(NUM_CLASSES),
            'hybrid_specialized': SegmentationMetrics(NUM_CLASSES),
            'perfect_synergy': SegmentationMetrics(NUM_CLASSES),
            'adaptive_uncertainty': SegmentationMetrics(NUM_CLASSES)


        }
        
        for metric in strategies.values():
            metric.reset()
        
        print(f"\n📊 Évaluation sur {len(test_loader.dataset)} images...")
        count = 0
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Fusion")):
                if count >= num_samples: break
                for i in range(images.shape[0]):
                    # Ground truth
                    gt_mask = masks[i].cpu().numpy()
                    
                    # Image pour YOLO (dénormalisée)
                    img_np = images[i].permute(1, 2, 0).cpu().numpy()
                    img_denorm = img_np * self.std + self.mean
                    img_uint8 = (img_denorm * 255).clip(0, 255).astype(np.uint8)
                    
                    # Image pour SegFormer (normalisée)
                    img_tensor = images[i]
                    
                    # === PRÉDICTIONS ===
                    yolo_mask, yolo_probs = self.predict_yolo(img_uint8)
                    seg_mask, seg_probs = self.predict_segformer(img_tensor)
                    
                    # === FUSION ===
                    fused_masks = {
                        'yolo_only': yolo_mask,
                        'segformer_only': seg_mask,
                        'weighted_average': self.fuse_weighted_average(yolo_probs, seg_probs),
                        'class_specific': self.fuse_class_specific(yolo_mask, seg_mask, yolo_probs, seg_probs),
                        'confidence_based': self.fuse_confidence_based(yolo_probs, seg_probs),
                        'voting': self.fuse_voting(yolo_mask, seg_mask),
                        'hybrid_specialized': self.fuse_hybrid_specialized(yolo_mask, seg_mask, yolo_probs, seg_probs),
                        'perfect_synergy': self.fuse_perfect_synergy(yolo_mask, seg_mask, yolo_probs, seg_probs),
                        'adaptive_uncertainty': self.fuse_adaptive_uncertainty(yolo_probs, seg_probs)
                    }
                    
                    # Mise à jour des métriques
                    for strategy_name, fused_mask in fused_masks.items():
                        strategies[strategy_name].update(fused_mask, gt_mask)
                    count += 1
        
        # === RÉSULTATS ===
        print("\n" + "="*70)
        print("📊 RÉSULTATS PAR STRATÉGIE")
        print("="*70)
        
        results = {}
        for strategy_name, metric in strategies.items():
            res = metric.get_results()
            results[strategy_name] = {
                'mIoU': float(res['mIoU']),
                'mAcc': float(res['mAcc']),
                'class_iou': [float(x) for x in res['class_iou']]
            }
            
            print(f"\n🎯 {strategy_name.upper()}")
            print(f"   mIoU: {res['mIoU']:.4f}")
            print(f"   mAcc: {res['mAcc']:.4f}")
            
            # Afficher IoU par classe
            for cls_id, iou in enumerate(res['class_iou']):
                if cls_id > 0:  # Skip background
                    print(f"     {CLASS_NAMES_SEG[cls_id]:15s}: {iou:.4f}")
        OUTPUT_DIR_FUSION = project_root / "output" / "fusion"
        # Sauvegarder
        if save_results:
            results_path = OUTPUT_DIR_FUSION / 'fusion_resultsV1.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Résultats sauvegardés: {results_path}")
        
        return results
    
    def visualize_comparison(
        self,
        image_tensor: torch.Tensor,
        gt_mask: np.ndarray,
        save_path: str = None
    ):
        """
        Visualise les prédictions de chaque stratégie sur une image.
        
        Args:
            image_tensor: Image normalisée (3, H, W)
            gt_mask: Ground truth (H, W)
            save_path: Chemin de sauvegarde (optionnel)
        """
        # Dénormaliser pour affichage
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_denorm = img_np * self.std + self.mean
        img_uint8 = (img_denorm * 255).clip(0, 255).astype(np.uint8)
        
        # Prédictions
        yolo_mask, yolo_probs = self.predict_yolo(img_uint8)
        seg_mask, seg_probs = self.predict_segformer(image_tensor)
        
        # Fusions
        weighted = self.fuse_weighted_average(yolo_probs, seg_probs)
        class_spec = self.fuse_class_specific(yolo_mask, seg_mask, yolo_probs, seg_probs)
        conf_based = self.fuse_confidence_based(yolo_probs, seg_probs)
        voting = self.fuse_voting(yolo_mask, seg_mask)
        
        # Plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        titles = [
            'Image', 'Ground Truth', 'YOLO', 'SegFormer',
            'Weighted Avg', 'Class-Specific', 'Confidence', 'Voting'
        ]
        
        masks_to_plot = [
            img_uint8, gt_mask, yolo_mask, seg_mask,
            weighted, class_spec, conf_based, voting
        ]
        
        for ax, title, mask_data in zip(axes.flat, titles, masks_to_plot):
            if title == 'Image':
                ax.imshow(mask_data)
            else:
                ax.imshow(mask_data, cmap='tab10', vmin=0, vmax=5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualisation sauvegardée: {save_path}")
        
        plt.show()


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_fusion_comparison_table(results: dict) -> str:
    """
    Crée un tableau comparatif format Markdown.
    
    Args:
        results: Résultats de evaluate_all_strategies()
        
    Returns:
        table: Tableau Markdown
    """
    lines = []
    lines.append("| Stratégie | mIoU | mAcc | Oil | Emulsion | Sheen | Ship | Oil-plat. |")
    lines.append("|-----------|------|------|-----|----------|-------|------|-----------|")
    
    for strategy, res in results.items():
        miou = res['mIoU'] * 100
        macc = res['mAcc'] * 100
        ious = [iou * 100 for iou in res['class_iou'][1:]]  # Skip background
        
        line = f"| {strategy:20s} | {miou:5.2f} | {macc:5.2f} |"
        for iou in ious:
            line += f" {iou:5.2f} |"
        lines.append(line)
    
    return "\n".join(lines)


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FUSION YOLOV11 + SEGFORMER - LADOS")
    print("="*70)
    
    # Chemins des modèles (à adapter selon ton environnement)
    YOLO_PATH = project_root / "output" / "yolo_best.pt"
    SEG_PATH = project_root / "output" / "segformer_best.pt"
    OUTPUT_DIR_FUSION = project_root / "output" / "fusion"
    
    # Vérifier existence
    if not YOLO_PATH.exists():
        print(f"❌ YOLO non trouvé: {YOLO_PATH}")
        sys.exit(1)
    if not SEG_PATH.exists():
        print(f"❌ SegFormer non trouvé: {SEG_PATH}")
        sys.exit(1)
    
    # Créer la fusion
    fusion = ModelFusion(
        yolo_model_path=str(YOLO_PATH),
        segformer_model_path=str(SEG_PATH)
    )
    
    # Évaluer toutes les stratégies
    results = fusion.evaluate_all_strategies()
    
    # Tableau comparatif
    print("\n" + "="*70)
    print("📋 TABLEAU COMPARATIF (Markdown)")
    print("="*70)
    print(create_fusion_comparison_table(results))
    
    # Visualisation d'exemple
    print("\n📊 Génération d'une visualisation d'exemple...")
    test_loader = get_dataloaders()['test']
    images, masks = next(iter(test_loader))
    
    fusion.visualize_comparison(
        images[0],
        masks[0].cpu().numpy(),
        save_path=str(OUTPUT_DIR_FUSION / 'fusion_comparison.png')
    )
    
    print("\n✅ Fusion terminée avec succès!")