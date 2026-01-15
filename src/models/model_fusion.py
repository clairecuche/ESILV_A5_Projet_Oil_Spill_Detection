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
import os
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
    
    
    def fuse_adaptive(self, yolo_probs, seg_probs):
        """
        Fusion adaptative avec poids optimisés par classe:
        - Liquides (Oil, Emulsion, Sheen) : SegFormer dominant (stabilité des textures)
        - Ship : YOLO ultra-dominant (contours nets)
        - Background & Oil-platform : Équilibre (0.4/0.6)
        """
        H, W, C = yolo_probs.shape
        fused_probs = np.zeros((H, W, C))
        
        # Poids optimisés (w_yolo, w_seg) par classe
        class_weights = {
            0: (0.40, 0.60),  # Background
            1: (0.15, 0.85),  # Oil
            2: (0.05, 0.95),  # Emulsion
            3: (0.10, 0.90),  # Sheen
            4: (0.85, 0.15),  # Ship
            5: (0.40, 0.60),  # Oil-platform
        }
        
        for cls in range(C):
            w_yolo, w_seg = class_weights.get(cls, (0.4, 0.6))
            fused_probs[:, :, cls] = (w_yolo * yolo_probs[:, :, cls]) + (w_seg * seg_probs[:, :, cls])
        
        return fused_probs.argmax(axis=2)




                
    # ========================================================================
    # ÉVALUATION
    # ========================================================================
    
    def evaluate_all_strategies(self, test_loader=None, save_results: bool = True, num_samples: int = 350):
        """
        Évalue les stratégies de fusion sur le test set.
        
        Args:
            test_loader: DataLoader test (si None, charge depuis get_dataloaders)
            save_results: Sauvegarder les résultats dans JSON
            num_samples: Nombre d'images à évaluer
            
        Returns:
            results: Dict avec les résultats de chaque stratégie
        """
        if test_loader is None:
            test_loader = get_dataloaders()['test']
        
        print("\n" + "▓" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 25 + "🔬 ÉVALUATION DES MODÈLES DE FUSION" + " " * 18 + "█")
        print("█" + " " * 78 + "█")
        print("▓" * 80)
        
        # Métriques pour chaque stratégie
        strategies = {
            'yolo_only': SegmentationMetrics(NUM_CLASSES),
            'segformer_only': SegmentationMetrics(NUM_CLASSES),
            'fusion_adaptive': SegmentationMetrics(NUM_CLASSES),
        }
        
        for metric in strategies.values():
            metric.reset()
        
        print(f"\n   📊 Évaluation sur {num_samples} images du test set...\n")
        count = 0
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="   ⏳ Traitement", ncols=70)):
                if count >= num_samples: 
                    break
                for i in range(images.shape[0]):
                    if count >= num_samples:
                        break
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
                        'fusion_adaptive': self.fuse_adaptive(yolo_probs, seg_probs),
                    }
                    
                    # Mise à jour des métriques
                    for strategy_name, fused_mask in fused_masks.items():
                        strategies[strategy_name].update(fused_mask, gt_mask)
                    count += 1
        
        # === RÉSULTATS ===
        print("\n" + "▓" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 30 + "📊 RÉSULTATS FINAUX" + " " * 29 + "█")
        print("█" + " " * 78 + "█")
        print("▓" * 80)
        
        results = {}
        strategy_order = ['yolo_only', 'segformer_only', 'fusion_adaptive']
        
        for strategy_name in strategy_order:
            metric = strategies[strategy_name]
            res = metric.get_results()
            results[strategy_name] = {
                'mIoU': float(res['mIoU']),
                'mAcc': float(res['mAcc']),
                'class_iou': [float(x) for x in res['class_iou']],
                'timestamp': datetime.now().isoformat()
            }
            
            # Affichage joli
            strategy_display = {
                'yolo_only': '🟡 YOLO Seul',
                'segformer_only': '🟢 SegFormer Seul',
                'fusion_adaptive': '🔵 Fusion Adaptative',
            }
            
            print(f"\n   {strategy_display.get(strategy_name, strategy_name)}")
            print(f"   {'─' * 60}")
            print(f"   │  📈 mIoU (Intersection over Union):  {res['mIoU']*100:6.2f}%")
            print(f"   │  🎯 mAcc (Moyenne Accuracy):        {res['mAcc']*100:6.2f}%")
            print(f"   │")
            print(f"   │  IoU par classe:")
            
            class_names = ['Background', 'Oil', 'Emulsion', 'Sheen', 'Ship', 'Oil-Platform']
            for cls_id, iou in enumerate(res['class_iou']):
                icon = '⬛' if cls_id == 0 else '▪️'
                print(f"   │    {icon} {class_names[cls_id]:15s} : {iou*100:6.2f}%")
            
            print(f"   └─────────────────────────────────────────────────────────────")
        
        OUTPUT_DIR_FUSION = project_root / "output" / "fusion"
        
        # Sauvegarder
        if save_results:
            results_path = OUTPUT_DIR_FUSION / 'fusion_results.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n   ✅ Résultats sauvegardés: {results_path}")
        
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
        
        # Fusion
        adaptive = self.fuse_adaptive(yolo_probs, seg_probs)
        
        # Plot
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        titles = [
            'Image Originale', 'Ground Truth', 'YOLO Seul', 'SegFormer Seul', 'Fusion Adaptative'
        ]
        
        masks_to_plot = [
            img_uint8, gt_mask, yolo_mask, seg_mask, adaptive
        ]
        
        cmap = plt.cm.get_cmap('tab20')
        
        for ax, title, mask_data in zip(axes.flat, titles, masks_to_plot):
            if title == 'Image Originale':
                ax.imshow(mask_data)
            else:
                im = ax.imshow(mask_data, cmap='tab10', vmin=0, vmax=5)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n   ✅ Visualisation sauvegardée: {save_path}")
        
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
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🚀 FUSION YOLOV11 + SEGFORMER - OIL SPILL DETECTION" + " " * 6 + "║")
    print("╚" + "═" * 78 + "╝")

    YOLO_PATH = project_root / "output" / "yolo_best.pt"
    SEG_PATH = project_root / "output" / "segformer_best.pt"
    OUTPUT_DIR_FUSION = project_root / "output" / "fusion"
        
    
    # Vérifier existence
    if not YOLO_PATH.exists():
        print(f"\n   ❌ YOLO non trouvé: {YOLO_PATH}")
        sys.exit(1)
    if not SEG_PATH.exists():
        print(f"\n   ❌ SegFormer non trouvé: {SEG_PATH}")
        sys.exit(1)
    
    print(f"\n   ✅ Modèles trouvés:")
    print(f"      • YOLO:      {YOLO_PATH}")
    print(f"      • SegFormer: {SEG_PATH}")
    
    # Créer la fusion
    print(f"\n   ⏳ Initialisation des modèles...")
    fusion = ModelFusion(
        yolo_model_path=str(YOLO_PATH),
        segformer_model_path=str(SEG_PATH)
    )
    
    # Évaluer toutes les stratégies
    print("\n")
    results = fusion.evaluate_all_strategies(num_samples=350)
    
    # Tableau comparatif
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 32 + "📋 TABLEAU RÉCAPITULATIF" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print(create_fusion_comparison_table(results))
    
    # Visualisation d'exemple
    print("\n   📊 Test sur une image du dataset...")
    test_loader = get_dataloaders()['test']
    images, masks = next(iter(test_loader))
    
    fusion.visualize_comparison(
        images[0],
        masks[0].cpu().numpy(),
        save_path=str(OUTPUT_DIR_FUSION / 'fusion_comparison_example.png')
    )
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "✅ FUSION TERMINÉE AVEC SUCCÈS!" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝\n")