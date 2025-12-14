# src/data/preprocessing.py

import json
import cv2
import numpy as np
import torch
from pathlib import Path
import os
from typing import Dict, Tuple
from config import CLASS_TO_ID, NUM_CLASSES, TARGET_SIZE

def convert_coco_to_masks(dataset_path: str, target_size: Tuple[int, int] = TARGET_SIZE) -> None:
    """
    Convertit les annotations COCO (JSON) en masques de segmentation (PNG), 
    avec mise à l'échelle des coordonnées.
    """
    dataset_path_p = Path(dataset_path)

    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path_p / split
        annotations_file = split_dir / '_annotations.coco.json'

        if not annotations_file.exists():
            continue

        print(f"\n🔄 Traitement des annotations COCO pour {split}...")

        masks_dir = split_dir / 'masks'
        masks_dir.mkdir(exist_ok=True)

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        id_to_image_info = {
            img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} 
            for img in coco_data['images']
        }

        image_annotations = {}
        for ann in coco_data['annotations']:
            image_annotations.setdefault(ann['image_id'], []).append(ann)

        processed = 0
        for image_id, anns in image_annotations.items():
            image_info = id_to_image_info.get(image_id)
            if not image_info: continue

            filename = image_info['file_name']
            img_path = split_dir / filename
            if not img_path.exists(): continue
            
            # Dimensions originales et facteurs d'échelle
            original_width = image_info['width']
            original_height = image_info['height']
            target_width, target_height = target_size
            scale_x = target_width / original_width
            scale_y = target_height / original_height

            # Créer masque vide (basé sur la taille cible)
            mask = np.zeros(target_size[::-1], dtype=np.uint8) # (H, W)

            for ann in anns:
                category_name = categories[ann['category_id']].lower().replace(' ', '-')
                class_id = CLASS_TO_ID.get(category_name, 0)

                if 'segmentation' in ann and ann['segmentation']:
                    seg = ann['segmentation']
                    if isinstance(seg, list) and len(seg) > 0:
                        for polygon in seg:
                            if len(polygon) >= 6: 
                                pts = np.array(polygon).reshape(-1, 2).astype(np.float32)

                                # Mise à l'échelle des coordonnées
                                pts[:, 0] = pts[:, 0] * scale_x
                                pts[:, 1] = pts[:, 1] * scale_y
                                pts = pts.astype(np.int32)
                                
                                # Dessiner le polygone
                                cv2.fillPoly(mask, [pts], class_id)
                                
            mask_filename = Path(filename).stem + '.png'
            cv2.imwrite(str(masks_dir / mask_filename), mask)
            processed += 1

        print(f"  ✓ {processed} masques créés dans {masks_dir}")
    print("TERMINÉ : Conversion des annotations en masques.")


def calculate_class_weights(dataset_path: str, split: str = 'train', num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Calcule les poids des classes en utilisant la méthode de la Fréquence Inverse Médiane.
    """
    masks_dir = Path(dataset_path) / split / 'masks'
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    mask_files = list(masks_dir.glob('*.png'))

    if not mask_files:
        print(f"ERREUR: Aucun masque trouvé dans {masks_dir}.")
        return torch.ones(num_classes).float() # Retourne des poids unitaires par défaut

    print(f"\n🔄 Démarrage du calcul des poids sur {len(mask_files)} masques...")

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        mask = np.clip(mask, 0, num_classes - 1) 
        unique, counts = np.unique(mask, return_counts=True)
        counts_map = dict(zip(unique, counts))
        
        for class_id in range(num_classes):
            count = counts_map.get(class_id, 0)
            pixel_counts[class_id] += count
            total_pixels += count
            
    frequencies = pixel_counts / total_pixels
    frequencies[frequencies == 0] = 1e-6 # Évite la division par zéro
    
    # Calcul des poids (Median Frequency Balancing)
    median_frequency = np.median(frequencies)
    class_weights = median_frequency / frequencies
    
    print("--- Résumé des Poids de Classe ---")
    for i, (name, id) in enumerate(CLASS_TO_ID.items()):
        print(f"Classe {id} ({name}): Fréquence = {frequencies[i]:.4f}, Poids = {class_weights[i]:.2f}")

    return torch.from_numpy(class_weights).float()