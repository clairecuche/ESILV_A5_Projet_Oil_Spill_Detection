# src/data/preprocessing.py

import json
import cv2
import numpy as np
import torch
from pathlib import Path
import os
from typing import Dict, Tuple
from config import CLASS_TO_ID, NUM_CLASSES, TARGET_SIZE, DATA_DIR
import shutil
from tqdm import tqdm
from pycocotools import mask as mask_utils
import yaml

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


def calculate_class_weights(dataset_path: str = DATA_DIR, split: str = 'train', num_classes: int = NUM_CLASSES) -> torch.Tensor:
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

def convert_coco_to_yolo_segmentation(dataset_path, output_path=None):
    """
    NOUVELLE MÉTHODE (Corrigée) : Mappe les noms de classes COCO vers CLASS_TO_ID
    et COPIE les images vers le dossier de destination.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path else dataset_path / 'yolo_format'
    output_path.mkdir(exist_ok=True, parents=True)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n📦 Conversion du split: {split}")
        images_dir = output_path / 'images' / split
        labels_dir = output_path / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        coco_json = dataset_path / split / '_annotations.coco.json'
        if not coco_json.exists(): 
            print(f"⚠️ Pas d'annotations pour {split}")
            continue
            
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        # 1. CRÉER LE PONT : COCO ID -> NOM -> TON ID (config.py)
        coco_id_to_project_id = {}
        for cat in coco_data['categories']:
            raw_name = cat['name'].lower().replace(' ', '-')
            # Mapping spécifique pour Roboflow
            if raw_name == 'oils-emulsions': raw_name = 'oil'
            
            project_id = CLASS_TO_ID.get(raw_name, 0)
            coco_id_to_project_id[cat['id']] = project_id
            print(f"  Mapping COCO {cat['id']} ({cat['name']}) -> Project ID {project_id}")

        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        for img_info in tqdm(coco_data['images'], desc=f"Converting {split}"):
            img_name = img_info['file_name']
            
            # --- AJOUT : COPIE PHYSIQUE DE L'IMAGE ---
            src_img = dataset_path / split / img_name
            dst_img = images_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img) # Copie l'image vers yolo_format
            
            label_file = labels_dir / (Path(img_name).stem + '.txt')
            anns = img_to_anns.get(img_info['id'], [])
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    # Utilisation du mapping corrigé
                    class_id = coco_id_to_project_id.get(ann['category_id'], 0)
                    
                    if 'segmentation' not in ann or not ann['segmentation']: continue
                    
                    for segmentation in ann['segmentation']:
                        if len(segmentation) < 6: continue
                        normalized_coords = []
                        for i in range(0, len(segmentation), 2):
                            # Normalisation relative aux dimensions de l'image
                            x = max(0, min(1, segmentation[i] / img_info['width']))
                            y = max(0, min(1, segmentation[i + 1] / img_info['height']))
                            normalized_coords.extend([x, y])
                        
                        coords_str = ' '.join([f'{c:.6f}' for c in normalized_coords])
                        f.write(f"{class_id} {coords_str}\n")
    
    # Génération du YAML pour YOLO
    create_yolo_yaml(output_path, dataset_path)
    return output_path

def create_yolo_yaml(yolo_path, original_dataset_path):
    """
    Crée le fichier data.yaml pour YOLO.
    
    Args:
        yolo_path: Path du dataset YOLO
        original_dataset_path: Path du dataset original (pour les class names)
    """
    from config import CLASS_NAMES, NUM_CLASSES
    
    # Créer la configuration YAML
    data_config = {
        'path': str(yolo_path.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': NUM_CLASSES,
        'names': [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    }
    
    yaml_path = yolo_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"📄 Fichier data.yaml créé: {yaml_path}")
    return yaml_path


def verify_conversion(yolo_path, num_samples=5):
    """
    Vérifie la conversion en visualisant quelques échantillons.
    
    Args:
        yolo_path: Path du dataset YOLO
        num_samples: Nombre d'échantillons à vérifier
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon
    
    yolo_path = Path(yolo_path)
    train_images = list((yolo_path / 'images' / 'train').glob('*.jpg'))
    
    if not train_images:
        print("⚠️ Aucune image trouvée pour la vérification.")
        return
    
    # Charger les noms de classes
    yaml_path = yolo_path / 'data.yaml'
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    
    # Sélectionner des échantillons aléatoires
    samples = np.random.choice(train_images, min(num_samples, len(train_images)), replace=False)
    
    fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 5))
    if len(samples) == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, samples):
        # Charger l'image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Charger les labels
        label_path = yolo_path / 'labels' / 'train' / (img_path.stem + '.txt')
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_path.name)
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Dénormaliser
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x = coords[i] * w
                        y = coords[i+1] * h
                        polygon.append([x, y])
                    
                    # Dessiner le polygone
                    poly = MPLPolygon(polygon, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(poly)
                    
                    # Ajouter le label de classe
                    if polygon:
                        cx = np.mean([p[0] for p in polygon])
                        cy = np.mean([p[1] for p in polygon])
                        ax.text(cx, cy, class_names[class_id], 
                               color='white', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(yolo_path / 'conversion_verification.png', dpi=150, bbox_inches='tight')
    print(f"✅ Vérification sauvegardée: {yolo_path / 'conversion_verification.png'}")
    plt.close()
