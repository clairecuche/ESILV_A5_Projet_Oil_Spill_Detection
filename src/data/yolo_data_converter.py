"""
Convertisseur de format COCO vers YOLO pour la segmentation d'instance.
Convertit les masques en polygones au format YOLO.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils
import yaml


def convert_coco_to_yolo_segmentation(dataset_path, output_path=None):
    """
    Convertit les annotations COCO en format YOLO pour la segmentation.
    
    Format YOLO pour segmentation:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    où les coordonnées sont normalisées entre 0 et 1.
    
    Args:
        dataset_path: Path vers le dataset LADOS (contient train/, valid/, test/)
        output_path: Path de sortie (par défaut: dataset_path/yolo_format/)
    """
    dataset_path = Path(dataset_path)
    if output_path is None:
        output_path = dataset_path / 'yolo_format'
    else:
        output_path = Path(output_path)
    
    # Créer la structure YOLO
    output_path.mkdir(exist_ok=True, parents=True)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n📦 Conversion du split: {split}")
        
        # Paths
        images_dir = output_path / 'images' / split
        labels_dir = output_path / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les annotations COCO
        coco_json = dataset_path / split / '_annotations.coco.json'
        if not coco_json.exists():
            print(f"⚠️ Pas d'annotations COCO pour {split}, skip.")
            continue
            
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        # Créer un mapping image_id -> annotations
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Traiter chaque image
        for img_info in tqdm(coco_data['images'], desc=f"Converting {split}"):
            img_id = img_info['id']
            img_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copier l'image
            src_img = dataset_path / split / img_name
            dst_img = images_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Créer le fichier label correspondant
            label_file = labels_dir / (Path(img_name).stem + '.txt')
            
            # Extraire les annotations pour cette image
            anns = img_to_anns.get(img_id, [])
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    # Classe (YOLO commence à 0)
                    class_id = ann['category_id']
                    
                    # Extraire le polygone de segmentation
                    if 'segmentation' not in ann or not ann['segmentation']:
                        continue
                    
                    # COCO peut avoir plusieurs polygones par annotation
                    for segmentation in ann['segmentation']:
                        if len(segmentation) < 6:  # Au moins 3 points
                            continue
                        
                        # Normaliser les coordonnées
                        normalized_coords = []
                        for i in range(0, len(segmentation), 2):
                            x = segmentation[i] / img_width
                            y = segmentation[i + 1] / img_height
                            # Clipper entre 0 et 1
                            x = max(0, min(1, x))
                            y = max(0, min(1, y))
                            normalized_coords.extend([x, y])
                        
                        # Écrire au format YOLO
                        coords_str = ' '.join([f'{c:.6f}' for c in normalized_coords])
                        f.write(f"{class_id} {coords_str}\n")
    
    # Créer le fichier data.yaml
    create_yolo_yaml(output_path, dataset_path)
    
    print(f"\n✅ Conversion terminée! Données YOLO dans: {output_path}")
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


if __name__ == '__main__':
    # Exemple d'utilisation
    from config import DATA_DIR
    
    print("🔄 Conversion COCO → YOLO pour segmentation d'instance...")
    yolo_path = convert_coco_to_yolo_segmentation(DATA_DIR)
    
    print("\n🔍 Vérification de la conversion...")
    verify_conversion(yolo_path, num_samples=3)
    
    print("\n✅ Conversion terminée avec succès!")