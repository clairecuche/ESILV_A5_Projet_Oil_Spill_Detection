"""
Comparaison des performances entre YOLOv11 et SegFormer.
Génère des visualisations et des tableaux comparatifs selon le format du paper LADOS.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

from config import CLASS_NAMES, OUTPUT_DIR_SEG, OUTPUT_DIR_YOLO, NUM_CLASSES


def load_results(model_type='yolo'):
    """
    Charge les résultats d'évaluation d'un modèle.
    
    Args:
        model_type: 'yolo' ou 'segformer'
        
    Returns:
        dict: Résultats avec mIoU, mAcc, class_iou
    """
    if model_type == 'yolo':
        results_path = OUTPUT_DIR_YOLO / 'test_results.json'
    elif model_type == 'segformer':
        results_path = OUTPUT_DIR_SEG / 'test_results.json'
    else:
        raise ValueError(f"model_type inconnu: {model_type}")
    
    if not results_path.exists():
        print(f"⚠️ Fichier de résultats non trouvé: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_comparison_table():
    """
    Crée un tableau comparatif au format du paper LADOS (Table 3).
    
    Format:
    Model Name | Oil | Emulsion | Sheen | Ship | Oil-Platform | mAcc% | mIoU%
    """
    # Charger les résultats
    yolo_results = load_results('yolo')
    seg_results = load_results('segformer')
    
    if yolo_results is None or seg_results is None:
        print("⚠️ Impossible de créer le tableau: résultats manquants.")
        return None
    
    # Créer le DataFrame
    data = {
        'Model': ['YOLOv11', 'SegFormer'],
        'Oil': [
            yolo_results['class_iou'][1] * 100,
            seg_results['class_iou'][1] * 100
        ],
        'Emulsion': [
            yolo_results['class_iou'][2] * 100,
            seg_results['class_iou'][2] * 100
        ],
        'Sheen': [
            yolo_results['class_iou'][3] * 100,
            seg_results['class_iou'][3] * 100
        ],
        'Ship': [
            yolo_results['class_iou'][4] * 100,
            seg_results['class_iou'][4] * 100
        ],
        'Oil-Platform': [
            yolo_results['class_iou'][5] * 100,
            seg_results['class_iou'][5] * 100
        ],
        'mAcc%': [
            yolo_results['mAcc'] * 100,
            seg_results['mAcc'] * 100
        ],
        'mIoU%': [
            yolo_results['mIoU'] * 100,
            seg_results['mIoU'] * 100
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Afficher le tableau
    print("\n" + "="*100)
    print("📊 TABLEAU COMPARATIF DES PERFORMANCES (Test Set)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Sauvegarder
    output_path = Path('./outputs/comparison_table.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Tableau sauvegardé: {output_path}")
    
    return df


def plot_class_performance_comparison():
    """
    Crée un graphique en barres comparant les performances par classe.
    """
    # Charger les résultats
    yolo_results = load_results('yolo')
    seg_results = load_results('segformer')
    
    if yolo_results is None or seg_results is None:
        print("⚠️ Impossible de créer le graphique: résultats manquants.")
        return
    
    # Préparer les données
    classes = [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]  # Exclure background
    yolo_ious = [yolo_results['class_iou'][i] * 100 for i in range(1, NUM_CLASSES)]
    seg_ious = [seg_results['class_iou'][i] * 100 for i in range(1, NUM_CLASSES)]
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, yolo_ious, width, label='YOLOv11', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, seg_ious, width, label='SegFormer', 
                   color='#4ECDC4', alpha=0.8)
    
    # Personnalisation
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Performances par Classe (Test Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ajouter les valeurs sur les barres
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path('./outputs/class_performance_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {output_path}")
    plt.close()


def plot_overall_metrics_comparison():
    """
    Crée un graphique radar comparant mIoU et mAcc.
    """
    # Charger les résultats
    yolo_results = load_results('yolo')
    seg_results = load_results('segformer')
    
    if yolo_results is None or seg_results is None:
        print("⚠️ Impossible de créer le graphique: résultats manquants.")
        return
    
    # Préparer les données
    categories = ['mIoU', 'mAcc']
    yolo_values = [yolo_results['mIoU'] * 100, yolo_results['mAcc'] * 100]
    seg_values = [seg_results['mIoU'] * 100, seg_results['mAcc'] * 100]
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv11',
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, seg_values, width, label='SegFormer',
                   color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Métriques Globales (Test Set)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # Ajouter les valeurs
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path('./outputs/overall_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {output_path}")
    plt.close()


def analyze_complementarity():
    """
    Analyse la complémentarité entre YOLOv11 et SegFormer.
    Identifie sur quelles classes chaque modèle excelle.
    """
    print("\n" + "="*70)
    print("🔍 ANALYSE DE COMPLÉMENTARITÉ")
    print("="*70)
    
    yolo_results = load_results('yolo')
    seg_results = load_results('segformer')
    
    if yolo_results is None or seg_results is None:
        print("⚠️ Impossible d'analyser: résultats manquants.")
        return
    
    print("\n📌 Classes où YOLOv11 performe mieux:")
    for i in range(1, NUM_CLASSES):
        class_name = CLASS_NAMES[i]
        yolo_iou = yolo_results['class_iou'][i] * 100
        seg_iou = seg_results['class_iou'][i] * 100
        diff = yolo_iou - seg_iou
        
        if diff > 0:
            print(f"  • {class_name:15s}: YOLO {yolo_iou:.2f}% vs SegFormer {seg_iou:.2f}% (+{diff:.2f}%)")
    
    print("\n📌 Classes où SegFormer performe mieux:")
    for i in range(1, NUM_CLASSES):
        class_name = CLASS_NAMES[i]
        yolo_iou = yolo_results['class_iou'][i] * 100
        seg_iou = seg_results['class_iou'][i] * 100
        diff = seg_iou - yolo_iou
        
        if diff > 0:
            print(f"  • {class_name:15s}: SegFormer {seg_iou:.2f}% vs YOLO {yolo_iou:.2f}% (+{diff:.2f}%)")
    
    # Identifier les catégories de classes
    liquid_classes = [1, 2, 3]  # Oil, Emulsion, Sheen
    solid_classes = [4, 5]       # Ship, Oil-platform
    
    yolo_liquid_avg = np.mean([yolo_results['class_iou'][i] for i in liquid_classes]) * 100
    seg_liquid_avg = np.mean([seg_results['class_iou'][i] for i in liquid_classes]) * 100
    
    yolo_solid_avg = np.mean([yolo_results['class_iou'][i] for i in solid_classes]) * 100
    seg_solid_avg = np.mean([seg_results['class_iou'][i] for i in solid_classes]) * 100
    
    print(f"\n📊 Performance moyenne sur classes LIQUIDES:")
    print(f"  • YOLOv11: {yolo_liquid_avg:.2f}%")
    print(f"  • SegFormer: {seg_liquid_avg:.2f}%")
    print(f"  → {'SegFormer' if seg_liquid_avg > yolo_liquid_avg else 'YOLOv11'} excelle sur les liquides")
    
    print(f"\n📊 Performance moyenne sur classes SOLIDES:")
    print(f"  • YOLOv11: {yolo_solid_avg:.2f}%")
    print(f"  • SegFormer: {seg_solid_avg:.2f}%")
    print(f"  → {'YOLOv11' if yolo_solid_avg > seg_solid_avg else 'SegFormer'} excelle sur les solides")
    
    print("\n💡 CONCLUSION pour la fusion:")
    if seg_liquid_avg > yolo_liquid_avg and yolo_solid_avg > seg_solid_avg:
        print("  ✅ Complémentarité claire détectée!")
        print("  → SegFormer meilleur sur liquides (Oil, Emulsion, Sheen)")
        print("  → YOLOv11 meilleur sur solides (Ship, Oil-platform)")
        print("  → Une stratégie de fusion pourrait combiner ces forces!")
    else:
        print("  ⚠️ Complémentarité moins évidente, analyse approfondie requise.")


def main():
    """
    Génère toutes les comparaisons et analyses.
    """
    print("\n" + "="*70)
    print("📊 GÉNÉRATION DES COMPARAISONS YOLOV11 vs SEGFORMER")
    print("="*70)
    
    # 1. Tableau comparatif
    create_comparison_table()
    
    # 2. Graphiques
    print("\n📈 Génération des graphiques...")
    plot_class_performance_comparison()
    plot_overall_metrics_comparison()
    
    # 3. Analyse de complémentarité
    analyze_complementarity()
    
    print("\n" + "="*70)
    print("✅ COMPARAISONS GÉNÉRÉES AVEC SUCCÈS!")
    print("="*70)
    print("\n📁 Fichiers générés dans: ./outputs/")


if __name__ == '__main__':
    main()