
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns #for vizualisation

def load_images_from_directory(data_path, img_size=(64, 64)):
    images_dir = os.path.join(data_path, "images")
    classes = ["spiral", "elliptical", "uncertain"]
    
    X = []
    y = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(images_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                # Load and convert to grayscale
                img = Image.open(img_path).convert('L')
                # Resize to standard size
                img = img.resize(img_size)
                # Convert to numpy array and flatten
                img_array = np.array(img).flatten()
                # Normalize pixel values to [0, 1]
                img_array = img_array / 255.0
                
                X.append(img_array)
                y.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y, classes


def train_decision_tree(X, y, test_size=0.2, random_state=42):
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y  #afin de garder un équilibre entre les classes
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    model = DecisionTreeClassifier(
        max_depth=20,  
        min_samples_split=5, 
        min_samples_leaf=2, 
        random_state=random_state
    )
    
    print("\nTraining Decision Tree...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, class_names):
 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")
    
    print(f"\nPrecision (per class):")
    precision = precision_score(y_test, y_test_pred, average=None)
    for i, name in enumerate(class_names):
        print(f"  {name}: {precision[i]:.4f}")
    print(f"  Macro avg: {precision_score(y_test, y_test_pred, average='macro'):.4f}")
    
    print(f"\nRecall (per class):")
    recall = recall_score(y_test, y_test_pred, average=None)
    for i, name in enumerate(class_names):
        print(f"  {name}: {recall[i]:.4f}")
    print(f"  Macro avg: {recall_score(y_test, y_test_pred, average='macro'):.4f}")
    
    print(f"\nF1-Score (per class):")
    f1 = f1_score(y_test, y_test_pred, average=None)
    for i, name in enumerate(class_names):
        print(f"  {name}: {f1[i]:.4f}")
    print(f"  Macro avg: {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    
    print("\n" + "-"*60)
    print("Detailed Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_test_pred': y_test_pred
    }


def plot_results(results, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Performance metrics
    metrics_data = {
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1']
    }
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        axes[1].bar(x + i*width, values, width, label=metric_name, alpha=0.8)
    
    axes[1].set_xlabel('Galaxy Class', fontweight='bold')
    axes[1].set_ylabel('Score', fontweight='bold')
    axes[1].set_title('Performance Metrics by Class', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(class_names)
    axes[1].legend()
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_tree_resultats.png', dpi=300, bbox_inches='tight')
    print("\nResultats : decision_tree_resultats.png")
    plt.show()


if __name__ == "__main__":
    DATA_PATH = r"galaxy_zoo_balanced_15000"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please ensure the galaxy_zoo_balanced_15000 folder exists in the current directory")
        exit(1)
    
    X, y, class_names = load_images_from_directory(DATA_PATH, img_size=(64, 64))
    
    model, X_train, X_test, y_train, y_test = train_decision_tree(
        X, y, 
        test_size=0.2,
        random_state=42
    )

    results = evaluate_model(model, X_train, X_test, y_train, y_test, class_names)
    
    plot_results(results, class_names)
    
