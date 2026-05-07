import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def load_images_from_directory(data_path, img_size=(64, 64)):
    images_dir = os.path.join(data_path, "images")
    classes = ["spiral", "elliptical", "uncertain"]
    X = []
    y = []

    print("Loading images...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(images_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        print(f"Loading {len(image_files)} images from {class_name}...")

        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).convert("L")
                img = img.resize(img_size)
                img_array = np.array(img).flatten().astype(np.float32) / 255.0
                X.append(img_array)
                y.append(class_idx)
            except Exception as exc:
                print(f"Error loading {img_path}: {exc}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nDataset shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    return X, y, classes


def train_random_forest(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete!")
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {acc:.4f}")

    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    print("\nPrecision (per class):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {precision[idx]:.4f}")
    print(f"  Macro avg: {precision_score(y_test, y_pred, average='macro'):.4f}")

    print("\nRecall (per class):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {recall[idx]:.4f}")
    print(f"  Macro avg: {recall_score(y_test, y_pred, average='macro'):.4f}")

    print("\nF1-Score (per class):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {f1[idx]:.4f}")
    print(f"  Macro avg: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\n" + "-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def plot_results(results, class_names):
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn is required for plotting. Install it with `pip install seaborn`.")
        return

    cm = results["confusion_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title("Confusion Matrix (Test Set)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    metrics = {
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1-Score": results["f1"],
    }
    x = np.arange(len(class_names))
    width = 0.25

    for i, (name, values) in enumerate(metrics.items()):
        axes[1].bar(x + i * width, values, width, label=name)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Per-Class Performance")
    axes[1].set_xlabel("Galaxy Class")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("random_forest_results.png", dpi=300, bbox_inches="tight")
    print("Results saved to random_forest_results.png")
    plt.show()


if __name__ == "__main__":
    data_path = r"galaxy_zoo_balanced_15000"

    if not os.path.exists(data_path):
        print(f"Error: dataset folder not found at {data_path}")
        exit(1)

    X, y, class_names = load_images_from_directory(data_path, img_size=(64, 64))
    model, X_train, X_test, y_train, y_test = train_random_forest(
        X, y, test_size=0.2, random_state=42
    )
    results = evaluate_model(model, X_test, y_test, class_names)
    plot_results(results, class_names)
    print("\n✓ Random Forest training and evaluation complete!")
