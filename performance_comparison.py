from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


folder = Path(__file__).resolve().parent
feature_file = folder / "octmnist_edge_histograms_36.npz"
if not feature_file.exists():
    subprocess.run([sys.executable, str(folder / "data_preparation.py")], check=True)

with np.load(feature_file) as data:
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
np.random.seed(42)
classifiers = [
    ("GaussianNB", GaussianNB(), "gaussian_nb"),
    ("DecisionTreeClassifier(max_depth=10)", DecisionTreeClassifier(max_depth=10), "decision_tree"),
    ("RandomForestClassifier", RandomForestClassifier(), "random_forest"),
]

print("| Classifier | Mean Validation Accuracy | Standard Deviation |", flush=True)
print("|---|---:|---:|", flush=True)
for name, classifier, _ in classifiers:
    # Refit scaling inside each fold, using only that fold's training portion.
    model = make_pipeline(StandardScaler(), classifier)
    scores = cross_val_score(model, train_features, train_labels, cv=cv, scoring="accuracy")
    print(f"| {name} | {scores.mean():.4f} | {scores.std():.4f} |", flush=True)

 #question 2: Random Forest. It had the highest mean validation accuracy and the lowest standard deviation   

class_names = [
    "CNV\n(choroidal neovascularization)",
    "DME\n(diabetic macular edema)",
    "Drusen",
    "Normal",
]

print("\n| Classifier | Test Accuracy | Macro F1-score |", flush=True)
print("|---|---:|---:|", flush=True)
for name, classifier, filename in classifiers:
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    print(f"| {name} | {accuracy:.4f} | {macro_f1:.4f} |", flush=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, predictions,
        labels=[0, 1, 2, 3], display_labels=class_names,
        cmap="Blues", values_format="d", colorbar=False,
        xticks_rotation=25, ax=ax,
    )
    ax.set_title(f"{name} — held-out test set")
    fig.tight_layout()
    fig.savefig(folder / f"{filename}_confusion_matrix.png", dpi=180)
    plt.close(fig)

#question d: CNV and normal.
#CNV can distort layers locally, but at 28×28 resolution, the difference is less distinct. 
#The 36-bin histogram counts edge directions while discarding where the edges occur, so it can give similar vectors.

#question e:  Left: true class CNV; predicted normal.  Right: true class normal; predicted CNV.
#Yes. The CNV image on the left has smooth, horizontal bright bands, so it resembles a normal scan. 
#The normal image on the right has a sloping, uneven bright band that could resemble the distortion seen in some CNV scans.
#At 28×28 pixels, those details are hard to distinguish.