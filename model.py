from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


folder = Path(__file__).resolve().parent
feature_file = folder / "octmnist_edge_histograms_36.npz"
if not feature_file.exists():
    subprocess.run([sys.executable, str(folder / "data_preparation.py")], check=True)

with np.load(feature_file) as data:
    labels = data["train_labels"]
    selected = np.isin(labels, [0, 3])
    features = data["train_features"][selected]
    labels = labels[selected]
    test_labels = data["test_labels"]
    test_selected = np.isin(test_labels, [0, 3])
    test_features = data["test_features"][test_selected]
    test_labels = test_labels[test_selected]

print("Selected classes: CNV and normal", flush=True)
print(f"Training images: {len(labels)}", flush=True)

c_values = [0.01, 0.1, 1, 10, 100, 1000]
cv_methods = [
    ("Standard 5-fold CV", KFold(n_splits=5, shuffle=True, random_state=42)),
    ("Stratified 5-fold CV", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
]
errors = {}

print("| C | Standard train error (%) | Standard validation error (%) | "
      "Stratified train error (%) | Stratified validation error (%) |", flush=True)
print("|---:|---:|---:|---:|---:|", flush=True)
for c in c_values:
    row = {}
    for method_name, cv in cv_methods:
        # Scaling is fitted separately on each fold's training portion.
        model = make_pipeline(StandardScaler(), LinearSVC(C=c))
        scores = cross_validate(
            model, features, labels, cv=cv, scoring="accuracy",
            return_train_score=True, n_jobs=5,
        )
        row[method_name] = (
            100 * (1 - scores["train_score"]).mean(),
            100 * (1 - scores["test_score"]).mean(),
        )
    errors[c] = row
    standard_train, standard_val = row["Standard 5-fold CV"]
    stratified_train, stratified_val = row["Stratified 5-fold CV"]
    print(f"| {c:g} | {standard_train:.3f} | {standard_val:.3f} | "
          f"{stratified_train:.3f} | {stratified_val:.3f} |", flush=True)

fig, ax = plt.subplots(figsize=(9, 6))
for method_name, color in (("Standard 5-fold CV", "tab:blue"),
                           ("Stratified 5-fold CV", "tab:orange")):
    train_errors = [errors[c][method_name][0] for c in c_values]
    validation_errors = [errors[c][method_name][1] for c in c_values]
    ax.plot(c_values, train_errors, marker="o", linestyle="-", color=color,
            label=f"{method_name} training error")
    ax.plot(c_values, validation_errors, marker="s", linestyle="--", color=color,
            label=f"{method_name} validation error")

ax.set_xscale("log")
ax.set_xticks(c_values)
ax.set_xticklabels([str(c) for c in c_values])
ax.set_xlabel("C")
ax.set_ylabel("Mean error (%)")
ax.set_title("LinearSVC model complexity: CNV vs normal")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plot_path = folder / "model_selection_complexity.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"Saved: {plot_path}")

best_c = min(c_values, key=lambda c: errors[c]["Stratified 5-fold CV"][1])
validation_error = errors[best_c]["Stratified 5-fold CV"][1]
model = make_pipeline(StandardScaler(), LinearSVC(C=best_c))
model.fit(features, labels)
test_error = 100 * (1 - model.score(test_features, test_labels))
print(f"Selected C: {best_c:g}")
print(f"Stratified CV validation error: {validation_error:.6f}%")
print(f"Held-out test error: {test_error:.6f}%")


"""
1.C = 1, 10, 100, and 1000 tie at 21.198% at the reported precision.
2.It weakens regularization, allowing the fitted boundary to respond more closely to the training data.
3.21.13% training error and 21.18 to 21.20% validation error stay high. That is consistent with the linear model or these edge-histogram features being too limited to separate the classes well.
4.Not observed. Validation error stays only about 0.05 to 0.07 percentage points above training error, and the gap does not grow with C.
5.Very similar.Their validation errors differ by at most 0.024 percentage.
"""
