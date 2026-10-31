import numpy as np
from collections import Counter
from pathlib import Path
from skimage import filters, exposure, util


def stratified_split(features, labels, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for class_id in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        n_test = round(len(indices) * test_fraction)
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)
    return (features[train_indices], features[test_indices],
            labels[train_indices], labels[test_indices])


folder = Path(__file__).resolve().parent
data = np.load(folder.parent / "octmnist.npz")

features = []
labels = []

for split in ["train", "val", "test"]:
    for image, label in zip(data[f"{split}_images"], data[f"{split}_labels"].flatten()):
        gray = util.img_as_float(image)
        angles = np.mod(
            np.arctan2(filters.sobel_v(gray), filters.sobel_h(gray)),
            np.pi
        )
        hist, _ = exposure.histogram(angles, nbins=36)

        features.append(hist)
        labels.append(label)

features = np.array(features)
labels = np.array(labels)
train_features, test_features, train_labels, test_labels = stratified_split(features, labels)
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_std[train_std == 0] = 1
train_features = (train_features - train_mean) / train_std
test_features = (test_features - train_mean) / train_std

np.savez_compressed(folder / "octmnist_edge_histograms_36.npz",
                    train_features=train_features, train_labels=train_labels,
                    test_features=test_features, test_labels=test_labels,
                    train_mean=train_mean, train_std=train_std)

print("Number of images:", len(labels))
print("Number of classes:", len(np.unique(labels)))
print("Images per class:", dict(sorted(Counter(map(int, labels)).items())))
print("Feature dimensionality:", features.shape[1])
print("Training images:", len(train_labels))
print("Test images:", len(test_labels))
print("Training images per class:", dict(sorted(Counter(map(int, train_labels)).items())))
print("Test images per class:", dict(sorted(Counter(map(int, test_labels)).items())))



#question e:
#Because standardize test with training data, will make the test result looks better than on unseen images.
