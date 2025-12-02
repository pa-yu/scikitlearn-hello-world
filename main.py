from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def build_training_data():
    """
    Tiny toy dataset:
    - Features: [weight_in_grams, texture]
      texture: 0 = smooth, 1 = bumpy
    - Labels: 0 = apple, 1 = orange
    """
    X = np.array([
        [150, 0],  # smooth, lighter  -> apple
        [140, 0],  # smooth, lighter  -> apple
        [130, 0],  # smooth           -> apple
        [170, 1],  # bumpy, heavier   -> orange
        [180, 1],  # bumpy, heavier   -> orange
        [160, 1],  # bumpy            -> orange
    ])

    y = np.array([
        0,  # apple
        0,  # apple
        0,  # apple
        1,  # orange
        1,  # orange
        1,  # orange
    ])

    return X, y


def main():
    X, y = build_training_data()

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    # Example "test" fruits: [weight, texture]
    #  texture: 0 = smooth, 1 = bumpy
    test_samples = np.array([
        [160, 0],  # 160g, smooth
        [155, 1],  # 155g, bumpy
    ])

    predictions = clf.predict(test_samples)
    proba = clf.predict_proba(test_samples)

    label_names = {0: "apple", 1: "orange"}

    print("Training data shape:", X.shape)
    print("Test samples shape:", test_samples.shape)
    print()

    for i, sample in enumerate(test_samples):
        weight, texture = sample
        pred_label = predictions[i]
        pred_name = label_names[pred_label]
        probs = proba[i]

        print(f"Sample {i + 1}:")
        print(f"  Features: weight={weight}g, texture={'smooth' if texture == 0 else 'bumpy'}")
        print(f"  Predicted label: {pred_name} (class {pred_label})")
        print(f"  Class probabilities:")
        print(f"    apple  (0): {probs[0]:.3f}")
        print(f"    orange (1): {probs[1]:.3f}")
        print()


if __name__ == "__main__":
    main()
