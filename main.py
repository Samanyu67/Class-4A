import numpy as np

names = np.array(["book", "laptop", "chair"])

objectness = np.array([0.92, 0.45, 0.88, 0.30, 0.76, 0.81])

name_scores = np.array([
    [0.85, 0.10, 0.05],
    [0.20, 0.70, 0.10],
    [0.10, 0.80, 0.10],
    [0.60, 0.30, 0.10],
    [0.15, 0.15, 0.70],
    [0.90, 0.05, 0.05]
])

scores = objectness[:, np.newaxis] * name_scores
print("Part 1 - Scores:\n", scores)

best_indices = np.argmax(scores, axis=1)
best_names = names[best_indices]
best_scores = np.max(scores, axis=1)
print("\nPart 2 - Best Names:", best_names)
print("Part 2 - Best Scores:", best_scores)

threshold = 0.5
keep = best_scores >= threshold
print("\nPart 3 - Kept Names:", best_names[keep])
print("Part 3 - Kept Scores:", best_scores[keep])

thresholds = [0.3, 0.5, 0.7]
print("\nPart 4 - Threshold Counts:")
for t in thresholds:
    count = np.sum(best_scores >= t)
    print(f"Threshold {t}: {count} kept")
