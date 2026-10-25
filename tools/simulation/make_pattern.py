"""Random dot projector pattern (1024x1024, dots ~2 px) used as the stereo projector 'gobo'."""
import sys

import cv2
import numpy as np

N, CELL = 1024, 6
rng = np.random.default_rng(12345)
img = np.zeros((N, N), np.float32)
for cy in range(0, N, CELL):
    for cx in range(0, N, CELL):
        if rng.random() < 0.55:
            y, x = cy + rng.uniform(1, CELL - 1), cx + rng.uniform(1, CELL - 1)
            cv2.circle(img, (int(x), int(y)), 1, float(rng.uniform(0.6, 1.0)), -1)
img = cv2.GaussianBlur(img, (0, 0), 0.7)
img = np.clip(img / img.max(), 0, 1)
cv2.imwrite(sys.argv[1], (img * 255).astype(np.uint8))
print("pattern", sys.argv[1], img.mean())
