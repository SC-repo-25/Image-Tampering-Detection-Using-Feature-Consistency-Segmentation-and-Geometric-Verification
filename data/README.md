# data/

## `sample/`
Small individual sample images for ad-hoc `--input` runs and demos.

## `test/`
Labeled dataset for `--evaluate` mode. Expected layout:

```
data/test/
├── authentic/*.jpg
└── tampered/*.jpg
```

### Generating a small synthetic dev set (used in this repo's README/report)

No external dataset is bundled (see main README §13 for licensing reasons).
The following script recreates the exact 6-image synthetic dev set used to
produce the metrics quoted in the README/report:

```python
import cv2, numpy as np
rng = np.random.default_rng(42)

def make_textured_image(w=800, h=600):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 90, 120)
    for _ in range(40):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(10, 40)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1)
    for _ in range(30):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

for i in range(3):
    cv2.imwrite(f'data/test/authentic/auth_{i}.jpg', make_textured_image())

for i in range(3):
    img = make_textured_image()
    patch = img[100:220, 100:260].copy()
    img[350:470, 500:660] = patch
    cv2.imwrite(f'data/test/tampered/tamp_{i}.jpg', img)
```

### Recommended real datasets for proper evaluation

See main `README.md` §13 (CASIA v2.0, CoMoFoD) — not bundled here due to
size/licensing; both are freely obtainable from their respective project
pages/request forms.

### Custom small datasets

A custom set is acceptable for course demonstration purposes **as long as
this is stated explicitly** wherever results are reported (see
`docs/methodology/threshold_calibration.md` for the honesty standard
followed throughout this project).
