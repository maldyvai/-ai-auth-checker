# helper.py

import io
import numpy as np
from PIL import Image, ImageChops

def analyze_ela(image: Image.Image, threshold: int = 50, quality: int = 60):
    """
    Error Level Analysis with adjustable JPEG quality.
    Returns:
      - ela_img: grayscale ELA image scaled for visibility
      - highlight_img: original with red overlay where diff > threshold
      - std: standard deviation of the ELA image
      - regions: count of pixels above threshold
    """
    # 1) Re-save at a lower JPEG quality to amplify compression artifacts
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    jpeg = Image.open(buf).convert('RGB')

    # 2) Compute absolute difference
    diff = ImageChops.difference(image.convert('RGB'), jpeg)

    # 3) Grayscale & find max for scaling
    gray = diff.convert('L')
    max_diff = gray.getextrema()[1]
    scale = 255.0 / max_diff if max_diff else 1.0

    # 4) Scale up for visibility
    ela_img = gray.point(lambda x: x * scale)

    # 5) Stats
    ela_np = np.array(ela_img)
    std = float(ela_np.std())

    # 6) Mask & count
    mask = ela_np > threshold
    regions = int(mask.sum())

    # 7) Highlight on original
    orig_np = np.array(image.convert('RGB')).copy()
    orig_np[mask] = [255, 0, 0]
    highlight_img = Image.fromarray(orig_np)

    return ela_img, highlight_img, std, regions
