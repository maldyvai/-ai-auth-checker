# helper.py

import io
import numpy as np
from PIL import Image, ImageChops

def analyze_ela(image: Image.Image, threshold: int = 50):
    """
    Performs Error Level Analysis on a PIL image and returns:
      - ela_img: a PIL Image showing the scaled ELA result
      - highlight_img: a PIL Image where pixels above the threshold are highlighted in red
      - std: the standard deviation of the ELA difference image (grayscale)
      - regions: the count of pixels above the threshold
    """
    # 1. Re‐save to JPEG at quality=90 to get compression artifacts
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    jpeg = Image.open(buf).convert('RGB')

    # 2. Compute absolute difference (ELA)
    diff = ImageChops.difference(image.convert('RGB'), jpeg)

    # 3. Convert to grayscale and find max diff for scaling
    gray_diff = diff.convert('L')
    extrema = gray_diff.getextrema()
    max_diff = extrema[1]
    scale = 255.0 / max_diff if max_diff != 0 else 1.0

    # 4. Scale the ELA image for visibility
    ela_img = gray_diff.point(lambda x: x * scale)

    # 5. Compute std‐dev of the scaled ELA image
    ela_np = np.array(ela_img)
    std = float(ela_np.std())

    # 6. Build a binary mask of pixels above the threshold
    mask = ela_np > threshold
    regions = int(mask.sum())

    # 7. Overlay mask in red on the original image
    highlight_arr = np.array(image.convert('RGB')).copy()
    highlight_arr[mask] = [255, 0, 0]   # red highlights
    highlight_img = Image.fromarray(highlight_arr)

    return ela_img, highlight_img, std, regions
