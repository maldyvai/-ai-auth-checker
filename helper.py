from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io

def analyze_ela(image: Image.Image, threshold: int = 50, quality: int = 90):
    """
    Perform Error Level Analysis on the given PIL Image and highlight
    pixels above the given threshold.

    Args:
        image: Input PIL Image (RGB).
        threshold: Grayscale threshold for suspicious pixels.
        quality: JPEG quality for recompression (lower → more compression).

    Returns:
        ela_image: PIL Image of the ELA result.
        highlight_image: PIL Image with suspicious areas marked in red.
        std_dev: Standard deviation of the ELA image array.
        regions: Number of pixels above the threshold.
    """
    # 1. Force a uniform JPEG recompression to reintroduce artifacts
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')

    # 2. Compute the difference (ELA) image
    ela_image = ImageChops.difference(image, recompressed)

    # 3. Scale brightness based on max diff
    extrema = ela_image.getextrema()
    max_diff = max(channel_max for _, channel_max in extrema) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 4. Convert to arrays
    ela_arr = np.array(ela_image)
    gray = np.array(ela_image.convert('L'))  # single‐channel grayscale

    # 5. Compute stats & mask
    std_dev = float(np.std(ela_arr))
    mask = gray > threshold           # boolean mask of suspicious pixels
    regions = int(mask.sum())

    # 6. Highlight on original
    orig_arr = np.array(image)
    highlight_arr = orig_arr.copy()
    highlight_arr[mask] = [255, 0, 0]  # red overlay on suspicious pixels
    highlight_image = Image.fromarray(highlight_arr)

    return ela_image, highlight_image, std_dev, regions
