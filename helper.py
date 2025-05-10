from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def analyze_ela(image):
    pil = image.copy()
    ioipp = ImageChops.difference(pil, pil)
    # ELA: re-save and diff
    import io
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    compressed = Image.open(buf)
    diff = ImageChops.difference(pil, compressed)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    factor = 255.0 / max_diff
    ela = ImageEnhance.Brightness(diff).enhance(factor)
    arr = np.array(ela)
    std = np.std(arr)
    regions = np.count_nonzero(arr > 50)
    # highlight
    orig = np.array(pil)
    highlight = orig.copy()
    highlight[arr > 50] = [255, 0, 0]
    from PIL import Image as PILImage
    return ela, PILImage.fromarray(highlight), std, regions
