import sys

print("Python version:", sys.version)
print("="*50)

# ---- Torch ----
try:
    import torch
    print("torch version:", torch.__version__)
    x = torch.tensor([1.0, 2.0])
    print("torch test:", x * 2)
except Exception as e:
    print("torch ERROR:", e)

print("-"*50)

# ---- Torchvision ----
try:
    import torchvision
    print("torchvision version:", torchvision.__version__)
except Exception as e:
    print("torchvision ERROR:", e)

print("-"*50)

# ---- tifffile ----
try:
    import tifffile
    import numpy as np
    print("tifffile version:", tifffile.__version__)
    
    test_img = np.zeros((10, 10), dtype=np.uint8)
    tifffile.imwrite("test.tiff", test_img)
    img = tifffile.imread("test.tiff")
    print("tifffile test shape:", img.shape)
except Exception as e:
    print("tifffile ERROR:", e)

print("-"*50)

# ---- pandas ----
try:
    import pandas as pd
    print("pandas version:", pd.__version__)
    
    df = pd.DataFrame({"a": [1, 2, 3]})
    print("pandas test:\n", df)
except Exception as e:
    print("pandas ERROR:", e)

print("-"*50)

# ---- scikit-image ----
try:
    import skimage
    from skimage.filters import sobel
    print("scikit-image version:", skimage.__version__)
    
    img = np.random.rand(10, 10)
    edge = sobel(img)
    print("scikit-image test shape:", edge.shape)
except Exception as e:
    print("scikit-image ERROR:", e)

print("-"*50)

# ---- bioio ----
try:
    from bioio import BioImage
    print("bioio import SUCCESS")
except Exception as e:
    print("bioio ERROR:", e)

print("-"*50)

# ---- napari ----
try:
    import napari
    print("napari version:", napari.__version__)
except Exception as e:
    print("napari ERROR:", e)

print("="*50)
print("DONE")
