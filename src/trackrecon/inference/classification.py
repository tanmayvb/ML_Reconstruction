from skimage.measure import label, regionprops
import numpy as np

def extract_structures(
        prediction,
        threshold=0.9,
        min_size=200
    ):

    mask = prediction > threshold

    labeled = label(mask)

    structures = []

    for region in regionprops(labeled):

        if region.area < min_size:
            continue

        z, y, x = region.centroid

        bbox = region.bbox

        dz = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dx = bbox[5] - bbox[2]

        elongation = max(dz,dy,dx) / (
            min(dz,dy,dx) + 1e-8
        )

        # ---------------------------------
        # Heuristic classification
        # ---------------------------------
        if elongation > 8 and region.area < 2000:

            structure_type = "axon"

        else:

            structure_type = "dendrite"

        structures.append({

            "centroid": (
                int(z),
                int(y),
                int(x)
            ),

            "volume": int(region.area),

            "elongation": float(elongation),

            "type": structure_type
        })

    return structures, labeled
