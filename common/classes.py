"""DOTA-v1.0 class definitions, shared by all pipelines.

The 15 categories of the DOTA aerial-image detection benchmark, in the canonical
order used throughout the project. ``CLASS2ID`` maps a class name to its integer id.
"""

DOTA_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle",
    "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank",
    "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter",
]

CLASS2ID = {name: i for i, name in enumerate(DOTA_CLASSES)}

NUM_CLASSES = len(DOTA_CLASSES)
