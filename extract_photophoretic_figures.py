"""Extract paper figures as source-image crops for the presentation."""

from pathlib import Path

import fitz


root = Path(__file__).resolve().parent
output = root / "tmp" / "photophoretic_figures"
output.mkdir(parents=True, exist_ok=True)

with fitz.open(root / "nature25176.pdf") as paper:
    crops = [
        (1, (38, 55, 561, 392)),
        (2, (90, 146, 510, 470)),
        (3, (65, 58, 528, 368)),
        (2, (90, 53, 510, 145)),
    ]
    for number, (page_index, bounds) in enumerate(crops, start=1):
        pix = paper[page_index].get_pixmap(
            matrix=fitz.Matrix(3, 3), clip=fitz.Rect(*bounds), alpha=False
        )
        pix.save(output / f"figure{number}.png")
