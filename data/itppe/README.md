# ITPPE

ITPPE (Industrial Target-domain Personal Protective Equipment Dataset) is an author-collected industrial-scene PPE dataset containing 494 images and 7,163 YOLO-format object instances across three classes: `person` (2,942 instances), `helmet` (2,509 instances), and `vest` (1,712 instances).

The fixed ITPPE split used in this study contains 454 training images, 20 validation images, and 20 held-out test images.

## Data access

During double-anonymized peer review, the complete ITPPE image-and-annotation archive is available through the anonymous Zenodo reviewer-access link supplied with the manuscript. The secret reviewer URL is intentionally not embedded in this public repository.

## Using ITPPE with this repository

After extracting the archive, pass the extracted `ITPPE/` directory—the directory containing `images/` and `labels/`—to `scripts/prepare_primary_dataset.py` through `--itppe-root`.

```bash
python scripts/prepare_primary_dataset.py \
  --rf100-root path/to/rf100 \
  --itppe-root path/to/ITPPE \
  --output path/to/primary