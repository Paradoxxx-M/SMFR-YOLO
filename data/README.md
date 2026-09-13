# Data

The primary benchmark combines 1,206 RF100 Construction Safety v2 images and 494 ITPPE images, with 14,559 annotated instances. Its exact membership is 1,451 training, 139 validation, and 110 held-out test images. The three classes are `person`, `helmet`, and `vest`.

Third-party RF100 and CPPE images are not redistributed. ITPPE images and annotations are provided through the persistent archive described in `itppe/README.md`. Use `scripts/prepare_primary_dataset.py` to assemble the primary benchmark after downloading both image sources.
