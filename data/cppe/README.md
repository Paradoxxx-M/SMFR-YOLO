# CPPE external evaluation

The 932-image construction PPE dataset is downloaded from its PPE-Detection-Pose release:

https://github.com/ruoxinx/PPE-Detection-Pose

Images are not redistributed. From the repository root, convert the source Pascal VOC annotations with the fixed mapping and membership:

```bash
python scripts/prepare_cppe.py --source path/to/cppe --output path/to/cppe_yolo
```

The script reads `mapping.tsv`, retains the numeric image identities, and uses `train.txt`, `val.txt`, and `test.txt` without resampling. It checks missing files, duplicate IDs, overlap, class counts, and the exact 746/93/93 image counts before writing `dataset.yaml`. Use `--validate-only` to run all source and conversion checks without writing a dataset.

Protocol A evaluates a primary-domain checkpoint zero-shot and uses Test93 as the final endpoint. Protocol B trains from scratch on Train746 for 100 epochs with `pretrained: false`, selects the best checkpoint on Val93, and reports Test93. Formal validation and test evaluation use batch 32, `half: false`, and image size 640.
