# PERL

PERL (Perception-to-Event Reasoning Layer) is the deterministic engineering layer used in the edge-video application of SMFR-YOLO.

It converts tracked `person`, `helmet`, and `vest` detections into zone-aware safety states and events through:

1. person–PPE association;
2. ground-contact-point estimation;
3. polygonal zone membership;
4. PPE, occupancy, and dwell-rule evaluation;
5. temporal confirmation; and
6. event OPEN/CLOSE lifecycle management.

`perl_runtime.py` provides the reference implementation used for the monitoring application.

The runtime is configuration driven. Zone geometry, rule enablement, thresholds, and timing parameters are application-specific and should be defined for the target camera and scene.

The publication-specific scene configurations are intentionally not distributed in this repository because their polygon coordinates and thresholds are tied to the demonstration videos rather than to SMFR-YOLO training or detector evaluation.

The PERL layer is deterministic engineering logic and is not an additional learned model.