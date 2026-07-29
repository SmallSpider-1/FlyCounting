# Configuration placeholder

`counting_candidate_v1.json` is the runnable pre-tuning SF-SORT candidate, not a validation-frozen configuration. Create a separately named formal config after validation tuning; cached detections, `per_class=false`, and no appearance/GMC remain mandatory.

`marginal_timeout` and `central_timeout` use an explicit `{"fps_ratio": ...}` formula in this multi-video
candidate so each segment resolves the protocol timeout from its own FPS. Fixed numeric frame-count overrides
are rejected because they violate the baseline protocol's per-segment dynamic resolution rule.
