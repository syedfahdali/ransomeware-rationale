# Laya fine-tuning sets

Built by `scripts/make_laya_finetune_set.py` from this repo's own artifacts.

| set | rows | independent labels | split |
|---|---|---|---|
| gate | 0 | 0 | {} |
| claims | 90 | 90 | {'test': 90} |

`independent` = the label does **not** come from the component being
cross-checked. Rows with `"independent": false` carry
`"label_provenance": "classifier_verdict"` (gate) or `"rule_verifier"`
(claims) and `"circular_for"` naming the component they would teach the model
to imitate.

Labels whose provenance is `classifier_verdict` (gate) or `rule_verifier` (claims) come from the very component the model is meant to cross-check. A cross-judge trained on them can only reproduce that component's errors; a gate trained on the classifier's verdict cannot disagree with the classifier. Group-aware splitting fixes train/test leakage, not label provenance circularity.

Run with `--allow-circular` to include the circular bulk set; the default emits
only the independent rows, which is why the gate set is empty unless you opt in.
