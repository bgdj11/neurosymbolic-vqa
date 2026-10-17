# Implementation review

Reviewed against baseline commit `df1fa922ac6ce39ffd937dbf70a0580ab78db34e` (`llm eval script`, 18 April 2026), the working tree, the July thesis PDF, and both defense presentations. Final benchmark values come from the thesis. This review does not reinterpret old local summaries as final results.

## Existing research differences to check

These behaviors existed before the repository cleanup. They were preserved so that organizational changes do not silently redefine the experiments.

| Area | Finding | Files to inspect |
| :--- | :--- | :--- |
| Object selection | The thesis describes strict `unique`. Current code selects the largest bounding box on ambiguous input and can select from the entire scene on empty input. Objects without boxes tie at zero area. This affects execution-error rates. | `dsl/operators.py`, `dsl/dsl_types.py` |
| Spatial relations | Current front/behind relations use normalized negative bounding-box bottom coordinates, rather than MiDaS depth statistics. Horizontal and depth thresholds also differ from the methodology chapter. | `scene_graph/relations.py` |
| Perception models | Code defaults use YOLOv8m, FastSAM-x for proposals, FastSAM-s for crop segmentation, and CLIP ViT-L/14. The thesis names FastSAM-s and CLIP ViT-B/32. Fine-tuned YOLO weights must be selected explicitly. | `perception/detector.py`, `perception/box_segmenter.py`, `perception/attribute_extractor.py` |
| Color and size | Current code adds HSV corrections and a vertical-position-dependent size threshold. The methodology describes prompt classification and a median-area size rule. | `perception/attribute_extractor.py` |
| Config file | `configs/default.yaml` is a reference snapshot, not a configuration consumed by the current evaluation CLI. Several values differ from Python defaults. | `configs/default.yaml`, `eval/run_eval.py` |
| Image arrays | The builder constructs RGB arrays, while several perception array APIs interpret arrays as BGR. Paths and in-memory images can therefore take different color-conversion routes. | `scene_graph/builder.py`, `perception/depth_estimator.py`, `perception/box_segmenter.py` |
| MiDaS preprocessing | `MiDaS_small` currently selects `dpt_transform`. Verify this against the intended small-model preprocessing before new perception runs. | `perception/depth_estimator.py` |
| Occlusion handling | Fully pruned masks become `None`, but detections remain in the list and can still contribute bounding-box crops and graph nodes. The thesis describes excluding fully occluded detections. | `perception/postprocess.py`, `scene_graph/builder.py` |
| CLIP scores | HSV can change a predicted color without recalculating the accompanying CLIP confidence and score dictionary. | `perception/attribute_extractor.py` |
| RL execution | The RL training and evaluation files retain a sequential interpreter that cannot parse branched `:i,j` references. Questions are unprefixed; the shared supervised inference code prefixes them. | `experiments/kaggle/rl_humans.py`, `experiments/kaggle/humans_val_eval.py`, `nl2dsl/infer.py` |
| RL objective | The baseline is computed per question; KL is evaluated only at the first decoder position. The code uses token-averaged negative loss rather than a summed sequence log-probability. | `experiments/kaggle/rl_humans.py` |
| Validation contamination | The heuristic Humans adaptation uses validation questions. Pseudo-label row splits can place variants of the same question in both partitions. Paraphrase anchors can overlap validation originals. These are separate from the thesis RL train-split experiment. | `nl2dsl/prepare_humans_mix.py`, `nl2dsl/paraphrase_pipeline.py`, `experiments/kaggle/nl2dsl_finetune.py` |
| Detector supervision | Training boxes are approximated from annotated centers and size labels, not obtained from rendered segmentation masks. | `experiments/kaggle/yolo_finetune.py` |
| Resume provenance | Evaluation checkpoints are keyed by question ID and are not bound to model hashes or perception settings. Reusing a checkpoint with a smaller question limit can also retain results beyond the requested subset. | `eval/run_eval.py`, `eval/run_llm_eval.py`, cloud evaluators |
| Perception metrics | `eval/perception_eval.py` uses capped object counts for recall and expects object-level relations, whereas the builder emits top-level adjacency lists. Other diagnostics use 60-pixel matching and relation-set equality; the thesis specifies a 50-pixel protocol. | `eval/perception_eval.py`, `scene_graph/validate_sg.py`, `experiments/kaggle/perception_breakdown.py` |
| Conversion validation | The older validator only accepts CLEVR/GQA and looks under `converters/converted`, while newer converters write under `datasets/converted`. | `converters/validate.py` |
| Generalization claims | The T5 training data is standard CLEVR rather than CoGenT-A only. Literature comparison values should not be described as a controlled reproduction of the restricted CoGenT protocol. | Thesis Table 6.9 and presentation slide 16 |

## Corrections made during cleanup

* Replaced embedded API credentials with `ANTHROPIC_API_KEY` environment lookup and a missing-key check. No API requests were made.
* Removed a stray Markdown fence from the paraphrase fine-tuning script and corrected the pseudo-label summary's undefined model variable.
* Corrected the Kaggle detected evaluator's undefined `done` reference, preserved existing checkpoint files, and added question categories to its results.
* Made cloud perception scripts use a writable code copy and replaced implicit notebook-variable references with explicit `globals().get(...)` lookup.
* Moved dependency installation out of cloud scripts into documented environment setup.
* Created evaluation output directories before opening checkpoints. Deferred the T5 import until actual evaluation so `--help` works without loading neural dependencies.
* Rejected negative, non-integer, Boolean, and forward step references in the shared executor. Regression tests cover these cases.
* Corrected logged FP16 metadata to reflect the actual training argument and bounded the Colab validation subset by its available length.
* Removed an unsupported `difference` operator from the pseudo-label prompt.
* Translated scene-graph plot labels, cleaned notebook outputs, and corrected one nested f-string for Python 3.10 compatibility.

The training algorithms, pre-existing model thresholds, RL reward definitions, and final thesis values were not retuned during cleanup.

## Documentation details worth checking in the thesis

* The English documentation sheet exchanges the mentor/member names relative to the Serbian sheet and the presentation.
* The task sheet contains a leftover sentence fragment beginning with `Humans`.
* The methodology and results sections describe different size-classification mechanisms; size is heuristic in the code.
* The description of `relationships["left"][i]` reverses the direction used by the executor and CLEVR adjacency lists.
* The configured three training epochs should be distinguished from the approximately 0.823 epochs actually present in the saved Trainer history.
* Multiplying marginal attribute accuracies gives a rough independence-based illustration; it does not establish which component caused every end-to-end error.

The thesis and presentation binaries were not edited.
