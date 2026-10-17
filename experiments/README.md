# Experiment workflows

These scripts preserve the Colab and Kaggle workflows used during development. Most Kaggle files are Python notebook exports with `# %%` cell boundaries and runtime-specific input paths. Install dependencies before running them. They execute work at module scope and should not be imported as library modules.

The main result tables are taken from the July thesis and defense presentations. The final Kaggle outputs were not retained. The scripts are research artifacts, not a recovered snapshot of every final cloud session.

## Setup

For a checkout available in the cloud runtime, install its `requirements.txt` in a notebook setup cell. Kaggle already supplies PyTorch; retain a CUDA-compatible build. Additional API experiments use `requirements/api.txt`.

Edit the input paths at the top of each script to match the datasets attached to the notebook. Use a writable copy of the code for perception: weight loaders may create files under `models/`. The detected evaluation and perception breakdown scripts copy their attached source to `/kaggle/working/code` for this reason.

## Main thesis workflows

| Script | Inputs | Outputs / purpose |
| :--- | :--- | :--- |
| `colab/train_nl2dsl.py` | CLEVR training and validation TSV files | Supervised T5 checkpoint on Google Drive |
| `colab/detected_eval.py` | Code, T5 weights, sampled images, converted questions | Cached graphs and resumable evaluation on Drive |
| `kaggle/yolo_finetune.py` | CLEVR training images and scene annotations | `yolov8m_clevr.pt`; 35,000 training images, 15 configured epochs |
| `kaggle/detected_eval.py` | Code, T5 weights, converted questions, images | Detected-graph evaluation summary and question records |
| `kaggle/perception_breakdown.py` | Existing graph cache or image/question inputs | Detection, attribute, and relation diagnostics |
| `kaggle/rl_humans.py` | CLEVR-trained T5, Humans training JSON, CLEVR training scenes | REINFORCE checkpoints and final model |
| `kaggle/humans_val_eval.py` | RL checkpoint, Humans validation JSON, CLEVR validation scenes | Validation with the RL experiment's interpreter |

`detected_eval.py` refuses to overwrite an existing checkpoint. Change `CKPT` for a new experiment. The Colab evaluator supports resume; keep its cached graphs and question results specific to a single model/configuration.

The supervised T5 workflow uses the prefix `translate to DSL: `. Its initial schedule is three epochs, but the retained training history ends at approximately 0.823 epochs. The RL workflow uses unprefixed questions, samples four programs per question, and evaluates answers through its own interpreter. Do not silently substitute `NL2DSLModel` when reproducing that workflow.

The RL reward implementation supports sequential programs. It does not parse explicit `:i` and `:i,j` references used by the shared DSL serializer. It retains a permissive object-selection fallback. Its KL term is computed at the first decoder position, and its baseline is the mean reward of the samples for each question. These details are preserved because changing them would define a different training experiment.

The YOLO label generator constructs approximate square boxes from object centers and size categories. These are proxy labels, not rendered instance-mask boxes. Keep this distinction when describing the detector training data.

## Additional adaptation experiments

| Script | Purpose | Interpretation |
| :--- | :--- | :--- |
| `kaggle/finetune_humans.py` | Train on heuristic question/program matches | Exploratory; input preparation uses Humans validation questions |
| `kaggle/nl2dsl_finetune.py` | Train on LLM pseudo-labels filtered by answer agreement | Unprefixed input convention; inspect source split and duplicate questions |
| `kaggle/finetune_paraphrases.py` | Adapt to paraphrased questions with CLEVR anchors | Prefixed input convention |
| `kaggle/humans_eval.py` | Compare checkpoints before/after supervised adaptation | Standalone DSL variant; not the shared validator |

The associated data tools remain under `nl2dsl/`: `prepare_humans_mix.py`, `paraphrase_pipeline.py`, `call_paraphrase_api.py`, and `llm_pseudolabel.py`. They are separate experiments from the thesis REINFORCE training on the Humans training split.

The heuristic mix matches scene, answer, and estimated terminal operation; this does not guarantee a semantically correct program. Pseudo-labels can contain several candidate programs for the same question, and the training script splits rows. Before interpreting a validation score as independent, ensure all variants of a question and scene remain in the same partition. Paraphrase validation also needs separation from the original CLEVR anchor questions.

## Local visual diagnostics

The former `tests/test_detector.py`, `tests/test_depth_estimator.py`, and `tests/test_segmenter.py` were executable visual demos rather than pytest tests. They now live in `local/`:

```bash
python -m experiments.local.detector_demo
python -m experiments.local.depth_demo
python -m experiments.local.segmenter_demo
```

They require the dataset and perception dependencies and write ignored images under `test_outputs/`. Lightweight automated tests remain in `tests/`.
