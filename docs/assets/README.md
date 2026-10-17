# Figure sources

## Figures used in the README

The main figures are taken from the Serbian thesis, `DIPLOMSKI_RAD_BOGDAN_PDF_JUL.pdf`. The copy restored under `docs/presentation/` has SHA-256 `8f2f8e2cd4fc981f493336d97da2627918683168ba5dce88d790eb8640d0b46a`, matching the source used for the result tables.

| README asset | Original source | Changes |
| :--- | :--- | :--- |
| [System architecture](thesis_architecture_en.png) | Figure 4.1, printed p. 20, PDF page 29 | English labels; original layout, arrows, image panels, and DSL example retained |
| [Training curves](thesis_training_en.png) | Figure 6.1, printed p. 48, PDF page 57 | English axis and legend labels; original curves, colors, scales, and tick values retained |
| [Input scene](source/scene.png) | `plots/fig4_1/a_original.png` | None |
| [Inverse depth](source/depth.png) | `plots/fig4_1/d_depth.png` | None |
| [Instance masks](source/masks.png) | `plots/fig4_1/c2_zbuffer.png` | None |

The unmodified images extracted from the PDF are saved as [source/thesis_architecture.png](source/thesis_architecture.png) and [source/thesis_training.png](source/thesis_training.png).

The translations are stored in [thesis_architecture_en.svg](thesis_architecture_en.svg) and [thesis_training_en.svg](thesis_training_en.svg). Each SVG embeds the original bitmap and places editable English text over its Serbian labels. PNG exports are used in the README for consistent rendering. These are translations of the existing figures, not newly drawn diagrams or replotted measurements.

The small detection and scene-graph panels in Figure 4.1 use the matching higher-resolution notebook exports, [source/detection.png](source/detection.png) and [source/scene_graph.png](source/scene_graph.png), with translated annotations. Confidence values, object positions, colors, and IDs are retained. All perception examples show saved notebook outputs; no new inference was run to produce them.

The notebook's configured target is `CLEVR_val_000138`. Confirm the exact image identity against the dataset if an image identifier is needed in a publication.

## Attribution

CLEVR image attribution: Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick, *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*, CVPR 2017. See the [dataset page and CC BY 4.0 notice](https://cs.stanford.edu/people/jcjohns/clevr/).
