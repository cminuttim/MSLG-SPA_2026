# AxoloTux @ MSLG-SPA 2026

Code, prompts and configuration for the AxoloTux submissions to the
MSLG-SPA 2026 shared task on bidirectional translation between
Mexican Sign Language Glosses (MSLG) and Spanish (IberLEF 2026).
AxoloTux ranked first in the official global ranking.

## Repository layout

```
MSLG-SPA_2026/
├── code/
│   ├── train_mslg_translator_qwen_9b.py    Main LoRA fine-tuning script
│   ├── train_classifier.py                 Gradient-boosting scorer training
│   └── generate_negatives.py               Builds negative pairs for the scorer
├── prompts/
│   ├── system_prompt.md                    Used by Claude Opus 4.5 and Gemini 3 Pro
│   └── user_prompt.txt                     Earlier minimal user-prompt variant
└── config/
    └── train_config.yaml                   Hyperparameters for all Q* variants
```

LoRA adapter weights for the fine-tuned variants are not included in
this repository because of their size. They will be released
separately on the Hugging Face Hub; see the link in the paper.

## Submitted variants

| Variant       | Approach                          | Training data                                                  |
|---------------|-----------------------------------|----------------------------------------------------------------|
| `QBase`       | Qwen 3.5 9B (8-bit) + LoRA        | Challenge corpus (490 pairs, 3 epochs)                         |
| `QDistil2`    | Qwen 3.5 9B (8-bit) + LoRA        | Lara-Ortiz Base (1 warm-up epoch) + Extended (3 epochs)        |
| `QDistilTest` | Qwen 3.5 9B (8-bit) + LoRA        | Challenge + Claude/Gemini predictions on test inputs (3 epochs)|
| `QDistilMix`  | Ensemble selected by scorer       | Post-hoc over the other submissions                            |
| `G`           | Gemini 3 Pro prompted             | Challenge corpus as in-context examples                        |
| `C`           | Claude Opus 4.5 prompted          | Challenge corpus as in-context examples                        |

The Qwen 3.5 9B base model is the abliterated variant
`huihui-ai/Huihui-Qwen3.5-9B-abliterated`.

## Reproducing a fine-tune

The full training pipeline for any `Q*` variant is driven by
`code/train_mslg_translator_qwen_9b.py`. The hyperparameters in
`config/train_config.yaml` reflect the configuration used for the
submitted runs; the differences between variants are limited to the
Phase 1/Phase 2 corpus selection documented in the same file. The
gradient-boosting scorer used for Stage 1 model selection and for
the `QDistilMix` ensemble is trained by `code/train_classifier.py`,
on the positive/negative pairs produced by
`code/generate_negatives.py`.

## Hardware

All LoRA fine-tuning and inference for the Qwen 3.5 9B runs were
performed on a single consumer-grade NVIDIA RTX 4090 (24 GB VRAM).
One Phase 2 specialization run of 3 epochs on the 490-pair challenge
corpus completes in under one hour.

## License

The code in this repository is released under the MIT license. The
Qwen 3.5 9B base model and its abliterated variant are subject to
their respective upstream licenses; consult the model cards on the
Hugging Face Hub before redistributing or deploying the adapters.

## Citation

If you use any of this material, please cite the AxoloTux system
description paper and the shared-task overview:

```bibtex
@inproceedings{axolotux2026mslgspa,
  title  = {LoRA Fine-Tuning, Frontier-LLM Prompting, and Encoder-Decoder
            Baselines for Low-Resource Bidirectional Sign Language Gloss
            Translation},
  author = {Minutti-Martinez, Carlos and Torres-N{\'a}jera, Alejandra and
            Escalante-Ramirez, Boris and Olveres, Jimena},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum
               (IberLEF 2026)},
  year   = {2026},
}
```
