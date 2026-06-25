#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mslg_translator_qwen_9b.py

Fine-tunes Qwen 3.5 9B with LoRA for bidirectional MSLG ↔ SPA translation.

Training schedule (curriculum):
  Phase 1 — 1 epoch  on corpus.csv          (broader, noisier corpus)
  Phase 2 — 5 epochs on MSLG_SPA_train.txt  (curated corpus)
             2 checkpoints saved per epoch (every ½ epoch)

Each epoch trains BOTH directions (MSLG→SPA and SPA→MSLG) using Qwen's
native chat format (<|im_start|>/<|im_end|>). Thinking mode is disabled
via <think>\n\n</think>\n\n prefix so the model outputs JSON directly.
Model output is always a JSON object with both "MSLG" and "SPA".

LoRA is tuned conservatively (r=16, dropout=0.10) to limit overfitting on
the small curated corpus (490 pairs). Batch size is reduced to 2 (vs 4 for
the 4B variant) with gradient accumulation doubled to 8 to keep the same
effective batch of 16 while fitting the larger model in GPU memory.

Modes:
  train            — full curriculum from scratch
  resume           — auto-detect where to continue
  translate        — translate text using the final adapter or any checkpoint
  list-checkpoints — show all saved Phase 2 checkpoints

Usage:
  python train_mslg_translator_qwen_9b.py --mode train
  python train_mslg_translator_qwen_9b.py --mode resume
  python train_mslg_translator_qwen_9b.py --mode list-checkpoints
  python train_mslg_translator_qwen_9b.py --mode translate --texto "YO COMPRAR AYER" --direction mslg2spa
  python train_mslg_translator_qwen_9b.py --mode translate --texto "Compré un libro." --direction spa2mslg
  python train_mslg_translator_qwen_9b.py --mode translate --texto "YO IR MAÑANA" --checkpoint checkpoint-56
"""

import argparse
import json
import math
import os
import re
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset as HfDataset, DatasetDict
from huggingface_hub import login
from peft import (LoraConfig, PeftModel, TaskType,
                  get_peft_model, prepare_model_for_kbit_training)
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig, DataCollatorForSeq2Seq,
                           Trainer, TrainingArguments)

# ─── Config ───────────────────────────────────────────────────────────────────

class CFG:
    #model_name_or_path = "Qwen/Qwen3.5-9B"
    model_name_or_path = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
    #model_name_or_path = "nDimensional/Qwen3.5-9B-Uncensored-Safetensors"
    model_out          = "./MSLG_SPA_Translator_Qwen35_9B_LoRA_"

    # Corpora
    corpus_csv = "corpus.csv"          # Phase 1 — broader/noisier
    #main_txt   = "MSLG_SPA_train_destil.txt"  # destill test
    #main_txt   = "train_corpus_destil.txt"  # Phase 2 — curated
    main_txt   = "MSLG_SPA_train.txt"  # Phase 2 — curated

    # Splits
    test_size  = 0.10
    seed       = 42

    # LoRA — small rank + higher dropout to reduce overfitting on 490 pairs
    lora_r              = 16
    lora_alpha          = 32            # conventional: alpha = 2 * r
    lora_dropout        = 0.10
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]

    # Quantization: 4 = NF4 4-bit (~5 GB), 8 = LLM.int8 8-bit (~10 GB)
    quant_bits             = 8

    # Training — lower LR for translation (precision > creativity)
    batch_size             = 1
    grad_accumulation      = 16          # effective batch = 16
    lr                     = 1e-4
    w_decay                = 0.01
    warmup_ratio           = 0.05
    max_length             = 1024  # system prompt alone = ~691 tokens; 1024 covers prompt+completion
    logging_steps          = 25

    # Phase 2 schedule
    phase2_epochs          = 3          # total epochs on curated corpus
    checkpoints_per_epoch  = 3          # how many checkpoints to save per epoch
    save_total_limit       = 20         # keep all checkpoints (5 epochs × 2 = 10)

    # Inference — beam search for translation quality
    max_new_tokens     = 128
    num_beams          = 4
    length_penalty     = 1.0
    repetition_penalty = 1.05

    # Derived paths (re-computed if --out is overridden)
    @property
    def phase1_dir(self):     return self.model_out + "/phase1"
    @property
    def phase2_dir(self):     return self.model_out + "/phase2"
    @property
    def phase1_adapter(self): return self.model_out + "/phase1/final_adapter"
    @property
    def phase2_adapter(self): return self.model_out + "/phase2/final_adapter"
    @property
    def tokenizer_path(self): return self.model_out + "/tokenizer"


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    corpus.csv arrives fully lowercase:
      - MSLG → uppercase   (e.g. "yo ir" → "YO IR")
      - SPA  → first letter capitalized only  (e.g. "yo voy" → "Yo voy")
    """
    df = df.copy().dropna(subset=["MSLG", "SPA"])
    df["MSLG"] = df["MSLG"].apply(lambda x: str(x).strip().upper())
    df["SPA"]  = df["SPA"].apply(lambda x: str(x).strip())
    df["SPA"]  = df["SPA"].apply(lambda x: x[0].upper() + x[1:] if x else x)
    return df.reset_index(drop=True)


def load_main_corpus(path: str) -> pd.DataFrame:
    """Load MSLG_SPA_train.txt (tab-separated: id / MSLG / SPA)."""
    df = pd.read_csv(path, sep="\t", header=0, names=["id", "MSLG", "SPA"])
    return df[["MSLG", "SPA"]].dropna().reset_index(drop=True)


# ─── Prompt / completion (JSON format) ───────────────────────────────────────

SYSTEM_PROMPT = """\
Eres un traductor entre Lengua de Señas Mexicana Glosada (MSLG) y español (SPA). MSLG es la transcripción de una lengua real con gramática propia, no una variante del español.

Traduce de forma literal y fiel. NO suavices, omitas ni reformules términos que parezcan negativos, médicos, descriptivos de discapacidad o coloquiales (SORDO, CIEGO, FLOJO, FEO, ENFERMO, EMBARAZADA, ALCOHOL, AUTISMO, MORIR, ARRESTADO, etc.). Son traducciones legítimas. Tu rol es traducir, no editar.

Notación MSLG (todo en MAYÚSCULAS):
- Guion `-` entre palabras = un solo signo: `YA-VEO`, `TARJETA-DE-CRÉDITO`.
- `+` = signo compuesto: `MAMÁ+PAPÁ` (padres), `HERMANO+MUJER` (hermana).
- `#` = préstamo deletreado: `#TV`, `#SEP`.
- `dm-` = nombre/palabra deletreada manualmente: `dm-LUIS`.
- Reduplicación = plural o intensidad: `NIÑO NIÑO` (niños), `TRABAJO TRABAJO` (muy trabajador).
- Sin cópula ni artículos: `ÉL SORDO` = "él es sordo".
- Negación al final o con `NO-`: `ALCOHOL YO NO-GUSTAR`.
- Perfectivo con `YA`: `YO YA GANAR` = "yo gané".
- Marcadores temporales al inicio: `AYER`, `MAÑANA`, `PRÓXIMO X`.
- Preguntas: WH al final, a veces reduplicada: `¿DÓNDE TUYO LIBRO DÓNDE?`.

Ejemplos:
- `MI HERMANO+MUJER YA EMBARAZADA` ↔ "Mi hermana está embarazada."
- `dm-PABLO FLOJO` ↔ "Pablo es flojo."
- `ALCOHOL YO NO-GUSTAR` ↔ "A mí no me gusta el alcohol."
- `NIÑO NIÑO TENER PIOJO` ↔ "Los niños tienen piojos."
- `AYER COCA-COLA YO COMPRAR` ↔ "Yo compré una Coca Cola ayer."
- `#TV PUBLICIDAD HABER MUCHO` ↔ "En la TV hay mucha publicidad."

SPA→MSLG: aplica gramática signada, MAYÚSCULAS, notación correcta. MSLG→SPA: produce español natural con mayúscula inicial y puntuación. No añadas información.

Entrada: JSON con "fuente" ("MSLG" o "SPA"), "texto", "objetivo".
Salida: SOLO un JSON con "MSLG" y "SPA". Sin texto extra, sin explicaciones, sin code fences.

Ejemplo Entrada: {"fuente": "MSLG", "texto": "MI HERMANO+MUJER YA EMBARAZADA", "objetivo": "SPA"} 
Ejemplo Salida: {"MSLG": "MI HERMANO+MUJER YA EMBARAZADA", "SPA": "Mi hermana está embarazada."}\
"""

# Qwen 3.5 chat tokens
EOS        = "<|im_end|>"
# Disables Qwen3.5 thinking mode — model outputs translation directly without reasoning
THINK_SKIP = "<think>\n\n</think>\n\n"


def _instruction_json(mslg: str, spa: str, direction: str) -> str:
    if direction == "mslg2spa":
        payload = {"fuente": "MSLG", "texto": mslg, "objetivo": "SPA"}
    else:
        payload = {"fuente": "SPA", "texto": spa, "objetivo": "MSLG"}
    return json.dumps(payload, ensure_ascii=False)


def _completion_json(mslg: str, spa: str) -> str:
    return json.dumps({"MSLG": mslg, "SPA": spa}, ensure_ascii=False)


def build_prompt(mslg: str, spa: str, direction: str) -> str:
    instr = _instruction_json(mslg, spa, direction)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instr}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_SKIP}"
    )


def build_full_text(mslg: str, spa: str, direction: str) -> str:
    return build_prompt(mslg, spa, direction) + _completion_json(mslg, spa) + EOS


# ─── Dataset construction ─────────────────────────────────────────────────────

def pairs_to_records(df: pd.DataFrame) -> list:
    """
    For each pair produce two examples (one per direction), interleaved so
    both tasks appear uniformly across every batch.
    """
    records = []
    for _, row in df.iterrows():
        mslg = str(row["MSLG"]).strip()
        spa  = str(row["SPA"]).strip()
        records.append({"MSLG": mslg, "SPA": spa, "direction": "mslg2spa"})
        records.append({"MSLG": mslg, "SPA": spa, "direction": "spa2mslg"})
    return records


def build_dataset(records: list, tokenizer, cfg: CFG) -> DatasetDict:
    """Tokenize with prompt-masking and split into train / validation."""

    def _tokenize(examples):
        ids_list, masks_list, labels_list = [], [], []
        n = len(examples["MSLG"])
        for i in range(n):
            mslg      = examples["MSLG"][i]
            spa       = examples["SPA"][i]
            direction = examples["direction"][i]

            full   = build_full_text(mslg, spa, direction)
            prompt = build_prompt(mslg, spa, direction)

            enc_full   = tokenizer(full,   truncation=True,
                                   max_length=cfg.max_length, add_special_tokens=True)
            enc_prompt = tokenizer(prompt, truncation=True,
                                   max_length=cfg.max_length, add_special_tokens=True)

            input_ids = enc_full["input_ids"]
            labels    = list(input_ids)

            # Mask the prompt so loss is computed only on the JSON completion
            prompt_len          = len(enc_prompt["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

            ids_list.append(input_ids)
            masks_list.append(enc_full["attention_mask"])
            labels_list.append(labels)

        return {"input_ids": ids_list, "attention_mask": masks_list, "labels": labels_list}

    raw   = HfDataset.from_list(records)
    split = raw.train_test_split(test_size=cfg.test_size, seed=cfg.seed)

    keep = {"input_ids", "attention_mask", "labels"}
    return DatasetDict({
        name: ds.map(_tokenize, batched=True,
                     remove_columns=[c for c in ds.column_names if c not in keep])
        for name, ds in split.items()
    })


# ─── Model helpers ────────────────────────────────────────────────────────────

def _compute_dtype():
    if torch.cuda.is_available():
        return (torch.bfloat16
                if torch.cuda.get_device_properties(0).major >= 8
                else torch.float16)
    return torch.float32


def _bnb_config(cfg: CFG, dtype):
    if cfg.quant_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def _load_base(cfg: CFG, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=_bnb_config(cfg, dtype),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return prepare_model_for_kbit_training(model)


def load_model_new_lora(cfg: CFG, dtype) -> PeftModel:
    """Base model + fresh LoRA adapter for Phase 1."""
    model = _load_base(cfg, dtype)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_from_adapter(cfg: CFG, adapter_path: str, dtype,
                             trainable: bool = True) -> PeftModel:
    """
    Base model + existing LoRA adapter.
    trainable=True  → fine-tune further (Phase 2 or resume)
    trainable=False → inference only
    """
    base = _load_base(cfg, dtype)
    try:
        model = PeftModel.from_pretrained(base, adapter_path,
                                           is_trainable=trainable)
    except TypeError:
        # Older PEFT versions lack is_trainable parameter
        model = PeftModel.from_pretrained(base, adapter_path)
        if trainable:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
    if trainable:
        model.print_trainable_parameters()
    return model


# ─── Training phase ───────────────────────────────────────────────────────────

def report_best_checkpoint(out_dir: str, trainer) -> None:
    """
    After training completes:
      1. Print a table of every checkpoint with its eval_loss.
      2. Write an eval_loss.txt file inside each checkpoint directory so
         the value is accessible later (e.g. by list_checkpoints).
    """
    # Build step → eval_loss from the Trainer's log history
    step_to_loss: dict = {}
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and "step" in entry:
            step_to_loss[int(entry["step"])] = float(entry["eval_loss"])

    if not step_to_loss:
        print("  (no evaluation records in trainer history)")
        return

    # Collect checkpoint dirs sorted by step number
    ckpts = sorted(
        (d for d in os.listdir(out_dir) if d.startswith("checkpoint-")),
        key=lambda x: int(x.split("-")[-1]),
    ) if os.path.isdir(out_dir) else []

    if not ckpts:
        print("  (no checkpoint directories found)")
        return

    # Match checkpoints to losses and locate the best
    ckpt_info: dict = {}
    best_ckpt, best_loss = None, float("inf")
    for ckpt in ckpts:
        step = int(ckpt.split("-")[-1])
        loss = step_to_loss.get(step)
        ckpt_info[ckpt] = (step, loss)
        if loss is not None and loss < best_loss:
            best_loss, best_ckpt = loss, ckpt

    # Write eval_loss.txt into every checkpoint directory
    for ckpt, (step, loss) in ckpt_info.items():
        if loss is not None:
            with open(os.path.join(out_dir, ckpt, "eval_loss.txt"), "w") as fh:
                fh.write(f"eval_loss={loss:.6f}\nstep={step}\n")

    # Print summary table
    print(f"\n{'─'*64}")
    print(f"Checkpoint evaluation summary — {out_dir}")
    print(f"  {'Checkpoint':<25}  {'Step':>6}  {'eval_loss':>12}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*12}")
    for ckpt in ckpts:
        step, loss = ckpt_info[ckpt]
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        marker   = "  ← best" if ckpt == best_ckpt else ""
        print(f"  {ckpt:<25}  {step:>6}  {loss_str:>12}{marker}")
    print(f"{'─'*64}")
    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt}  (eval_loss={best_loss:.6f})")
        print(f"  → Use with: --checkpoint {best_ckpt}")


def _latest_checkpoint(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    ckpts = sorted(d for d in os.listdir(directory) if d.startswith("checkpoint-"))
    return os.path.join(directory, ckpts[-1]) if ckpts else None


def run_phase(cfg: CFG, model, tokenizer, dataset: DatasetDict,
              n_epochs: int, out_dir: str, adapter_save: str,
              resume_from: Optional[str] = None,
              save_steps: Optional[int] = None,
              save_total_limit: Optional[int] = None):
    """
    Generic training phase runner.

    save_steps: if given, use steps-based checkpointing (Phase 2).
                eval_steps is set to the same value so load_best_model_at_end works.
                If None, checkpoint once per epoch (Phase 1).
    save_total_limit: max checkpoints to keep (None = keep all).
    """
    dtype = _compute_dtype()
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    if save_steps is not None:
        strategy   = "steps"
        extra_step = dict(save_steps=save_steps, eval_steps=save_steps)
    else:
        strategy   = "epoch"
        extra_step = {}

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.lr,
        weight_decay=cfg.w_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        eval_strategy=strategy,
        save_strategy=strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=save_total_limit,
        logging_steps=cfg.logging_steps,
        report_to="none",
        seed=cfg.seed,
        **extra_step,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Report per-checkpoint eval_loss and write eval_loss.txt files (Phase 2 only)
    if save_steps is not None:
        report_best_checkpoint(out_dir, trainer)

    os.makedirs(adapter_save, exist_ok=True)
    model.save_pretrained(adapter_save)
    print(f"\nAdapter saved → {adapter_save}")


# ─── Main training orchestrator ───────────────────────────────────────────────

def train(cfg: CFG, resume: bool = False, skip_phase1: bool = False):
    dtype = _compute_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | dtype: {dtype}")

    # Save tokenizer once (shared across phases)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    os.makedirs(cfg.tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(cfg.tokenizer_path)

    # ── Detect where to start / resume ────────────────────────────────────────
    phase2_done = os.path.isdir(cfg.phase2_adapter)
    phase1_done = os.path.isdir(cfg.phase1_adapter)

    if phase2_done:
        print("Both phases complete. Use --mode translate.")
        return

    # ── Phase 1: corpus.csv (1 epoch) ─────────────────────────────────────────
    if skip_phase1:
        if phase1_done:
            print(f"\n--skip-phase1: Phase 1 already complete — using existing adapter at {cfg.phase1_adapter}")
        else:
            print("\n--skip-phase1: Skipping Phase 1 — Phase 2 will start from base model with fresh LoRA.")
    elif not phase1_done:
        print("\n" + "═" * 62)
        print("PHASE 1 — corpus.csv  (1 epoch, MSLG↔SPA both directions)")
        print("═" * 62)

        df = preprocess_corpus(pd.read_csv(cfg.corpus_csv))
        records = pairs_to_records(df)
        print(f"Pairs: {len(df)}  →  examples (×2 directions): {len(records)}")

        dataset = build_dataset(records, tokenizer, cfg)
        print(f"Train: {len(dataset['train'])}  |  Val: {len(dataset['test'])}")

        model = load_model_new_lora(cfg, dtype)

        phase1_ckpt = _latest_checkpoint(cfg.phase1_dir) if resume else None
        if phase1_ckpt:
            print(f"Resuming Phase 1 from {phase1_ckpt}")

        run_phase(cfg, model, tokenizer, dataset,
                  n_epochs=1,
                  out_dir=cfg.phase1_dir,
                  adapter_save=cfg.phase1_adapter,
                  resume_from=phase1_ckpt)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        phase1_done = True
    else:
        print(f"\nPhase 1 already done — {cfg.phase1_adapter}")

    # ── Phase 2: MSLG_SPA_train.txt (5 epochs, 2 checkpoints/epoch) ─────────
    print("\n" + "═" * 62)
    print(f"PHASE 2 — MSLG_SPA_train.txt  "
          f"({cfg.phase2_epochs} epochs, {cfg.checkpoints_per_epoch} ckpt/epoch)")
    print("═" * 62)

    df = load_main_corpus(cfg.main_txt)
    records = pairs_to_records(df)
    print(f"Pairs: {len(df)}  →  examples (×2 directions): {len(records)}")

    dataset = build_dataset(records, tokenizer, cfg)
    n_train = len(dataset["train"])
    print(f"Train: {n_train}  |  Val: {len(dataset['test'])}")

    # Compute save_steps so exactly checkpoints_per_epoch saves occur each epoch.
    # steps_per_epoch = ceil(n_train / effective_batch_size)
    steps_per_epoch = math.ceil(n_train / (cfg.batch_size * cfg.grad_accumulation))
    save_steps      = max(1, steps_per_epoch // cfg.checkpoints_per_epoch)
    total_steps     = steps_per_epoch * cfg.phase2_epochs
    n_checkpoints   = total_steps // save_steps
    print(f"Steps/epoch: {steps_per_epoch}  |  "
          f"Save every {save_steps} steps  |  "
          f"~{n_checkpoints} checkpoints total")

    if phase1_done:
        model = load_model_from_adapter(cfg, cfg.phase1_adapter, dtype, trainable=True)
    else:
        # --skip-phase1 with no existing phase1 adapter: fresh LoRA on base model
        model = load_model_new_lora(cfg, dtype)

    phase2_ckpt = _latest_checkpoint(cfg.phase2_dir) if resume else None
    if phase2_ckpt:
        print(f"Resuming Phase 2 from {phase2_ckpt}")

    run_phase(cfg, model, tokenizer, dataset,
              n_epochs=cfg.phase2_epochs,
              out_dir=cfg.phase2_dir,
              adapter_save=cfg.phase2_adapter,
              resume_from=phase2_ckpt,
              save_steps=save_steps,
              save_total_limit=cfg.save_total_limit)


# ─── Checkpoint utilities ────────────────────────────────────────────────────

def _read_eval_loss(ckpt_dir: str) -> Optional[float]:
    txt = os.path.join(ckpt_dir, "eval_loss.txt")
    if not os.path.isfile(txt):
        return None
    with open(txt) as fh:
        for line in fh:
            if line.startswith("eval_loss="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    return None


def list_checkpoints(cfg: CFG) -> None:
    """Print all available Phase 2 checkpoints with eval_loss (when available)."""
    phase2_dir = cfg.phase2_dir
    final_path = cfg.phase2_adapter

    print(f"\nPhase 2 checkpoints — {phase2_dir}")
    print(f"  {'Name':<25}  {'eval_loss':>12}  Path")
    print(f"  {'─'*25}  {'─'*12}  {'─'*50}")

    ckpts = []
    if os.path.isdir(phase2_dir):
        ckpts = sorted(
            (d for d in os.listdir(phase2_dir) if d.startswith("checkpoint-")),
            key=lambda x: int(x.split("-")[-1]),
        )

    best_ckpt, best_loss = None, float("inf")
    ckpt_losses = {}
    for c in ckpts:
        loss = _read_eval_loss(os.path.join(phase2_dir, c))
        ckpt_losses[c] = loss
        if loss is not None and loss < best_loss:
            best_loss, best_ckpt = loss, c

    for c in ckpts:
        loss     = ckpt_losses[c]
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        marker   = "  ← best" if c == best_ckpt else ""
        print(f"  {c:<25}  {loss_str:>12}  {os.path.join(phase2_dir, c)}{marker}")

    if os.path.isdir(final_path):
        print(f"  {'final_adapter':<25}  {'(best ckpt)':>12}  {final_path}  ← default")

    if not ckpts and not os.path.isdir(final_path):
        print("  (none found — run --mode train first)")
        return

    print(f"\nUsage:")
    example = ckpts[0] if ckpts else "checkpoint-28"
    print(f"  --checkpoint {example}")
    print(f"  --checkpoint {os.path.join(phase2_dir, example)}")
    print(f"  (omit --checkpoint to use the final best adapter)")


def _resolve_checkpoint(cfg: CFG, checkpoint: Optional[str]) -> str:
    """
    Resolve the --checkpoint argument to an absolute directory path.

    Accepts:
      None                  → cfg.phase2_adapter  (best adapter, default)
      "final" / "best"      → cfg.phase2_adapter
      "checkpoint-56"       → <phase2_dir>/checkpoint-56
      absolute or relative path that exists  → used as-is
    """
    if checkpoint is None or checkpoint in ("final", "best", "final_adapter"):
        return cfg.phase2_adapter

    # Already an existing path
    if os.path.isdir(checkpoint):
        return checkpoint

    # Name relative to phase2_dir (e.g. "checkpoint-56")
    candidate = os.path.join(cfg.phase2_dir, checkpoint)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint}' not found.\n"
        f"Run --mode list-checkpoints to see available options."
    )


# ─── Inference / translation ─────────────────────────────────────────────────

def _parse_output(raw: str) -> dict:
    """Extract JSON object from model output; fallback to raw text."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": text}


def translate(cfg: CFG, inputs: list,
              checkpoint: Optional[str] = None,
              skip_lora: bool = False) -> list:
    """
    inputs: list of dicts with keys:
      'direction' — "mslg2spa" or "spa2mslg"
      'texto'     — source text to translate
    checkpoint: adapter/checkpoint path or name (None → final best adapter).
    skip_lora: if True, use the base model directly without any LoRA adapter.
    Returns list of parsed dicts with keys 'MSLG', 'SPA' (plus '_checkpoint').
    """
    dtype = _compute_dtype()

    tok_path  = cfg.tokenizer_path if os.path.isdir(cfg.tokenizer_path) else cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if skip_lora:
        print(f"Loading base model (no LoRA) from {cfg.model_name_or_path} …")
        model = _load_base(cfg, dtype)
        adapter_label = "base_model"
    else:
        adapter = _resolve_checkpoint(cfg, checkpoint)
        if not os.path.isdir(adapter):
            raise FileNotFoundError(
                f"No adapter found at '{adapter}'.\n"
                f"Run --mode train first, or use --mode list-checkpoints."
            )
        print(f"Loading model + adapter from {adapter} …")
        model = load_model_from_adapter(cfg, adapter, dtype, trainable=False)
        adapter_label = os.path.basename(adapter)

    model.eval()

    results = []
    for item in tqdm(inputs, desc="Translating"):
        direction = item["direction"]
        texto     = item["texto"]

        mslg = texto.strip().upper() if direction == "mslg2spa" else ""
        spa  = texto.strip()         if direction == "spa2mslg" else ""

        prompt = build_prompt(mslg, spa, direction)
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=cfg.max_length).to(model.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
                #length_penalty=cfg.length_penalty,
                #repetition_penalty=cfg.repetition_penalty,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = out_ids[0][prompt_len:]
        raw     = tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed  = _parse_output(raw)
        parsed["_direction"]  = direction
        parsed["_checkpoint"] = adapter_label
        results.append(parsed)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Bidirectional MSLG ↔ SPA translator (Qwen 3.5 9B + LoRA)"
    )
    p.add_argument("--mode",
                   choices=["train", "resume", "translate", "list-checkpoints"],
                   default="train",
                   help=("train: full curriculum | resume: continue where stopped | "
                         "translate: run inference | list-checkpoints: show saved ckpts"))
    p.add_argument("--texto",
                   help="Source text to translate (translate mode)")
    p.add_argument("--direction",
                   choices=["mslg2spa", "spa2mslg"],
                   default="mslg2spa",
                   help="Translation direction (default: mslg2spa)")
    p.add_argument("--checkpoint",
                   default=None,
                   help=("Checkpoint to use for translation. "
                         "Accepts: a checkpoint name (e.g. 'checkpoint-56'), "
                         "a full path, or 'final'/'best' for the best saved adapter. "
                         "Defaults to the final best adapter when omitted."))
    p.add_argument("--input-file", default=None, metavar="FILE",
                   help="Text file with one source text per line to translate. "
                        "Results written to --output-csv.")
    p.add_argument("--output-csv", default=None, metavar="CSV",
                   help="Output CSV for batch translation "
                        "(default: <input-file>_translations.csv).")
    p.add_argument("--corpus", default=CFG.corpus_csv,
                   help="Phase 1 corpus CSV (default: corpus.csv)")
    p.add_argument("--main",   default=CFG.main_txt,
                   help="Phase 2 curated corpus TXT (default: MSLG_SPA_train.txt)")
    p.add_argument("--out",    default=CFG.model_out,
                   help="Root output directory for adapters and checkpoints")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Skip Phase 1 (corpus.csv) and go directly to Phase 2. "
                        "If a Phase 1 adapter already exists it is reused; "
                        "otherwise Phase 2 starts from the base model with fresh LoRA.")
    p.add_argument("--skip-lora", action="store_true",
                   help="(translate mode) Use the base model directly, without loading "
                        "any LoRA adapter. Useful to compare base vs fine-tuned output.")
    return p.parse_args()


def main():
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face.")

    args = parse_args()
    cfg  = CFG()
    cfg.corpus_csv = args.corpus
    cfg.main_txt   = args.main
    cfg.model_out  = args.out

    if args.mode in ("train", "resume"):
        train(cfg, resume=(args.mode == "resume"), skip_phase1=args.skip_phase1)

    elif args.mode == "list-checkpoints":
        list_checkpoints(cfg)

    elif args.mode == "translate":
        if args.input_file:
            with open(args.input_file, encoding="utf-8") as fh:
                texts = [line.strip() for line in fh if line.strip()]
            if not texts:
                print("Error: input file is empty.")
                return
            inputs = [{"texto": t, "direction": args.direction} for t in texts]
            results = translate(cfg, inputs, checkpoint=args.checkpoint, skip_lora=args.skip_lora)
            rows = [
                {
                    "texto_fuente": item["texto"],
                    "direccion":    item["direction"],
                    "MSLG":         r.get("MSLG", ""),
                    "SPA":          r.get("SPA", ""),
                    "checkpoint":   r.get("_checkpoint", ""),
                }
                for item, r in zip(inputs, results)
            ]
            if args.output_csv:
                out_csv = args.output_csv
            else:
                stem = os.path.splitext(args.input_file)[0]
                out_csv = stem + "_translations.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
            print(f"\n{len(rows)} translations saved → {out_csv}")
        elif args.texto:
            results = translate(
                cfg,
                [{"texto": args.texto, "direction": args.direction}],
                checkpoint=args.checkpoint,
                skip_lora=args.skip_lora,
            )
            for r in results:
                print(json.dumps(r, ensure_ascii=False, indent=2))
        else:
            print("Error: --texto or --input-file is required in translate mode.")


if __name__ == "__main__":
    main()
