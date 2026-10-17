"""Fine-tune T5 on paraphrases mixed with CLEVR questions."""

BASE_MODEL_DIR = '/kaggle/input/datasets/bogdandjukic/t5-nl2dsl-final/t5-nl2dsl-final'
TRAIN_TSV      = '/kaggle/input/datasets/bogdandjukic/para-mix/final_mix_train.tsv'
VAL_TSV        = '/kaggle/input/datasets/bogdandjukic/para-mix/final_mix_val.tsv'
OUTPUT_DIR     = '/kaggle/working/t5-nl2dsl-para'

MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 128
BATCH_SIZE     = 16
EPOCHS         = 4
LR             = 5e-5
WEIGHT_DECAY   = 0.01
WARMUP_STEPS   = 200
EVAL_STEPS     = 500
SAVE_STEPS     = 500
LOGGING_STEPS  = 100

import os, csv
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


class DSLDataset(Dataset):
    def __init__(self, tsv_path, tokenizer):
        self.samples = []
        with open(tsv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                self.samples.append((row['question'], row['program']))
        self.tok = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, p = self.samples[idx]
        inp = self.tok(f'translate to DSL: {q}',
                       max_length=MAX_INPUT_LEN, padding='max_length', truncation=True)
        tgt = self.tok(p,
                       max_length=MAX_TARGET_LEN, padding='max_length', truncation=True)
        labels = np.array(tgt['input_ids'], dtype=np.int64)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            'input_ids':      np.array(inp['input_ids'],      dtype=np.int64),
            'attention_mask': np.array(inp['attention_mask'], dtype=np.int64),
            'labels':         labels,
        }


def compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            [[t for t in lbl if t != -100] for lbl in labels],
            skip_special_tokens=True
        )
        em = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
        return {'exact_match': round(em / len(decoded_preds), 4)}
    return compute_metrics


def main():
    print(f'Model: {BASE_MODEL_DIR}')
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_DIR)
    model     = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_DIR)
    print(f'Parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    train_ds = DSLDataset(TRAIN_TSV, tokenizer)
    val_ds   = DSLDataset(VAL_TSV,   tokenizer)
    print(f'Train: {len(train_ds):,}   Val: {len(val_ds):,}')

    args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = WARMUP_STEPS,
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LEN,
        generation_num_beams        = 1,
        eval_strategy               = 'steps',
        eval_steps                  = EVAL_STEPS,
        save_strategy               = 'steps',
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'exact_match',
        greater_is_better           = True,
        logging_steps               = LOGGING_STEPS,
        fp16                        = torch.cuda.is_available(),
        report_to                   = 'none',
    )

    trainer = Seq2SeqTrainer(
        model             = model,
        args              = args,
        train_dataset     = train_ds,
        eval_dataset      = val_ds,
        processing_class  = tokenizer,
        data_collator     = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics   = compute_metrics_fn(tokenizer),
    )

    print('Pocinjem fine-tuning...')
    trainer.train()

    best = os.path.join(OUTPUT_DIR, 'best')
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f'Najbolji model sacuvan: {best}')
    print('Preuzmi /kaggle/working/t5-nl2dsl-para/ iz Output panela.')


if __name__ == '__main__':
    main()
