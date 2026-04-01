DRIVE_BASE = '/content/drive/MyDrive/neurosymbolic-vqa'
TRAIN_TSV = f'{DRIVE_BASE}/datasets/nl2dsl/clevr_train.tsv'
VAL_TSV = f'{DRIVE_BASE}/datasets/nl2dsl/clevr_val.tsv'
CHECKPOINT_DIR = f'{DRIVE_BASE}/checkpoints/t5-nl2dsl'
FINAL_MODEL_DIR = f'{DRIVE_BASE}/models/t5-nl2dsl-final'

MODEL_NAME = 't5-small'
MAX_INPUT_LEN = 64    # max question token length
MAX_TARGET_LEN = 128   # max program token length

BATCH_SIZE = 16 # 16 for free T4, 32 for A100/Colab Pro
EPOCHS = 3
LR = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
SAVE_STEPS = 2000
EVAL_STEPS = 2000
LOGGING_STEPS = 200

RESUME_FROM_CHECKPOINT = False # set True to resume from last checkpoint

import os
import csv
import signal
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,

)

class DSLDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_input_len, max_target_len):
        self.samples = []
        with open(tsv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                self.samples.append((row['question'], row['program']))

        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, program = self.samples[idx]

        input_enc = self.tokenizer(
            f'translate to DSL: {question}',
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
        )
        target_enc = self.tokenizer(
            program,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
        )

        import numpy as np
        labels = np.array(target_enc['input_ids'], dtype=np.int64)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      np.array(input_enc['input_ids'],      dtype=np.int64),
            'attention_mask': np.array(input_enc['attention_mask'], dtype=np.int64),
            'labels':         labels,
        }


# METRICS 
def make_compute_metrics(tokenizer):
    
    # valid_program_rate = how many decoded programs pass DSL round-trip check.
    try:
        import sys
        sys.path.insert(0, '/content/drive/MyDrive/neurosymbolic-vqa')
        from nl2dsl.prepare_data import linear_to_program, program_to_linear
        from dsl.validator import DSLValidator
        validator = DSLValidator()
        has_validator = True
    except ImportError:
        has_validator = False

    def compute_metrics(eval_pred):
        import numpy as np
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = [
            [t for t in label if t != -100]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
        exact_match = exact_matches / len(decoded_preds)

        if has_validator:
            valid = 0
            for pred in decoded_preds:
                try:
                    program = linear_to_program(pred.strip())
                    result = validator.validate(program)
                    if result.is_valid:
                        valid += 1
                except Exception:
                    pass
            valid_program_rate = valid / len(decoded_preds)
        else:
            valid_program_rate = 0.0

        return {
            'exact_match': round(exact_match, 4),
            'valid_program_rate': round(valid_program_rate, 4),
        }

    return compute_metrics


# GRACEFUL INTERRUPT 
_trainer_ref = None

def _handle_sigint(sig, frame):
    print('\nInterrupt received — saving checkpoint before exit...')
    if _trainer_ref is not None:
        try:
            _trainer_ref.save_model(f'{CHECKPOINT_DIR}/interrupt-checkpoint')
            print(f'Checkpoint saved to {CHECKPOINT_DIR}/interrupt-checkpoint')
        except Exception as e:
            print(f'Could not save: {e}')
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)


# MAIN 

def main():
    global _trainer_ref

    # Mount Google Drive 
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print('Google Drive mounted.')
    except ImportError:
        print('Not running in Colab — skipping Drive mount.')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    # Load tokenizer and model
    print(f'Loading {MODEL_NAME}...')
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params / 1e6:.1f}M')

    # Datasets 
    print('Loading datasets...')
    train_dataset = DSLDataset(TRAIN_TSV, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    full_val = DSLDataset(VAL_TSV, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    from torch.utils.data import Subset
    val_dataset = Subset(full_val, range(2000))
    print(f'  Train: {len(train_dataset):,} samples')
    print(f'  Val:   {len(val_dataset):,} samples (subset for fast eval)')

    # Training arguments 
    training_args = Seq2SeqTrainingArguments(
        output_dir = CHECKPOINT_DIR,
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate = LR,
        weight_decay = WEIGHT_DECAY,
        warmup_steps = WARMUP_STEPS,
        predict_with_generate = True,
        generation_max_length = MAX_TARGET_LEN,
        generation_num_beams = 1, # greedy during training eval (faster)
        eval_strategy = 'steps',
        eval_steps = EVAL_STEPS,
        save_strategy = 'steps',
        save_steps = SAVE_STEPS,
        save_total_limit = 3, # keep last 3 checkpoints
        load_best_model_at_end = False,
        logging_steps = LOGGING_STEPS,
        logging_dir = f'{CHECKPOINT_DIR}/logs',
        fp16 = torch.cuda.is_available(),
        dataloader_num_workers = 2,
        report_to = 'none',
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics = make_compute_metrics(tokenizer),
    )
    _trainer_ref = trainer

    # Train
    resume = RESUME_FROM_CHECKPOINT
    if resume:
        checkpoints = sorted(
            [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('checkpoint-')],
            key=lambda x: int(x.split('-')[-1])
        )
        if checkpoints:
            last_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f'Resuming from {last_ckpt}')
        else:
            print('No checkpoint found — starting from scratch.')
            resume = False

    print('Starting training...')
    trainer.train(resume_from_checkpoint=last_ckpt if resume else None)

    # Save final model
    print(f'Saving final model to {FINAL_MODEL_DIR}')
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print('Done.')


if __name__ == '__main__':
    main()
