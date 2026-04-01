from typing import List, Optional, Dict
from pathlib import Path

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from nl2dsl.prepare_data import linear_to_program
from dsl.validator import DSLValidator


class NL2DSLModel:

    def __init__(
        self,
        model_path: str = 'models/t5-nl2dsl-final',
        device: str = None,
        num_beams: int = 5,
        max_length: int = 128,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_beams = num_beams
        self.max_length = max_length
        self.validator = DSLValidator()

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        self.model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict_linear(self, question: str) -> List[str]:
        #Returns top-k beam candidates as raw linear strings.
        input_text = f'translate to DSL: {question}'
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=64,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                max_length=self.max_length,
                early_stopping=True,
            )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def predict_program(self, question: str) -> Optional[List[Dict]]:
        # Returns the first valid DSL program from beam candidates.

        candidates = self.predict_linear(question)
        for candidate in candidates:
            try:
                program = linear_to_program(candidate.strip())
                result = self.validator.validate(program)
                if result.is_valid:
                    return program
            except Exception:
                continue
        return None

    def predict_batch(
        self,
        questions: List[str],
        batch_size: int = 32,
    ) -> List[Optional[List[Dict]]]:
        # Batch inference — much faster than calling predict_program in a loop.
        programs = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            input_texts = [f'translate to DSL: {q}' for q in batch]

            inputs = self.tokenizer(
                input_texts,
                return_tensors='pt',
                max_length=64,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    max_length=self.max_length,
                    early_stopping=True,
                )

            # outputs shape: (batch * num_beams, seq_len)
            for j in range(len(batch)):
                beam_outputs = outputs[j * self.num_beams:(j + 1) * self.num_beams]
                program = None
                for beam in beam_outputs:
                    candidate = self.tokenizer.decode(beam, skip_special_tokens=True)
                    try:
                        prog = linear_to_program(candidate.strip())
                        result = self.validator.validate(prog)
                        if result.is_valid:
                            program = prog
                            break
                    except Exception:
                        continue
                programs.append(program)

        return programs

    def diagnose(self, question: str) -> Dict:
        """
        Returns detailed info about all beam candidates — useful for debugging.
        """
        candidates = self.predict_linear(question)
        results = []
        for i, candidate in enumerate(candidates):
            try:
                program = linear_to_program(candidate.strip())
                validation = self.validator.validate(program)
                results.append({
                    'beam': i,
                    'linear': candidate,
                    'valid': validation.is_valid,
                    'errors': [
                        {'step': e.step_idx, 'type': e.error_type, 'msg': e.message}
                        for e in validation.errors
                    ],
                    'program': program if validation.is_valid else None,
                })
            except Exception as e:
                results.append({
                    'beam': i,
                    'linear': candidate,
                    'valid': False,
                    'errors': [{'step': -1, 'type': 'parse_error', 'msg': str(e)}],
                    'program': None,
                })
        return {
            'question': question,
            'beams': results,
            'best_valid': next((r for r in results if r['valid']), None),
        }
