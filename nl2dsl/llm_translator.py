import json
import re
from typing import List, Optional, Dict

from nl2dsl.prepare_data import linear_to_program
from dsl.validator import DSLValidator

FEW_SHOT_EXAMPLES = [
    # count
    ("How many cubes are there?",
     "scene | count"),
    ("How many red things are there?",
     "scene | filter_color red | count"),
    ("How many large metal cubes are there?",
     "scene | filter_size large | filter_material metal | filter_shape cube | count"),
    ("How many objects are either red or cubes?",
     "scene | filter_color red:0 | scene | filter_shape cube:0 | union:1,3 | count"),
    # exist
    ("Are there any red spheres?",
     "scene | filter_color red | filter_shape sphere | exist"),
    ("Is there a large metallic object?",
     "scene | filter_size large | filter_material metal | exist"),
    # query
    ("What color is the cube?",
     "scene | filter_shape cube | unique | query_color"),
    ("What size is the metallic sphere?",
     "scene | filter_material metal | filter_shape sphere | unique | query_size"),
    ("What material is the large red object?",
     "scene | filter_size large | filter_color red | unique | query_material"),
    # relate
    ("How many objects are to the left of the red cube?",
     "scene | filter_color red | filter_shape cube | unique | relate left | count"),
    ("What color is the object behind the small sphere?",
     "scene | filter_size small | filter_shape sphere | unique | relate behind | unique | query_color"),
    ("Are there any cylinders to the right of the blue object?",
     "scene | filter_color blue | unique | relate right | filter_shape cylinder | exist"),
    ("There is a small gray block; are there any spheres to the left of it?",
     "scene | filter_size small | filter_color gray | filter_shape cube | unique | relate left | filter_shape sphere | exist"),
    # compare
    ("Are there more cubes than spheres?",
     "scene | filter_shape cube:0 | count:1 | scene | filter_shape sphere:0 | count:3 | greater_than:2,4"),
    ("Is the number of red things greater than the number of blue things?",
     "scene | filter_color red:0 | count:1 | scene | filter_color blue:0 | count:3 | greater_than:2,4"),
    ("Are there the same number of large things as red things?",
     "scene | filter_size large:0 | count:1 | scene | filter_color red:0 | count:3 | equal_integer:2,4"),
    # same-attribute (use same_shape / same_size / same_color / same_material)
    ("Are there any other things that are the same shape as the big metallic object?",
     "scene | filter_size large | filter_material metal | unique | same_shape | exist"),
    ("How many other objects are the same size as the purple shiny object?",
     "scene | filter_color purple | filter_material metal | unique | same_size | count"),
    ("Is there anything else that has the same color as the large shiny cube?",
     "scene | filter_size large | filter_material metal | filter_shape cube | unique | same_color | exist"),
    ("How many other things are the same material as the big gray cylinder?",
     "scene | filter_size large | filter_color gray | filter_shape cylinder | unique | same_material | count"),
    ("Is there a big brown object of the same shape as the green thing?",
     "scene | filter_color green | unique | same_shape | filter_color brown | filter_size large | exist"),
    # boolean
    ("Is the cube the same color as the sphere?",
     "scene | filter_shape cube | unique | query_color:2 | scene | filter_shape sphere | unique | query_color:5 | equal_color:3,6"),
    ("Does the small object have the same material as the large cylinder?",
     "scene | filter_size small | unique | query_material:2 | scene | filter_size large | filter_shape cylinder | unique | query_material:6 | equal_material:3,7"),
    # integer compare
    ("Are there fewer red things than blue things?",
     "scene | filter_color red:0 | count:1 | scene | filter_color blue:0 | count:3 | less_than:2,4"),
]

SYSTEM_PROMPT = """You are a semantic parser. Convert natural language questions about 3D scenes into a DSL (Domain Specific Language).

CRITICAL OUTPUT RULES:
- Output EXACTLY one line: the DSL string only
- NO explanations, NO comments, NO extra text before or after
- If unsure, output your best guess DSL string anyway

DSL rules:
- Programs are sequences of operations separated by " | "
- Steps are numbered 0, 1, 2, ... from left to right
- scene: returns all objects (always first)
- filter_color <color>: keep objects with color. Colors: red, blue, green, yellow, purple, gray, brown, cyan. ALWAYS write "filter_color red", never just "red"
- filter_shape <shape>: keep objects with shape. Shapes: cube, sphere, cylinder. "block" = cube. ALWAYS write "filter_shape cube", never just "cube"
- filter_size <size>: keep objects with size. Sizes: large, small. "big" = large. ALWAYS write "filter_size large", never just "large"
- filter_material <mat>: keep objects with material. Materials: metal, rubber. "shiny" = metal, "matte" = rubber. ALWAYS write "filter_material metal", never just "metal"
- unique: select the single object (use when exactly one object matches)
- count: count objects → integer
- exist: do any objects match → yes/no
- query_color / query_shape / query_size / query_material: get attribute of unique object (NO arguments)
- same_shape / same_size / same_color / same_material: return all OTHER objects with the same attribute as current unique object (NO arguments). Use for "same X as Y" questions.
- relate <dir>: objects spatially related to current unique object. Directions: left, right, front, behind
- greater_than, less_than, equal_integer: compare two integers
- equal_color, equal_shape, equal_size, equal_material: compare two attribute values
- union / intersect: set operations on two object sets

Step reference syntax (:i and :i,j):
- Use :i ONLY to refer to the output of step number i (0-indexed)
- Use op:i when the input is NOT the immediately previous step
- Use op:i,j for binary operations taking two step outputs as inputs
- NEVER use :value (like :metal or :red) — that is wrong. :i means step index only
- Example: query_shape:3 means "query_shape using output of step 3", NOT "query shape=3"
- Example: filter_shape:4 means "filter using the shape value produced at step 4"

"Same X as Y" pattern (IMPORTANT):
- Find Y → unique → same_X → ...
- Example "same shape as the red cube": scene | filter_color red | filter_shape cube | unique | same_shape | ..."""


def build_prompt(question: str) -> str:
    examples = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in FEW_SHOT_EXAMPLES
    )
    return f"{SYSTEM_PROMPT}\n\nExamples:\n{examples}\n\nQ: {question}\nA:"


class LLMTranslator:

    def __init__(
        self,
        model: str = 'llama3',
        temperature: float = 0.0,
        num_predict: int = 128,
    ):
        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            raise ImportError("pip install ollama")

        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.validator = DSLValidator()

    def _call(self, question: str) -> str:
        prompt = build_prompt(question)
        response = self._ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.num_predict,
            },
        )
        return response['response'].strip()

    def predict_linear(self, question: str) -> List[str]:
        """Returns list with one candidate (LLM is not beam search)."""
        try:
            return [self._call(question)]
        except Exception:
            return []

    def predict_program(self, question: str) -> Optional[List[Dict]]:
        raw = self._call(question)
        # Strip any explanation text — take first line that looks like DSL
        for line in raw.splitlines():
            line = line.strip()
            if '|' in line or line.startswith('scene'):
                try:
                    program = linear_to_program(line)
                    result = self.validator.validate(program)
                    if result.is_valid:
                        return program
                except Exception:
                    continue
        return None

    def predict_batch(
        self,
        questions: List[str],
        batch_size: int = 1,  # Ollama is sequential
    ) -> List[Optional[List[Dict]]]:
        programs = []
        for i, q in enumerate(questions):
            programs.append(self.predict_program(q))
            if (i + 1) % 50 == 0:
                print(f'  LLM: {i+1}/{len(questions)}')
        return programs

    def diagnose(self, question: str) -> Dict:
        raw = self._call(question)
        candidates = [l.strip() for l in raw.splitlines() if l.strip()]
        results = []
        for i, candidate in enumerate(candidates):
            try:
                program = linear_to_program(candidate)
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
                    'beam': i, 'linear': candidate,
                    'valid': False,
                    'errors': [{'step': -1, 'type': 'parse_error', 'msg': str(e)}],
                    'program': None,
                })
        return {
            'question': question,
            'raw_output': raw,
            'beams': results,
            'best_valid': next((r for r in results if r['valid']), None),
        }
