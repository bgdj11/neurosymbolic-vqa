import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")


class AttributeExtractor:
    COLORS = ["red", "blue", "green", "yellow", "cyan", "purple", "brown", "gray"]
    SHAPES = ["cube", "sphere", "cylinder"]
    SIZES = ["small", "large"]
    MATERIALS = ["metal", "rubber"]
    
    def __init__(self, model_name: str = "ViT-L/14", device: str = None):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self._precompute_text_embeddings()
        print("CLIP loaded and text embeddings cached!")
    

    def _precompute_text_embeddings(self):
        self.text_embeddings = {}
        
        color_prompts = {
            "red":    ["a red object", "a bright red 3D shape", "a red colored solid"],
            "blue":   ["a blue object", "a dark blue 3D shape, not cyan", "a deep blue colored solid"],
            "green":  ["a green object", "a green 3D shape", "a green colored solid"],
            "yellow": ["a bright lemon yellow object, not orange and not brown", "a vivid yellow 3D shape", "a pure yellow colored solid, not gold"],
            "cyan":   ["a cyan object", "a light cyan teal 3D shape, not blue", "a cyan blue-green colored solid"],
            "purple": ["a purple object", "a purple violet 3D shape, not blue", "a purple colored solid"],
            "brown":  ["a dark brown object, not yellow and not orange", "a chocolate brown 3D shape, darker than yellow", "a warm dark brown solid, like dark wood or chocolate"],
            "gray":   ["a gray object", "a neutral gray 3D shape, not brown", "a gray silver colored solid"],
        }

        shape_prompts = {
            "cube":     ["a cube", "a 3D cube", "a cubic box shape", "a rectangular box object", "a block cube"],
            "sphere":   ["a sphere", "a 3D sphere", "a round ball shape", "a spherical object", "a smooth ball"],
            "cylinder": ["a cylinder", "a 3D cylinder", "a cylindrical tube shape", "a round cylinder object", "a cylindrical solid"],
        }

        size_prompts = {
            "small": ["a small object", "a tiny object", "a small 3D shape"],
            "large": ["a large object", "a big object", "a large 3D shape"],
        }

        material_prompts = {
            "metal": [
                "a shiny metallic object with specular highlights",
                "a glossy metal surface with bright reflections",
                "a smooth gleaming metallic 3D object",
                "a reflective metal shape with light glare",
            ],
            "rubber": [
                "a matte rubber object with no shine",
                "a dull non-reflective rubber surface",
                "a flat matte 3D object without any highlights",
                "a rough rubber shape with uniform diffuse color",
            ],
        }
        
        self.text_embeddings["color"] = self._encode_attribute_prompts(color_prompts, self.COLORS)
        self.text_embeddings["shape"] = self._encode_attribute_prompts(shape_prompts, self.SHAPES)
        self.text_embeddings["size"] = self._encode_attribute_prompts(size_prompts, self.SIZES)
        self.text_embeddings["material"] = self._encode_attribute_prompts(material_prompts, self.MATERIALS)
    
    def _encode_attribute_prompts(self, prompts_dict: Dict[str, List[str]], labels: List[str]) -> Dict:
        embeddings = []
        
        for label in labels:
            label_prompts = prompts_dict[label]
            
            tokens = clip.tokenize(label_prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
            
            embeddings.append(text_features)
        
        return {
            "embeddings": torch.stack(embeddings),
            "labels": labels
        }
    
    def _encode_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.squeeze(0)

    def _classify_attribute(self, image_embedding: torch.Tensor, attribute: str) -> Tuple[str, float]:
        attr_data = self.text_embeddings[attribute]
        text_embeddings = attr_data["embeddings"]
        labels = attr_data["labels"]
        
        similarities = (image_embedding @ text_embeddings.T).cpu().numpy()
        
        best_idx = similarities.argmax()
        best_label = labels[best_idx]
        confidence = float(similarities[best_idx])
        
        return best_label, confidence
    
    def extract_attributes(self, 
        image: Union[Image.Image, np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        attributes: List[str] = None
    ) -> Dict[str, Dict[str, float]]:

        if attributes is None:
            attributes = ["color", "shape", "size", "material"]
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            image = image.crop((x1, y1, x2, y2))
        
        image_embedding = self._encode_image(image)

        results = {}
        for attr in attributes:
            if attr not in self.text_embeddings:
                continue

            predicted, confidence = self._classify_attribute(image_embedding, attr)

            attr_data = self.text_embeddings[attr]
            all_similarities = (image_embedding @ attr_data["embeddings"].T).cpu().numpy()
            all_scores = {label: float(score) for label, score in zip(attr_data["labels"], all_similarities)}

            results[attr] = {
                "value": predicted,
                "confidence": confidence,
                "all_scores": all_scores
            }

        if "color" in results:
            results["color"]["value"] = self._hsv_color_correction(image, results["color"]["value"])

        return results

    def _hsv_color_correction(self, image: Image.Image, clip_color: str) -> str:
        if clip_color not in ("brown", "yellow", "gray", "purple"):
            return clip_color

        import colorsys
        arr = np.array(image.resize((32, 32))).astype(float) / 255.0
        r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360

        if clip_color in ("brown", "yellow"):
            # Hue je pouzdan diskriminator u CLEVR:
            # Yellow: H=42-75° (lemon/golden), Brown: H=10-42° (topla zemlja)
            # V prag se ne koristi — CLEVR cropovi su osvetljeni 0.55-0.70,
            # V≥0.70 threshold je bio previsok i blokirao korekciju zute.
            if h_deg >= 42:
                return "yellow"
            elif v < 0.55:
                return "brown"

        elif clip_color in ("gray", "purple"):
            # Gray: very low saturation; Purple: clearly colored
            if s < 0.15:
                return "gray"
            if s > 0.22:
                return "purple"

        return clip_color


    def extract_from_detections(
        self,
        image: Image.Image,
        detections: List[Dict],
        attributes: Optional[List[str]] = None,
        use_heuristics: bool = True,
    ) -> List[Dict]:

        pil_image = image.convert("RGB")

        wanted = [a.lower() for a in (attributes or ["color", "shape", "size", "material"])]
        clip_wanted = [a for a in wanted if a in ("color", "shape", "material")]
        want_size = "size" in wanted

        img_area = float(pil_image.width * pil_image.height)

        results = []
        for i, det in enumerate(detections):
            enriched = det.copy()
            enriched["attributes"] = {}

            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                results.append(enriched)
                continue

            bbox_t = tuple(bbox)

            if clip_wanted:
                clip_attrs = self.extract_attributes(pil_image, bbox=bbox_t, attributes=clip_wanted)
                enriched["attributes"].update(clip_attrs)

            if want_size:
                if use_heuristics:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    cy = (y1 + y2) / 2.0
                    y_frac = min(1.0, max(0.0, cy / pil_image.height))
                    # Depth-adjusted threshold: farther objects (high in image, low y_frac)
                    # appear smaller → lower threshold to still classify them as "large"
                    adj_threshold = (0.018 + 0.026 * y_frac) * img_area
                    size_val = "large" if a >= adj_threshold else "small"
                    conf = min(0.95, 0.5 + abs(a - adj_threshold) / (adj_threshold + 1.0) * 0.5)
                    enriched["attributes"]["size"] = {"value": size_val, "confidence": float(conf)}
                else:
                    enriched["attributes"]["size"] = self.extract_attributes(
                        pil_image, bbox=bbox_t, attributes=["size"]
                    )["size"]

            results.append(enriched)

        return results


