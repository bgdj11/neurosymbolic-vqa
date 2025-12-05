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
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self._precompute_text_embeddings()
        print("CLIP loaded and text embeddings cached!")
    

    def _precompute_text_embeddings(self):
        self.text_embeddings = {}
        
        color_prompts = {color: [
            f"a {color} object",
            f"a {color} colored object",
            f"{color} color",
        ] for color in self.COLORS}
        
        shape_prompts = {shape: [
            f"a {shape}",
            f"a 3D {shape}",
            f"a geometric {shape}",
        ] for shape in self.SHAPES}
        
        size_prompts = {
            "small": ["a small object", "a tiny object", "small size"],
            "large": ["a large object", "a big object", "large size"],
        }
        
        material_prompts = {
            "metal": ["a shiny metal object", "metallic surface", "reflective object"],
            "rubber": ["a matte rubber object", "rubber surface", "non-reflective object"],
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
        
        return results
    
    def extract_from_detections(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[Dict],
        attributes: List[str] = None
    ) -> List[Dict]:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        results = []
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                results.append(det)
                continue
            
            attrs = self.extract_attributes(image, bbox=tuple(bbox), attributes=attributes)
            
            enriched = det.copy()
            enriched["attributes"] = attrs
            results.append(enriched)
        
        return results