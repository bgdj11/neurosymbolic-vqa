from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OBJECT_COLORS = {
    'red':    '#FF4444',
    'blue':   '#4499FF',
    'green':  '#44CC44',
    'yellow': '#FFCC00',
    'cyan':   '#00CCDD',
    'purple': '#AA44CC',
    'brown':  '#AA6633',
    'gray':   '#999999',
}
_FALLBACK_COLOR = '#FFFFFF'
_GRAPH_BG  = '#16213e'
_GRAPH_FG  = '#0f0f23'
_AXIS_COLOR = '#888888'


def _load_image(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert('RGB'))
    if isinstance(image, Image.Image):
        return np.array(image.convert('RGB'))
    return np.asarray(image)


def _obj_color(obj: Dict) -> str:
    return OBJECT_COLORS.get(obj.get('color'), _FALLBACK_COLOR)


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Image panel

def _draw_image_panel(
    ax: plt.Axes,
    img: np.ndarray,
    scene_graph: Dict,
    title: str = '',
    marker: str = 'o',
) -> None:
    ax.imshow(img)
    ax.axis('off')
    objects = scene_graph.get('objects', [])
    ax.set_title(title or f'{len(objects)} objects', fontsize=12, pad=8)

    for i, obj in enumerate(objects):
        color = _obj_color(obj)
        coords = obj.get('pixel_coords', [0, 0, 0])
        cx = _safe_float(coords[0] if len(coords) > 0 else 0)
        cy = _safe_float(coords[1] if len(coords) > 1 else 0)

        # Bbox (only detected objects have this)
        bbox = obj.get('bbox')
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [_safe_float(v) for v in bbox]
            # Filled semi-transparent rectangle
            ax.add_patch(mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle='round,pad=2', linewidth=0,
                facecolor=color, alpha=0.18, zorder=3,
            ))
            # Solid border
            ax.add_patch(mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle='round,pad=2', linewidth=2,
                edgecolor=color, facecolor='none', alpha=0.9, zorder=4,
            ))
            # Label top-left of box
            ax.text(
                x1 + 4, y1 - 4,
                f"{obj.get('color', '?')} {obj.get('shape', '?')}",
                ha='left', va='bottom', fontsize=7, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, lw=0),
                zorder=8,
            )

        # Numbered circle marker
        ax.plot(cx, cy, marker, markersize=18, color=color,
                markeredgecolor='white', markeredgewidth=2, zorder=6)
        ax.text(cx, cy, str(i), ha='center', va='center',
                fontsize=8, weight='bold', color='white', zorder=7)


# Spatial scatter panel

def _draw_spatial_scatter(ax: plt.Axes, scene_graph: Dict, title: str = 'Spatial Layout') -> None:

    ax.set_facecolor(_GRAPH_BG)
    ax.set_title(title, fontsize=11, color='white', pad=8)
    ax.set_xlabel('← left          right →', color='#aaaacc', fontsize=9)
    ax.set_ylabel('← behind        front →', color='#aaaacc', fontsize=9)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    objects = scene_graph.get('objects', [])
    if not objects:
        ax.text(0.5, 0.5, 'No objects', ha='center', va='center',
                color='gray', fontsize=11, transform=ax.transAxes)
        return

    # Extract horizontal position and depth from pixel_coords
    xs     = [_safe_float(o.get('pixel_coords', [0])[0]) for o in objects]
    depths = [_safe_float(o.get('pixel_coords', [0, 0, 0])[2] if len(o.get('pixel_coords', [])) > 2 else 0)
              for o in objects]

    x_min, x_max = min(xs), max(xs)
    d_min, d_max = min(depths), max(depths)
    x_range = x_max - x_min or 1.0
    d_range = d_max - d_min or 1.0

    xs_n = [(x - x_min) / x_range for x in xs]
    ds_n = [(d - d_min) / d_range for d in depths]

    # Grid
    ax.grid(True, alpha=0.15, color='#5555aa', linestyle='--')
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.22, 1.18)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['left', 'center', 'right'], color='#aaaaaa', fontsize=8)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['behind', 'mid', 'front'], color='#aaaaaa', fontsize=8)

    for i, (obj, xn, dn) in enumerate(zip(objects, xs_n, ds_n)):
        color = _obj_color(obj)

        # White ring + colored fill
        ax.scatter(xn, dn, s=700, color='white',  zorder=3)
        ax.scatter(xn, dn, s=580, color=color,    zorder=4)
        ax.text(xn, dn, str(i), ha='center', va='center',
                fontsize=9, weight='bold', color='white', zorder=5)

        # Short label below node
        label = f"{obj.get('color', '?')} {obj.get('shape', '?')}"
        ax.annotate(
            label, xy=(xn, dn),
            xytext=(0, -22), textcoords='offset points',
            ha='center', va='top', fontsize=6.5, color='#dddddd',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a3e', alpha=0.85, lw=0),
            zorder=6,
        )

# Public API

def render_scene_graph(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    scene_graph: Dict,
    title: str = 'Scene Graph',
    show_graph: bool = True,
) -> plt.Figure:

    img = _load_image(image)
    n = len(scene_graph.get('objects', []))

    if show_graph:
        fig = plt.figure(figsize=(17, 7))
        ax_img   = fig.add_axes([0.02, 0.05, 0.50, 0.88])
        ax_graph = fig.add_axes([0.55, 0.05, 0.35, 0.88])
        ax_graph.set_facecolor(_GRAPH_BG)
    else:
        fig, ax_img = plt.subplots(figsize=(10, 7))
        ax_graph = None

    _draw_image_panel(ax_img, img, scene_graph,
                      title=f'{title}  ({n} objects)')

    if ax_graph is not None:
        _draw_spatial_scatter(ax_graph, scene_graph)

    fig.suptitle(title, fontsize=13, y=0.99)
    return fig


def compare_scene_graphs(
    image: Union[str, Path, 'Image.Image', np.ndarray],
    gt_graph: Dict,
    detected_graph: Dict,
    figsize: Tuple[int, int] = (18, 7),
) -> plt.Figure:
    img = _load_image(image)
    gt_n  = len(gt_graph.get('objects', []))
    det_n = len(detected_graph.get('objects', []))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _draw_image_panel(axes[0], img, gt_graph,
                      title=f'Ground Truth  ({gt_n} objects)', marker='o')
    _draw_image_panel(axes[1], img, detected_graph,
                      title=f'Detected  ({det_n} objects)', marker='s')

    plt.tight_layout(pad=1.5)
    return fig


def get_object_description(obj: Dict) -> str:
    parts = [obj[a] for a in ('size', 'color', 'material', 'shape') if obj.get(a)]
    return ' '.join(parts) if parts else 'unknown object'
