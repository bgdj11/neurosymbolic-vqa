import json
from pathlib import Path

import numpy as np
import pytest

from dsl.scene_graph import SceneGraph
from dsl.executor import Executor
from scene_graph.builder import SceneGraphBuilder, save_scene_graph, load_scene_graph

CLEVR_SAMPLE_IMAGE = Path('datasets/clevr/images/val/CLEVR_val_000000.png')
CLEVR_VAL = Path('datasets/converted/clevr/clevr_val.jsonl')

REQUIRED_RELATIONS = {'left', 'right', 'front', 'behind'}
REQUIRED_OBJECT_ATTRS = {'color', 'shape', 'size', 'material'}


# Fixture: module-scoped builder (loads models once per test session)

@pytest.fixture(scope='module')
def builder():
    return SceneGraphBuilder()


@pytest.fixture(scope='module')
def sample_graph(builder):
    """Build a scene graph from the first CLEVR val image (requires models)."""
    if not CLEVR_SAMPLE_IMAGE.exists():
        pytest.skip(f"CLEVR image not found: {CLEVR_SAMPLE_IMAGE}")
    return builder.build(str(CLEVR_SAMPLE_IMAGE))

# Tests that do NOT require model weights

def test_builder_instantiates_without_args():
    b = SceneGraphBuilder()
    assert b is not None


def test_builder_accepts_relation_config():
    from scene_graph.relations import RelationConfig
    cfg = RelationConfig(horizontal_threshold=0.08)
    b = SceneGraphBuilder(relation_config=cfg)
    assert b.relation_config.horizontal_threshold == 0.08


def test_save_and_load_roundtrip(tmp_path):
    graph = {
        'objects': [
            {'color': 'red', 'shape': 'cube', 'size': 'small', 'material': 'rubber',
             'pixel_coords': [100, 200, 0.5]},
        ],
        'relationships': {
            'left': [[]], 'right': [[]], 'front': [[]], 'behind': [[]],
        }
    }
    out = tmp_path / 'sg.json'
    save_scene_graph(graph, str(out))
    loaded = load_scene_graph(str(out))
    assert loaded['objects'][0]['color'] == 'red'
    assert set(loaded['relationships'].keys()) == REQUIRED_RELATIONS


# Tests that require blank image (no GPU model results needed)

@pytest.mark.slow
def test_empty_scene_returns_valid_structure(builder):
    """A blank white image should return an empty but structurally valid scene graph."""
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    sg = builder.build(blank)
    assert 'objects' in sg
    assert 'relationships' in sg
    assert set(sg['relationships'].keys()) == REQUIRED_RELATIONS
    assert isinstance(sg['objects'], list)
    for rel_list in sg['relationships'].values():
        assert isinstance(rel_list, list)
        assert len(rel_list) == len(sg['objects'])


# Tests that require the full perception pipeline (models needed)

@pytest.mark.slow
def test_output_has_required_keys(sample_graph):
    assert 'objects' in sample_graph
    assert 'relationships' in sample_graph
    assert set(sample_graph['relationships'].keys()) == REQUIRED_RELATIONS


@pytest.mark.slow
def test_relationships_are_correct_length(sample_graph):
    n = len(sample_graph['objects'])
    for rel in REQUIRED_RELATIONS:
        assert len(sample_graph['relationships'][rel]) == n, (
            f"Relation '{rel}' list length {len(sample_graph['relationships'][rel])} "
            f"!= number of objects {n}"
        )


@pytest.mark.slow
def test_objects_have_required_attributes(sample_graph):
    for i, obj in enumerate(sample_graph['objects']):
        for attr in REQUIRED_OBJECT_ATTRS:
            assert attr in obj, f"Object {i} missing attribute '{attr}': {obj}"


@pytest.mark.slow
def test_relationship_indices_in_range(sample_graph):
    n = len(sample_graph['objects'])
    for rel, lists in sample_graph['relationships'].items():
        for i, related in enumerate(lists):
            for idx in related:
                assert 0 <= idx < n, (
                    f"Relation '{rel}': object {i} references out-of-range index {idx}"
                )


@pytest.mark.slow
def test_scenegraph_compatible_with_dsl_executor(sample_graph):
    #Verify that the builder output can be consumed by the DSL executor without errors
    sg = SceneGraph(sample_graph)
    ex = Executor(sg)
    result = ex.execute([
        {'function': 'scene', 'inputs': [], 'value_inputs': [], 'side_inputs': []},
        {'function': 'count', 'inputs': [0], 'value_inputs': [], 'side_inputs': []},
    ])
    assert isinstance(result, int)
    assert result == len(sample_graph['objects'])


@pytest.mark.slow
def test_build_with_intermediates_returns_tuple(builder):
    if not CLEVR_SAMPLE_IMAGE.exists():
        pytest.skip(f"CLEVR image not found: {CLEVR_SAMPLE_IMAGE}")
    output = builder.build(str(CLEVR_SAMPLE_IMAGE), return_intermediates=True)
    assert isinstance(output, tuple)
    assert len(output) == 2
    sg, intermediates = output
    assert 'objects' in sg
    assert isinstance(intermediates, dict)


# Smoke test: 20 images

@pytest.mark.slow
@pytest.mark.skipif(not CLEVR_VAL.exists(), reason="CLEVR val JSONL not found")
def test_builder_on_20_clevr_images(builder):
    # Build scene graphs for 20 unique CLEVR images and verify structure
    seen_images = set()
    processed = 0

    with open(CLEVR_VAL) as f:
        for line in f:
            if processed >= 20:
                break
            sample = json.loads(line)
            image_path = Path(sample['image_path'])
            if not image_path.exists() or sample['image_id'] in seen_images:
                continue
            seen_images.add(sample['image_id'])

            sg = builder.build(str(image_path))

            n = len(sg['objects'])
            assert set(sg['relationships'].keys()) == REQUIRED_RELATIONS
            for rel_list in sg['relationships'].values():
                assert len(rel_list) == n
            for obj in sg['objects']:
                for attr in REQUIRED_OBJECT_ATTRS:
                    assert attr in obj

            processed += 1

    print(f"\nProcessed {processed} CLEVR images successfully")
