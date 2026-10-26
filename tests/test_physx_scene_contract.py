"""Source-scene reconstruction contracts; no Isaac runtime needed for these checks."""
import json
import socket
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from physx_scene_builder import (
    UnsupportedRawSceneError, _hand_reach_from_geometry, _object_geometry, _validate_objects,
)
from isaac_physx_client import IsaacPhysxClient, UnsupportedRawSceneError as ClientUnsupportedRawSceneError
from isaac_render_protocol import send_packet


def _object(class_name, index, *, joint=None, size=None):
    dimensions = {'bottle': [0.035, 0.18], 'bin': [0.35, 0.25, 0.15],
                  'cup': [0.045, 0.09, 0.002, 0.028, 0.018],
                  'mug': [0.04, 0.09, 0.003], 'mug_tree': [0.2, 0.2, 0.3],
                  'cabinet': [0.26, 0.32, 0.426], 'drawer': [0.238, 0.292, 0.085],
                  'letter_block': [0.04, 0.04, 0.04]}[class_name]
    return {'name': f'object_{index:03}', 'class_name': class_name,
            'instance_id': index, 'size': size or dimensions, 'mass_kg': 0.1,
            'friction': 0.5, 'collision_debug_rgb': [0.2, 0.3, 0.4],
            'joint': joint or {}}


def _scene(objects, pose_indices):
    return {'pose_node_ids': [f'body/object_{i:03}' for i in pose_indices],
            'nodes': [{'id': f'body/object_{o["instance_id"]:03}',
                       'parent_id': 'world', 'driver': 'object_pose' if o['instance_id'] in pose_indices else 'static',
                       'translation': [0.5, 0, 0.65], 'rotation_wxyz': [1, 0, 0, 0]}
                      for o in objects],
            'sampled_parameters': {'scene_manifest.objects': objects,
                                   'scene_manifest.table': {'center_xyz_m': [0.5, 0, 0.625],
                                                            'size_xyz_m': [0.8, 1.1, 0.05]},
                                   'scene_manifest.sampled_values': {'geometry_revision': 'paper-aligned-scenes-v2',
                                                                       'affordances': {
                                       str(o['instance_id']): {'wall_thickness_m': 0.008}
                                       for o in objects if o['class_name'] == 'bin'}}}}


def test_multibottle_pose_order_is_recorded_not_object_manifest_order():
    objects = [_object('bin', 1), _object('bottle', 2), _object('bottle', 3), _object('bottle', 4)]
    scene = _scene(objects, [4, 2, 3])
    assert _validate_objects(scene) == [('object_004', 'free'), ('object_002', 'free'), ('object_003', 'free')]


def test_cup_and_mug_collisions_have_open_cavities_and_physical_handles():
    for name, min_count in [('cup', 36), ('mug', 65)]:
        obj = _object(name, 1)
        geoms = _object_geometry(obj, _scene([obj], [1]))
        assert len(geoms) >= min_count
        assert all(geom['type'] != 'mesh' for geom in geoms)
        assert any('bottom' in geom['name'] for geom in geoms)
        assert any('wall_' in geom['name'] for geom in geoms)
        assert not any('filled' in geom['name'] for geom in geoms)


def test_drawer_slide_is_object_pose_not_free_robot_axis():
    objects = [_object('cabinet', 1), *[_object('drawer', i, joint={
        'type': 'slide', 'axis': [-1, 0, 0], 'range': [0, .25], 'ref': 0.,
        'limited': 'true', 'damping': 2., 'frictionloss': .3}) for i in (2, 3, 4)],
        _object('letter_block', 5)]
    scene = _scene(objects, [2, 3, 4, 5])
    assert _validate_objects(scene) == [('object_002', 'slide'), ('object_003', 'slide'),
                                        ('object_004', 'slide'), ('object_005', 'free')]
    geoms = _object_geometry(objects[0], scene)
    assert any('shelf_' in geom['name'] for geom in geoms)
    assert not any(geom['name'].endswith('_filled') for geom in geoms)


def test_unsupported_joint_and_missing_dynamic_pose_are_rejected():
    obj = _object('letter_block', 1, joint={'type': 'ball'})
    with pytest.raises(ValueError, match='Unsupported object joint'):
        _validate_objects(_scene([obj], [1]))
    obj = _object('letter_block', 1)
    with pytest.raises(ValueError, match='Object driver differs|pose_node_ids'):
        _validate_objects(_scene([obj], []))


def test_unknown_source_geometry_revision_is_explicitly_unsupported():
    obj = _object('letter_block', 1)
    scene = _scene([obj], [1])
    scene['sampled_parameters']['scene_manifest.sampled_values']['geometry_revision'] = 'future-v3'
    with pytest.raises(ValueError, match='geometry revision'):
        _validate_objects(scene)


def test_client_requires_measured_hands_on_every_nonimage_tick():
    client = IsaacPhysxClient.__new__(IsaacPhysxClient)
    client.capture = 'policy_only'
    client.round_id = 'test-round'
    client._shape = (3, 480, 640, 3)
    packet = {'round_id': 'test-round', 'tick': 1, 'capture_mode': 'policy_only',
              'policy_images': False, 'raw_count': 0, 'policy_count': 0,
              'dtype': 'uint8', 'shape': [0, 480, 640, 3],
              'state': [0.] * 54, 'applied_command': [0.] * 54,
              'object_pose': [[.5, 0., .65, 1., 0., 0., 0.]],
              'hand_positions': [[.1, .2, .3], [.4, .5, .6]]}
    frame = client._frame(packet, b'', 1)
    np.testing.assert_array_equal(frame['hand_positions'], [[.1, .2, .3], [.4, .5, .6]])
    assert frame['object_pose'].shape == (1, 7)
    del packet['hand_positions']
    with pytest.raises((KeyError, RuntimeError)):
        client._frame(packet, b'', 1)


def test_client_distinguishes_raw_unsupported_from_runtime_error(tmp_path):
    client = IsaacPhysxClient.__new__(IsaacPhysxClient)
    client.channel, server = socket.socketpair()
    client.log_path = tmp_path / 'worker.log'
    try:
        send_packet(server, {'ok': False, 'error_kind': 'unsupported_raw_scene',
                             'error': 'missing original table metadata'})
        with pytest.raises(ClientUnsupportedRawSceneError, match='missing original table metadata'):
            client._receive()
        send_packet(server, {'ok': False, 'error': 'PhysX did not initialize'})
        with pytest.raises(RuntimeError, match='PhysX did not initialize'):
            client._receive()
    finally:
        server.close()
        client.channel.close()


@pytest.mark.parametrize('task', ['hang_mug', 'pyramid', 'spelling', 'toss_in_bin',
                                  'unstack', 'vowel_consonant_sort'])
def test_real_raw_scene_patterns_are_represented_or_explicitly_unsupported(task):
    root = Path('/root/sunyz/data/spd-sim/raw') / task / 'scenes'
    if not root.is_dir():
        pytest.skip('Raw source scenes are only available on the evaluation host')
    paths = sorted(root.glob('*.json'))
    assert paths, f'No raw scenes for {task}'
    unsupported = 0
    for path in paths:
        scene = json.loads(path.read_text())
        sampled = scene['sampled_parameters']
        if 'scene_manifest.table' not in sampled or 'scene_manifest.sampled_values' not in sampled:
            assert task == 'toss_in_bin'
            with pytest.raises(UnsupportedRawSceneError, match='scene_manifest.table.*scene_manifest.sampled_values'):
                _validate_objects(scene, task)
            unsupported += 1
            continue
        objects = sampled['scene_manifest.objects']
        kinds = _validate_objects(scene, task)
        assert len(kinds) == len(scene['pose_node_ids'])
        for obj in objects:
            if obj['class_name'] == 'bottle':
                continue  # convex asset manifest is loaded by the physical builder
            geoms = _object_geometry(obj, scene)
            assert geoms, (task, path, obj['name'])
    if task == 'toss_in_bin':
        assert unsupported > 0
    else:
        assert unsupported == 0


def test_wrist_bound_includes_all_descendant_joint_offsets_and_mesh_vertices(tmp_path):
    mesh = tmp_path / 'collision.STL'
    vertices = ((.04, 0., 0.), (0., 0., 0.), (0., 0., 0.))
    mesh.write_bytes(b'\x00' * 80 + struct.pack('<I', 1) +
                     struct.pack('<12fH', 0., 0., 1., *vertices[0], *vertices[1], *vertices[2], 0))
    root = ET.Element('mujoco')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='physical', file=str(mesh))
    world = ET.SubElement(root, 'worldbody')
    for side, length in (('l', .1), ('r', .2)):
        wrist = ET.SubElement(world, 'body', name=f'{side}__{side}_wrist')
        finger = ET.SubElement(wrist, 'body', name=f'{side}__finger', pos=f'{length} 0 0')
        ET.SubElement(finger, 'geom', type='mesh', mesh='physical', group='3')
    reach, sides = _hand_reach_from_geometry(root)
    assert np.isclose(reach, .24)
    np.testing.assert_allclose(sides, [.14, .24])
