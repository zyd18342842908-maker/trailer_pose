"""Check annotation gestures and confirmed-label persistence without an X server."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, NavigationToolbar2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'annotate_pose_gui.py'
SPEC = importlib.util.spec_from_file_location('annotate_pose_gui', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODULE.configure_plotting()


def write_pcd(path):
    path.write_text('VERSION 0.7\nFIELDS intensity x z y\nSIZE 4 4 4 4\n'
                    'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\n'
                    'POINTS 3\nDATA ascii\n10 -1 0 0.2\n20 -3 0 -0.2\n30 nan 0 0\n')


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'initial.csv'
        self.source.write_text('session,hook_x,hook_y,angle,length\n'
                               'record001,-1.2,0.2,0,1.8\n'
                               'record002,-1.4,0.1,15,1.9\n')
        self.original = self.source.read_bytes()
        self.output = self.root / 'manual.csv'
        self.sessions = []
        for name in ['record001', 'record002']:
            directory = self.root / name
            directory.mkdir()
            write_pcd(directory / '0001.pcd')
            self.sessions.append(directory)

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def make_app(self):
        store = MODULE.LabelStore(self.output, self.source)
        return MODULE.PoseAnnotator(self.sessions, store)

    def mouse(self, app, name, point, button=1, key=None, dblclick=False):
        x, y = app.ax.transData.transform(point)
        return MouseEvent(name, app.fig.canvas, x, y, button=button, key=key, dblclick=dblclick)

    def drag_view_box(self, app, start=(-2.8, -.35), end=(-1.2, .35), key=None):
        for name, point in [('button_press_event', start), ('motion_notify_event', end),
                            ('button_release_event', end)]:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, point, key=key))

    def native_type_entry(self, app):
        """Exercise the native adapter without an X server; Tk owns text editing."""
        value, bindings = {'text': '', 'focus': None}, {}
        entry, canvas = mock.Mock(), mock.Mock()
        entry.get.side_effect = lambda: value['text']
        entry.delete.side_effect = lambda *_: value.update(text='')
        entry.insert.side_effect = lambda _, text: value.update(text=text)
        entry.focus_get.side_effect = lambda: value['focus']
        entry.focus_set.side_effect = lambda: value.update(focus=entry)
        canvas.focus_set.side_effect = lambda: value.update(focus=canvas)
        entry.bind.side_effect = lambda key, callback: bindings.update({key: callback})
        with mock.patch.object(app.fig.canvas, 'get_tk_widget', return_value=canvas, create=True), \
                mock.patch('tkinter.Entry', return_value=entry):
            native = MODULE.NativeTypeEntry(app.type_box, app.on_type,
                                             app.begin_type_edit, app.annotation_mode)
        app.type_box = native
        return entry, bindings

    def replace_native_text(self, entry, text):
        entry.delete(0, 'end')
        entry.insert(0, text)

    def test_named_pcd_fields_and_nan_filter(self):
        points = MODULE.read_pcd_xyz(self.sessions[0] / '0001.pcd')
        np.testing.assert_allclose(points, [[-1, .2, 0], [-3, -.2, 0]])

    def test_binary_pcd_is_rejected(self):
        path = self.root / 'binary.pcd'
        path.write_text('FIELDS x y z\nDATA binary\n')
        with self.assertRaisesRegex(ValueError, 'ASCII'):
            MODULE.read_pcd_xyz(path)

    def test_initial_and_skipped_sessions_are_not_saved(self):
        app = self.make_app()
        self.assertFalse(self.output.exists())
        app.navigate(1)
        self.assertEqual(app.name, 'record002')
        app.save_current()
        with self.output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['session'] for row in rows], ['record002'])
        self.assertEqual(self.source.read_bytes(), self.original)
        resumed = self.make_app()
        self.assertEqual(resumed.name, 'record001')

    def test_drag_hook_rotate_tail_and_undo(self):
        app = self.make_app()
        before = app.state.copy()
        app.on_press(self.mouse(app, 'button_press_event', [-1.3, .15]))
        app.on_motion(self.mouse(app, 'motion_notify_event', [-1.5, .1]))
        app.on_release(self.mouse(app, 'button_release_event', [-1.5, .1]))
        np.testing.assert_allclose(app.state[:2], [-1.5, .1], atol=.005)
        np.testing.assert_allclose(app.state[2:], before[2:])
        hook = app.state[:2].copy()
        tail = [app.tail.get_xdata()[0], app.tail.get_ydata()[0]]
        app.on_press(self.mouse(app, 'button_press_event', tail))
        app.on_motion(self.mouse(app, 'motion_notify_event', hook - [0, 1]))
        app.on_release(self.mouse(app, 'button_release_event', hook - [0, 1]))
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        np.testing.assert_allclose(app.state[:2], hook)
        self.assertEqual(app.state[3], before[3])
        app.undo()
        self.assertAlmostEqual(app.state[2], before[2])

    def test_right_click_heading_and_text_validation(self):
        app = self.make_app()
        hook = app.state[:2].copy()
        app.on_press(self.mouse(app, 'button_press_event', hook - [0, 1], button=3))
        app.on_release(self.mouse(app, 'button_release_event', hook - [0, 1], button=3))
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        app.boxes[0].set_val('-1.65')
        self.assertAlmostEqual(app.state[0], -1.65)
        before = app.state.copy()
        app.boxes[2].set_val('nan')
        app.boxes[3].set_val('-2')
        np.testing.assert_allclose(app.state, before)
        self.assertAlmostEqual(float(app.boxes[2].text), before[2])

    def test_canvas_click_moves_to_destination_and_same_point_shows_drag_hint(self):
        app = self.make_app()
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, [-1.6, .25]))
        np.testing.assert_allclose(app.state[:2], [-1.6, .25], atol=.005)
        before = app.state.copy()
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, before[:2]))
        np.testing.assert_allclose(app.state, before, atol=.005)
        self.assertIn('拖动', app.notice.get_text())

    def test_escape_restores_annotation_after_pan_and_zoom(self):
        app = self.make_app()
        toolbar = NavigationToolbar2(app.fig.canvas)
        for mode in ['pan', 'zoom']:
            getattr(toolbar, mode)()
            before = app.state.copy()
            app.on_press(self.mouse(app, 'button_press_event', [-1.65, .3]))
            np.testing.assert_allclose(app.state, before)
            self.assertIn('标注暂停', app.notice.get_text())
            app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='escape'))
            self.assertFalse(app.toolbar_active())
            app.on_press(self.mouse(app, 'button_press_event', [-1.65, .3]))
            app.on_release(self.mouse(app, 'button_release_event', [-1.65, .3]))
            np.testing.assert_allclose(app.state[:2], [-1.65, .3], atol=.005)

    def test_escape_exits_text_editing_before_annotation(self):
        app = self.make_app()
        app.boxes[0].begin_typing()
        self.assertTrue(app.boxes[0].capturekeystrokes)
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='escape'))
        self.assertFalse(any(box.capturekeystrokes for box in app.boxes))
        self.assertIn('已恢复标注', app.notice.get_text())

    def test_keyboard_fine_adjustment_save_next_and_resume(self):
        app = self.make_app()
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='shift+right'))
        self.assertAlmostEqual(app.state[0], -1.199)
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='E'))
        self.assertAlmostEqual(app.state[2], .1)
        app.save_next()
        self.assertEqual(app.name, 'record002')
        resumed = self.make_app()
        self.assertEqual(resumed.name, 'record002')
        self.assertAlmostEqual(float(resumed.store.confirmed['record001']['angle']), .1)

    def test_failed_atomic_write_preserves_previous_labels(self):
        store = MODULE.LabelStore(self.output, self.source)
        row = dict(store.initial['record001'])
        store.save(row)
        previous = self.output.read_bytes()
        with mock.patch.object(MODULE.os, 'replace', side_effect=PermissionError('read-only')):
            with self.assertRaises(PermissionError):
                store.save(dict(store.initial['record002']))
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(set(store.confirmed), {'record001'})
        self.assertEqual(list(self.root.glob('manual.csv.*')), [])

    def test_seed_file_protection_and_angle_wrap(self):
        with self.assertRaisesRegex(ValueError, '初值'):
            MODULE.LabelStore(self.source, self.source)
        pose = MODULE.normalize_pose([-1, 0, 181, 1.8])
        self.assertEqual(pose[2], -179)

    def test_discover_recording_directory_file_and_unprefixed_sessions(self):
        self.assertEqual(MODULE.discover_sessions(self.root), self.sessions)
        self.assertEqual(MODULE.discover_sessions(self.sessions[0]), [self.sessions[0]])
        self.assertEqual(MODULE.discover_sessions(self.sessions[0] / '0001.pcd'), [self.sessions[0]])
        extra = self.root / 'capture-A'
        extra.mkdir()
        write_pcd(extra / 'scan.PCD')
        self.assertIn(extra, MODULE.discover_sessions(self.root))
        with self.assertRaises(ValueError):
            MODULE.discover_sessions(self.source)

    def test_nested_datasets_use_separate_labels_even_under_project_data(self):
        package_data = self.root / 'package-data'
        recordings = package_data / 'bag_combined'
        original = recordings / 'record001'
        nested_root = recordings / 'another-dataset'
        nested = nested_root / 'record001'
        for directory in [original, nested]:
            directory.mkdir(parents=True)
            write_pcd(directory / 'a.pcd')
        with mock.patch.object(MODULE, 'DATA_ROOT', package_data):
            shared = package_data / 'manual_pose_labels.csv'
            self.assertEqual(MODULE.default_label_paths(recordings)[0], shared)
            self.assertEqual(MODULE.default_label_paths(original / 'a.pcd')[0], shared)
            self.assertEqual(MODULE.default_label_paths(nested_root)[0],
                             nested_root / 'manual_pose_labels.csv')
            self.assertEqual(MODULE.default_label_paths(nested / 'a.pcd')[0],
                             nested / 'manual_pose_labels.csv')

    def test_open_pcd_previews_selected_frame_but_saves_recording_label(self):
        app = self.make_app()
        frame = self.sessions[0] / '0002.PCD'
        frame.write_text('FIELDS x y z\nDATA ascii\n-1.8 0.3 0\n-2.4 0.1 0\n')
        app.load_source(frame)
        self.assertEqual(app.name, 'record001')
        self.assertEqual(app.displayed_files, [frame])
        self.assertEqual(app.n_frames, 2)
        np.testing.assert_allclose(app.points[:, :2], [[-1.8, .3], [-2.4, .1]])
        self.assertEqual(app.store.output, self.output)
        self.assertIn('整段', app.notice.get_text())
        self.assertTrue(app.save_current())
        with self.output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['session'] for row in rows], ['record001'])
        self.assertEqual(rows[0]['n_frames'], '2')
        expanded = self.root / 'frame_labels.csv'
        result = subprocess.run(['python3', str(SCRIPT.parent / 'prepare_dataset.py'),
                                 '--data_dir', str(self.root), '--session_csv', str(self.output),
                                 '--output', str(expanded)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with expanded.open() as stream:
            frames = list(csv.DictReader(stream))
        self.assertEqual(len(frames), 2)
        self.assertEqual({Path(row['file']).name for row in frames}, {'0001.pcd', '0002.PCD'})

    def test_switch_dataset_isolates_same_names_and_restores_custom_output_and_draft(self):
        app = self.make_app()
        app.change([-1.9, .15, 8, 1.8])
        draft = app.state.copy()
        other = self.root / 'other-data'
        recording = other / 'record001'
        recording.mkdir(parents=True)
        write_pcd(recording / 'a.pcd')
        app.load_source(other)
        self.assertEqual(app.store.output, other / 'manual_pose_labels.csv')
        np.testing.assert_allclose(app.state, [-2.2, 0, 0, 1.8])
        app.save_current()
        self.assertTrue((other / 'manual_pose_labels.csv').exists())
        self.assertFalse(self.output.exists())
        app.load_source(self.root)
        self.assertEqual(app.store.output, self.output)
        np.testing.assert_allclose(app.state, draft)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_bad_cloud_or_cancelled_dialog_preserves_current_dataset(self):
        app = self.make_app()
        state, output = app.state.copy(), app.store.output
        with mock.patch.object(app, 'choose_path', return_value=''):
            app.open_directory_dialog()
        np.testing.assert_array_equal(app.state, state)
        bad = self.root / 'bad-data' / 'record001'
        bad.mkdir(parents=True)
        (bad / 'bad.pcd').write_text('FIELDS x y z\nDATA binary\n')
        with mock.patch.object(app, 'choose_path', return_value=str(bad)):
            app.open_directory_dialog()
        self.assertIn('打开失败', app.notice.get_text())
        np.testing.assert_array_equal(app.state, state)
        self.assertEqual(app.store.output, output)
        self.assertEqual(app.sessions, self.sessions)
        self.assertFalse((bad / 'manual_pose_labels.csv').exists())

    def test_filter_search_list_selection_and_confirmation(self):
        app = self.make_app()
        app.save_current()
        app.set_filter('pending')
        self.assertEqual(app.visible_indices, [1])
        app.search_box.set_val('002')
        self.assertEqual(app.visible_indices, [1])
        x, y = app.list_ax.transData.transform([.5, app.page_size - .5])
        event = MouseEvent('button_press_event', app.fig.canvas, x, y, button=1)
        app.fig.canvas.callbacks.process('button_press_event', event)
        self.assertEqual(app.name, 'record002')
        app.save_current()
        self.assertEqual(app.visible_indices, [])
        app.set_filter('confirmed')
        self.assertEqual(app.visible_indices, [1])
        app.search_box.set_val('')
        self.assertEqual(app.visible_indices, [0, 1])

    def test_list_pagination_and_filtered_navigation(self):
        for number in range(3, 13):
            folder = self.root / ('record%03d' % number)
            folder.mkdir()
            write_pcd(folder / 'a.pcd')
        sessions = sorted(self.root.glob('record*'))
        app = MODULE.PoseAnnotator(sessions, MODULE.LabelStore(self.output, self.source))
        app.turn_page(1)
        self.assertEqual(app.list_page, 1)
        x, y = app.list_ax.transData.transform([.5, app.page_size - .5])
        app.on_press(MouseEvent('button_press_event', app.fig.canvas, x, y, button=1))
        self.assertEqual(app.name, 'record009')
        app.search_box.set_val('012')
        app.navigate(1)
        self.assertEqual(app.name, 'record012')
        app.search_box.set_val('no-match')
        app.navigate(-1)
        self.assertEqual(app.name, 'record012')

    def test_explicit_heading_mode_and_keyboard_modes(self):
        app = self.make_app()
        hook = app.state[:2].copy()
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='2'))
        self.assertEqual(app.mode, 'heading')
        app.on_press(self.mouse(app, 'button_press_event', hook - [0, 1]))
        app.on_release(self.mouse(app, 'button_release_event', hook - [0, 1]))
        np.testing.assert_allclose(app.state[:2], hook)
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='1'))
        app.on_press(self.mouse(app, 'button_press_event', [-1.6, .3]))
        app.on_release(self.mouse(app, 'button_release_event', [-1.6, .3]))
        np.testing.assert_allclose(app.state[:2], [-1.6, .3], atol=.005)
        self.assertAlmostEqual(app.state[2], 90, delta=.4)

    def test_browse_pan_zoom_and_keys_never_change_pose(self):
        app = self.make_app()
        pose, limits = app.state.copy(), app.ax.get_xlim()
        app.set_mode('browse')
        app.on_press(self.mouse(app, 'button_press_event', [-2, 0]))
        app.on_motion(self.mouse(app, 'motion_notify_event', [-2.3, .2]))
        app.on_release(self.mouse(app, 'button_release_event', [-2.3, .2]))
        self.assertFalse(np.allclose(app.ax.get_xlim(), limits))
        span = np.ptp(app.ax.get_xlim())
        center = [np.mean(app.ax.get_xlim()), np.mean(app.ax.get_ylim())]
        x, y = app.ax.transData.transform(center)
        app.on_scroll(MouseEvent('scroll_event', app.fig.canvas, x, y, button='up', step=1))
        self.assertAlmostEqual(np.ptp(app.ax.get_xlim()), span / 1.2)
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='right'))
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='e'))
        np.testing.assert_array_equal(app.state, pose)

    def test_browse_toolbar_release_and_search_input_does_not_save(self):
        app = self.make_app()
        toolbar = NavigationToolbar2(app.fig.canvas)
        app.set_mode('browse')
        self.assertTrue(toolbar.mode)
        app.set_mode('heading')
        self.assertFalse(toolbar.mode)
        app.search_box.begin_typing()
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='enter'))
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='s'))
        self.assertFalse(self.output.exists())
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='escape'))
        self.assertFalse(app.search_box.capturekeystrokes)
        self.assertEqual(app.mode, 'position')

    def test_native_pan_activated_during_drag_stops_pose_changes(self):
        app = self.make_app()
        app.on_press(self.mouse(app, 'button_press_event', [-1.5, .2]))
        before = app.state.copy()
        toolbar = NavigationToolbar2(app.fig.canvas)
        toolbar.pan()
        app.on_motion(self.mouse(app, 'motion_notify_event', [-1.8, .3]))
        np.testing.assert_array_equal(app.state, before)
        self.assertEqual(app.mode, 'browse')
        self.assertIsNone(app.dragging)
        toolbar.pan()
        toolbar.zoom()
        app.on_motion(self.mouse(app, 'motion_notify_event', [-1.8, .3]))
        self.assertEqual(app.mode, 'zoom')
        np.testing.assert_array_equal(app.state, before)

    def test_type_is_saved_resumed_and_edit_marks_dirty(self):
        app = self.make_app()
        app.type_box.set_val('cart_A')
        self.assertEqual(app.trailer_type, 'cart_A')
        self.assertFalse(self.output.exists())
        app.save_current()
        with self.output.open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['trailer_type'], 'cart_A')
        self.assertEqual(row['session_dir'], str(self.sessions[0].resolve()))
        self.assertTrue(app.current_is_saved())
        app.on_type('cart_B')
        self.assertFalse(app.current_is_saved())
        app.undo()
        self.assertEqual(app.trailer_type, 'cart_A')
        self.assertTrue(app.current_is_saved())
        resumed = MODULE.PoseAnnotator(self.sessions, MODULE.LabelStore(self.output, self.source),
                                       start_session='record001')
        self.assertEqual(resumed.trailer_type, 'cart_A')

    def test_type_draft_carry_and_search(self):
        app = self.make_app()
        app.on_type('cart_A')
        app.navigate(1)
        self.assertEqual(app.trailer_type, 'cart_A')
        app.on_type('cart_B')
        app.navigate(-1)
        self.assertEqual(app.trailer_type, 'cart_A')
        app.search_box.set_val('cart_B')
        self.assertEqual(app.visible_indices, [1])
        app.reset()
        self.assertEqual(app.trailer_type, 'default')
        self.assertFalse(self.output.exists())

    def test_default_type_and_type_input_guard(self):
        app = MODULE.PoseAnnotator(self.sessions, MODULE.LabelStore(self.output, self.source),
                                   trailer_type='cart_A')
        self.assertEqual(app.trailer_type, 'cart_A')
        app.on_type(' ')
        self.assertEqual(app.trailer_type, 'cart_A')
        app.type_box.begin_typing()
        before = app.state.copy()
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='right'))
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='enter'))
        np.testing.assert_array_equal(app.state, before)
        self.assertFalse(self.output.exists())
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='escape'))
        self.assertFalse(app.type_box.capturekeystrokes)

    def test_same_name_in_another_directory_cannot_overwrite_saved_label(self):
        app = self.make_app()
        app.on_type('cart_A')
        app.save_current()
        before = self.output.read_bytes()
        row = dict(app.store.confirmed[app.name], session_dir=str(self.root / 'other' / app.name),
                   trailer_type='cart_B')
        with self.assertRaisesRegex(ValueError, '独立'):
            app.store.save(row)
        self.assertEqual(self.output.read_bytes(), before)

    def test_native_type_input_submits_unicode_without_saving_or_moving_pose(self):
        app = self.make_app()
        before = app.state.copy()
        entry, bindings = self.native_type_entry(app)
        app.type_box.begin_typing()
        bindings['<FocusIn>'](None)
        self.replace_native_text(entry, '挂车 TG-01')
        self.assertEqual(bindings['<Return>'](None), 'break')
        self.assertEqual(app.trailer_type, '挂车 TG-01')
        np.testing.assert_array_equal(app.state, before)
        self.assertFalse(self.output.exists())
        app.save_current()
        with self.output.open(encoding='utf-8') as stream:
            self.assertEqual(next(csv.DictReader(stream))['trailer_type'], '挂车 TG-01')
        app.on_type('TG-02')
        app.undo()
        self.assertEqual(entry.get(), '挂车 TG-01')

    def test_native_type_focus_select_all_and_escape(self):
        app = self.make_app()
        entry, bindings = self.native_type_entry(app)
        app.search_box.begin_typing()
        app.type_box.begin_typing()
        bindings['<FocusIn>'](None)
        self.assertFalse(app.search_box.capturekeystrokes)
        self.assertTrue(app.type_box.capturekeystrokes)
        self.assertEqual(bindings['<Control-a>'](None), 'break')
        entry.selection_range.assert_called_once_with(0, 'end')
        entry.icursor.assert_called_once_with('end')
        self.replace_native_text(entry, '车型乙')
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='s'))
        self.assertFalse(self.output.exists())
        bindings['<Escape>'](None)
        self.assertFalse(app.type_box.capturekeystrokes)
        self.assertEqual(app.trailer_type, '车型乙')
        self.assertEqual(app.mode, 'position')
        self.assertFalse(self.output.exists())

    def test_native_type_save_and_navigation_commit_current_entry(self):
        app = self.make_app()
        entry, bindings = self.native_type_entry(app)
        self.replace_native_text(entry, 'TG-01')
        self.assertEqual(app.trailer_type, 'default')
        app.save_current()
        self.assertEqual(app.store.confirmed[app.name]['trailer_type'], 'TG-01')
        self.replace_native_text(entry, 'TG-02')
        app.navigate(1)
        self.assertEqual(app.trailer_type, 'TG-02')
        app.navigate(-1)
        self.assertEqual(app.trailer_type, 'TG-02')
        self.assertFalse(app.current_is_saved())
        self.replace_native_text(entry, '')
        bindings['<FocusOut>'](None)
        self.assertEqual(entry.get(), 'TG-02')

    def test_type_mouse_click_and_keyboard_input_reach_annotation_state(self):
        app = self.make_app()
        x, y = app.type_box.ax.transAxes.transform((.9, .5))
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, MouseEvent(name, app.fig.canvas, x, y, button=1))
        self.assertTrue(app.type_box.capturekeystrokes)
        for key in ['end'] + ['backspace'] * len(app.type_box.text) + list('TG-01') + ['enter']:
            app.fig.canvas.callbacks.process('key_press_event',
                                             KeyEvent('key_press_event', app.fig.canvas, key=key))
        self.assertEqual(app.trailer_type, 'TG-01')
        self.assertFalse(self.output.exists())

    def test_box_zoom_draws_rectangle_without_editing_or_confirming_labels(self):
        app = self.make_app()
        pose = app.state.copy()
        limits = np.array([app.ax.get_xlim(), app.ax.get_ylim()])
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='4'))
        self.assertEqual(app.mode, 'zoom')
        app.on_press(self.mouse(app, 'button_press_event', [-2.8, -.35]))
        app.on_motion(self.mouse(app, 'motion_notify_event', [-1.2, .35]))
        self.assertTrue(app.zoom_patch.get_visible())
        self.assertGreater(app.zoom_patch.get_width(), 1.5)
        app.on_release(self.mouse(app, 'button_release_event', [-1.2, .35]))
        self.assertFalse(app.zoom_patch.get_visible())
        self.assertLess(np.ptp(app.ax.get_xlim()), np.ptp(limits[0]))
        self.assertLess(np.ptp(app.ax.get_ylim()), np.ptp(limits[1]))
        self.assertLessEqual(app.ax.get_xlim()[0], -2.8 + .01)
        self.assertGreaterEqual(app.ax.get_xlim()[1], -1.2 - .01)
        np.testing.assert_array_equal(app.state, pose)
        self.assertEqual(app.histories[app.session_id], [])
        self.assertFalse(self.output.exists())
        app.previous_view()
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], limits)

    def test_box_zoom_right_click_wheel_and_view_history_with_toolbar(self):
        app = self.make_app()
        pose = app.state.copy()
        toolbar = NavigationToolbar2(app.fig.canvas)
        toolbar.pan()
        app.set_mode('zoom')
        self.assertFalse(toolbar.mode)
        self.drag_view_box(app, start=(-1.2, .35), end=(-2.8, -.35))
        selected = np.array([app.ax.get_xlim(), app.ax.get_ylim()])
        center = selected.mean(axis=1)
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, center, button=3))
        self.assertAlmostEqual(np.ptp(app.ax.get_xlim()), np.ptp(selected[0]) * 1.5)
        x, y = app.ax.transData.transform(center)
        app.on_scroll(MouseEvent('scroll_event', app.fig.canvas, x, y, button='up', step=1))
        self.assertAlmostEqual(np.ptp(app.ax.get_xlim()), np.ptp(selected[0]) * 1.5 / 1.2)
        app.previous_view()
        app.previous_view()
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], selected)
        for key in ['right', 'e']:
            app.on_key(KeyEvent('key_press_event', app.fig.canvas, key=key))
        np.testing.assert_array_equal(app.state, pose)

    def test_shift_box_zoom_and_cancel_or_small_selection(self):
        app = self.make_app()
        pose = app.state.copy()
        self.drag_view_box(app, key='shift')
        self.assertEqual(app.mode, 'position')
        np.testing.assert_array_equal(app.state, pose)
        selected = np.array([app.ax.get_xlim(), app.ax.get_ylim()])
        app.set_mode('zoom')
        self.drag_view_box(app, start=(-2, 0), end=(-2, 0))
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], selected)
        app.on_press(self.mouse(app, 'button_press_event', [-2.2, -.1]))
        app.on_motion(self.mouse(app, 'motion_notify_event', [-1.8, .1]))
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='escape'))
        app.on_release(self.mouse(app, 'button_release_event', [-1.8, .1]))
        self.assertFalse(app.zoom_patch.get_visible())
        self.assertIsNone(app.dragging)
        self.assertEqual(app.mode, 'position')
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], selected)
        np.testing.assert_array_equal(app.state, pose)

    def test_box_zoom_release_outside_plot_and_scene_switch(self):
        app = self.make_app()
        app.set_mode('zoom')
        app.on_press(self.mouse(app, 'button_press_event', [-2, 0]))
        outside = MouseEvent('button_release_event', app.fig.canvas,
                              app.ax.bbox.x0 - 50, app.ax.bbox.y0 - 50, button=1)
        app.on_release(outside)
        self.assertTrue(np.isfinite([app.ax.get_xlim(), app.ax.get_ylim()]).all())
        self.assertGreater(np.ptp(app.ax.get_xlim()), 0)
        self.assertTrue(app.view_history)
        app.navigate(1)
        self.assertEqual(app.view_history, [])
        self.assertFalse(app.zoom_patch.get_visible())
        self.drag_view_box(app)
        self.assertTrue(app.view_history)
        self.assertFalse(self.output.exists())

    def test_box_zoom_button_full_view_and_double_click(self):
        app = self.make_app()
        pose = app.state.copy()
        full = np.array([app.ax.get_xlim(), app.ax.get_ylim()])

        def click_button(button):
            x, y = button.ax.transAxes.transform((.5, .5))
            for name in ['button_press_event', 'button_release_event']:
                app.fig.canvas.callbacks.process(name, MouseEvent(name, app.fig.canvas, x, y, button=1))

        click_button(app.mode_buttons['zoom'])
        self.drag_view_box(app)
        selected = np.array([app.ax.get_xlim(), app.ax.get_ylim()])
        click_button(next(button for button in app.buttons if button.label.get_text() == '恢复全图 F'))
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], full)
        app.on_key(KeyEvent('key_press_event', app.fig.canvas, key='v'))
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], selected)
        center = selected.mean(axis=1)
        app.on_press(self.mouse(app, 'button_press_event', center, dblclick=True))
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], full)
        np.testing.assert_array_equal(app.state, pose)
        self.assertFalse(self.output.exists())

    def test_zoomed_arrow_stays_in_plot_and_axis_label_does_not_overlap_pose(self):
        app = MODULE.PoseAnnotator(self.sessions, MODULE.LabelStore(self.output, self.source),
                                   font_size=20)
        app.set_mode('zoom')
        self.drag_view_box(app, start=(-1.65, -.1), end=(-.95, .5))
        app.fig.canvas.draw()
        renderer = app.fig.canvas.get_renderer()
        self.assertFalse(app.live.get_window_extent(renderer).overlaps(
            app.ax.xaxis.label.get_window_extent(renderer)))
        pixels = np.asarray(app.fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        red = ((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 110)
               & (pixels[:, :, 2] < 110) & (np.abs(pixels[:, :, 1] - pixels[:, :, 2]) < 25))
        self.assertTrue(red.any(), 'The hook or arrow must be visible for this clipping check')
        outside = red.copy()
        x0, y0, x1, y1 = app.ax.bbox.extents.astype(int)
        height, width = red.shape
        outside[max(0, height-y1-2):min(height, height-y0+2),
                max(0, x0-2):min(width, x1+2)] = False
        self.assertFalse(outside.any(), 'The red pose arrow must not cross the controls')

    def test_quick_heading_gestures_allow_immediate_position_edits(self):
        app = self.make_app()
        hook = app.state[:2].copy()
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, hook - [0, 1], button=3))
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        self.assertEqual(app.mode, 'position')
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, [-1.6, .25]))
        np.testing.assert_allclose(app.state[:2], [-1.6, .25], atol=.005)
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        # A 90-degree tail is initially outside the view; zoom out to reach its handle.
        x, y = app.ax.transData.transform([-2, 0])
        app.fig.canvas.callbacks.process('scroll_event', MouseEvent(
            'scroll_event', app.fig.canvas, x, y, button='down', step=-2))
        tail = [app.tail.get_xdata()[0], app.tail.get_ydata()[0]]
        target = app.state[:2] - [1, 0]
        for name, point in [('button_press_event', tail), ('motion_notify_event', target),
                            ('button_release_event', target)]:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, point))
        self.assertAlmostEqual(app.state[2], 0, delta=.4)
        self.assertEqual(app.mode, 'position')
        self.assertFalse(self.output.exists())

    def test_quick_wheel_zooms_and_ctrl_wheel_only_adjusts_heading(self):
        app = self.make_app()
        pose = app.state.copy()
        span = np.ptp(app.ax.get_xlim())
        x, y = app.ax.transData.transform([-2, 0])
        app.fig.canvas.callbacks.process('scroll_event', MouseEvent(
            'scroll_event', app.fig.canvas, x, y, button='up', step=1))
        self.assertAlmostEqual(np.ptp(app.ax.get_xlim()), span / 1.2)
        np.testing.assert_array_equal(app.state, pose)
        limits = [app.ax.get_xlim(), app.ax.get_ylim()]
        for key, step in [('control', 1), ('ctrl', -1)]:
            x, y = app.ax.transData.transform([-2, 0])
            app.fig.canvas.callbacks.process('scroll_event', MouseEvent(
                'scroll_event', app.fig.canvas, x, y, button='up' if step > 0 else 'down',
                key=key, step=step))
            self.assertAlmostEqual(app.state[2], pose[2] + (.1 if step > 0 else 0))
            np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], limits)
        self.assertEqual(app.mode, 'position')
        self.assertFalse(self.output.exists())

    def test_quick_middle_pan_and_double_middle_restore_with_toolbar(self):
        app = self.make_app()
        pose = app.state.copy()
        full = np.array([app.ax.get_xlim(), app.ax.get_ylim()])
        toolbar = NavigationToolbar2(app.fig.canvas)
        toolbar.zoom()
        for name, point in [('button_press_event', [-2, 0]), ('motion_notify_event', [-2.3, .2]),
                            ('button_release_event', [-2.3, .2])]:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, point, button=2))
        self.assertFalse(toolbar.mode)
        self.assertFalse(np.allclose([app.ax.get_xlim(), app.ax.get_ylim()], full))
        np.testing.assert_array_equal(app.state, pose)
        self.assertEqual(app.mode, 'position')
        center = np.array([app.ax.get_xlim(), app.ax.get_ylim()]).mean(axis=1)
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, center, button=2, dblclick=True))
        np.testing.assert_allclose([app.ax.get_xlim(), app.ax.get_ylim()], full)
        np.testing.assert_array_equal(app.state, pose)
        self.assertEqual(app.histories[app.session_id], [])
        self.assertFalse(self.output.exists())

    def test_quick_workflow_saves_pose_and_type_without_mode_switches(self):
        app = self.make_app()
        app.type_box.set_val('TG-01')
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, [-1.6, .2]))
        hook = app.state[:2].copy()
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, hook - [0, 1], button=3))
        before_view = app.state.copy()
        self.drag_view_box(app, key='shift')
        for name, point in [('button_press_event', [-2, 0]), ('motion_notify_event', [-2.1, .1]),
                            ('button_release_event', [-2.1, .1])]:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, point, button=2))
        np.testing.assert_array_equal(app.state, before_view)
        target = np.array([app.ax.get_xlim(), app.ax.get_ylim()]).mean(axis=1)
        for name in ['button_press_event', 'button_release_event']:
            app.fig.canvas.callbacks.process(name, self.mouse(app, name, target))
        np.testing.assert_allclose(app.state[:2], target, atol=.005)
        self.assertAlmostEqual(app.state[2], 90, delta=.4)
        self.assertEqual(app.mode, 'position')
        self.assertFalse(self.output.exists())
        self.assertTrue(app.save_current())
        with self.output.open(encoding='utf-8') as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved['trailer_type'], 'TG-01')
        np.testing.assert_allclose([float(saved['hook_x']), float(saved['hook_y'])], target, atol=.005)
        self.assertAlmostEqual(float(saved['angle']), 90, delta=.4)

    def test_render_drag_reuses_cloud_and_release_keeps_final_pointer(self):
        app = self.make_app()
        canvas = app.fig.canvas
        with mock.patch.object(canvas, 'draw', wraps=canvas.draw) as full_draw, \
                mock.patch.object(app.ax, 'draw', wraps=app.ax.draw) as plot_draw:
            canvas.callbacks.process('button_press_event', self.mouse(app, 'button_press_event', [-1.5, .2]))
            for index in range(20):
                canvas.callbacks.process('motion_notify_event', self.mouse(
                    app, 'motion_notify_event', [-1.5 - .01 * index, .2]))
            self.assertEqual(full_draw.call_count, 0)
            self.assertEqual(plot_draw.call_count, 0)
            cached = np.asarray(canvas.buffer_rgba()).copy()
            canvas.draw()
            np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba()), cached)
            full_draw.reset_mock()
            canvas.callbacks.process('button_release_event', self.mouse(
                app, 'button_release_event', [-1.8, .3]))
            self.assertEqual(full_draw.call_count, 1)
        np.testing.assert_allclose(app.state[:2], [-1.8, .3], atol=.005)
        np.testing.assert_allclose([float(box.text) for box in app.boxes], app.state)
        self.assertEqual(len(app.histories[app.session_id]), 1)
        self.assertFalse(self.output.exists())

    def test_render_pan_and_zoom_refresh_only_plot_without_ghosts(self):
        app = self.make_app()
        canvas = app.fig.canvas
        pose = app.state.copy()
        with mock.patch.object(canvas, 'draw', wraps=canvas.draw) as full_draw, \
                mock.patch.object(app.ax, 'draw', wraps=app.ax.draw) as plot_draw:
            app.on_press(self.mouse(app, 'button_press_event', [-2, 0], button=2))
            app.on_motion(self.mouse(app, 'motion_notify_event', [-2.3, .1], button=2))
            x, y = app.ax.transData.transform([-2, 0])
            app.on_scroll(MouseEvent('scroll_event', canvas, x, y, button='up', step=1))
            self.assertEqual(full_draw.call_count, 0)
            self.assertGreater(plot_draw.call_count, 0)
            cached = np.asarray(canvas.buffer_rgba()).copy()
            canvas.draw()
            np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba()), cached)
        np.testing.assert_array_equal(app.state, pose)
        self.assertFalse(self.output.exists())

    def test_render_coalesces_events_and_release_flushes_latest_state(self):
        app = self.make_app()
        painter = app.renderer
        painter.interactive = True
        painter.timer = mock.Mock()
        painter.last_paint = 100.
        with mock.patch.object(MODULE.time, 'perf_counter', return_value=100.001), \
                mock.patch.object(app.fig.canvas, 'draw', wraps=app.fig.canvas.draw) as full_draw:
            app.on_press(self.mouse(app, 'button_press_event', [-1.5, .2]))
            for index in range(20):
                app.on_motion(self.mouse(app, 'motion_notify_event', [-1.5 - .01 * index, .2]))
            painter.timer.start.assert_called_once()
            self.assertTrue(painter.pending)
            self.assertEqual(full_draw.call_count, 0)
            app.on_release(self.mouse(app, 'button_release_event', [-1.9, .3]))
            self.assertEqual(full_draw.call_count, 1)
            self.assertFalse(painter.pending)
            self.assertFalse(painter.timer_running)
        np.testing.assert_allclose(app.state[:2], [-1.9, .3], atol=.005)
        self.assertEqual(app.boxes[0].text, '-1.900000')
        self.assertFalse(self.output.exists())
        with mock.patch.object(app.fig.canvas, 'draw', wraps=app.fig.canvas.draw) as full_draw:
            painter.close()
            painter.request(full=True, immediate=True)
            painter.flush()
            self.assertEqual(full_draw.call_count, 0)
            self.assertFalse(painter.pending)

    def test_render_resize_and_export_preserve_annotations(self):
        app = self.make_app()
        canvas = app.fig.canvas
        app.fig.set_size_inches(16, 9)
        with mock.patch.object(canvas, 'draw', wraps=canvas.draw) as full_draw:
            app.renderer.request()
            self.assertEqual(full_draw.call_count, 1)
        app.change([-1.4, .15, 5, 1.8])
        cached_export = self.root / 'cached.png'
        regular_export = self.root / 'regular.png'
        app.fig.savefig(cached_export, dpi=110)
        self.assertIsNone(app.renderer.ui_background)
        app.renderer.close()
        app.fig.savefig(regular_export, dpi=110)
        np.testing.assert_array_equal(plt.imread(cached_export), plt.imread(regular_export))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
