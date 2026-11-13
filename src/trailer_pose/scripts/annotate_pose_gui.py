#!/usr/bin/env python3
"""Manually label hook x/y and heading in a draggable BEV view."""
import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from matplotlib.widgets import TextBox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trailer_types import DEFAULT_TYPE, normalize_type, row_type

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / 'data'
FIELDS = ['session', 'hook_x', 'hook_y', 'angle', 'length', 'n_frames', 'n_points',
          'trailer_type', 'session_dir']


def pcd_files(directory):
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == '.pcd')


def discover_sessions(source):
    """Accept a PCD, a recording folder, or a root of recording folders."""
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != '.pcd':
            raise ValueError('请选择 PCD 点云文件')
        return [source.parent]
    if not source.is_dir():
        raise ValueError('点云路径不存在：%s' % source)
    if pcd_files(source):
        return [source]
    sessions = sorted(path for path in source.iterdir() if path.is_dir() and pcd_files(path))
    if not sessions:
        raise ValueError('该目录及其直接子目录中没有 PCD 文件：%s' % source)
    return sessions


def default_label_paths(source):
    folder = source.parent if source.is_file() else source
    project_data = (DATA_ROOT / 'bag_combined').resolve()
    if folder.resolve() == project_data or (
            folder.resolve().parent == project_data and pcd_files(folder)):
        return DATA_ROOT / 'manual_pose_labels.csv', DATA_ROOT / 'session_analysis.csv'
    seed = folder / 'session_analysis.csv'
    return folder / 'manual_pose_labels.csv', seed if seed.is_file() else None


def read_pcd_xyz(path):
    """Read named XYZ fields from an ASCII PCD without importing Torch or ROS."""
    with path.open() as stream:
        fields = []
        for line in stream:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'FIELDS':
                fields = tokens[1:]
            if tokens[0] == 'DATA':
                if tokens[1:] != ['ascii']:
                    raise ValueError('%s: 只支持 ASCII PCD' % path)
                break
        else:
            raise ValueError('%s: 找不到 DATA ascii 头' % path)
        if not all(axis in fields for axis in ['x', 'y', 'z']):
            raise ValueError('%s: PCD 缺少 x/y/z 字段' % path)
        points = np.loadtxt(stream, usecols=[fields.index(a) for a in ['x', 'y', 'z']],
                            ndmin=2)
    if points.size == 0:
        return np.empty((0, 3))
    return points[np.isfinite(points).all(axis=1)]


def normalize_pose(values):
    pose = np.asarray(values, dtype=float).copy()
    if pose.shape != (4,) or not np.isfinite(pose).all() or pose[3] <= 0:
        raise ValueError('x、y、heading 必须是有限数值，length 必须大于 0')
    pose[2] = (pose[2] + 180) % 360 - 180
    return pose


def pose_from_row(row, default_length):
    return normalize_pose([row['hook_x'], row['hook_y'], row['angle'],
                           row.get('length') or default_length])


def load_labels(path):
    if path is None:
        return {}, list(FIELDS)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'session', 'hook_x', 'hook_y', 'angle', 'length'}
        if not required.issubset(fields):
            raise ValueError('%s: CSV 必须包含 %s' % (path, ', '.join(sorted(required))))
        rows = {}
        for row in reader:
            name = row['session']
            if not name or name in rows:
                raise ValueError('%s: session 为空或重复：%s' % (path, name))
            pose_from_row(row, 1.8)
            rows[name] = row
    return rows, list(dict.fromkeys(fields + FIELDS))


class LabelStore:
    """Only explicitly confirmed sessions are persisted, using an atomic replace."""

    def __init__(self, output, initial_csv=None):
        self.output = Path(output)
        if initial_csv is not None and self.output.resolve() == Path(initial_csv).resolve():
            raise ValueError('输出文件必须与初值 CSV 不同，请保留原标签')
        self.initial, _ = load_labels(initial_csv)
        self.confirmed, self.fields = load_labels(self.output if self.output.exists() else None)

    def save(self, row):
        pose_from_row(row, 1.8)
        row = dict(row, trailer_type=row_type(row))
        previous = self.confirmed.get(row['session'], {})
        if (previous.get('session_dir') and row.get('session_dir') and
                Path(previous['session_dir']).resolve() != Path(row['session_dir']).resolve()):
            raise ValueError('同名采集段属于不同目录，请为每个数据集使用独立的输出 CSV')
        updated = dict(self.confirmed)
        updated[row['session']] = dict(row)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=str(self.output.parent),
                                             prefix=self.output.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=self.fields)
                writer.writeheader()
                for name in sorted(updated):
                    writer.writerow({key: updated[name].get(key, '') for key in self.fields})
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(self.output))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.confirmed = updated


class AnnotationTextBox(TextBox):
    """Avoid a synchronous figure repaint for every inactive input on each click."""

    def stop_typing(self):
        if self.capturekeystrokes:
            super().stop_typing()

    def set_display_value(self, value):
        value = str(value)
        if self.text == value:
            return
        if self.capturekeystrokes:
            self.set_val(value)
        else:
            self.text_disp.set_text(value)
            self.cursor_index = len(value)
            self.cursor.set_visible(False)


class PlotRenderer:
    """Cache the controls and point cloud separately; repaint only changed layers."""

    def __init__(self, figure, axes):
        self.figure, self.axes, self.canvas = figure, axes, figure.canvas
        self.artists = []
        self.ui_background = self.plot_background = self.ui_key = self.plot_key = None
        self.supports_blit = self.canvas.supports_blit
        self.interactive = hasattr(self.canvas, 'get_tk_widget')
        self.pending = self.full_pending = self.timer_running = False
        self.closed = False
        self.last_paint = 0.0
        self.interval = 1 / 60
        self.timer = self.canvas.new_timer(interval=16)
        self.timer.single_shot = True
        self.timer.add_callback(self.flush)
        self.draw_connection = self.canvas.mpl_connect('draw_event', self.on_draw)
        self.axes.set_animated(self.supports_blit)

    def set_artists(self, artists):
        for artist in self.artists:
            artist.set_animated(False)
        self.artists = artists
        for artist in artists:
            artist.set_animated(self.supports_blit)
        self.invalidate()

    def invalidate(self):
        self.ui_background = self.plot_background = self.ui_key = self.plot_key = None

    def view_key(self):
        return (tuple(self.figure.bbox.bounds), self.figure.dpi,
                tuple(self.axes.get_position(original=True).bounds),
                tuple(self.axes.bbox.bounds),
                tuple(self.axes.get_xlim()), tuple(self.axes.get_ylim()))

    def draw_overlays(self, renderer, figure_only=False):
        for artist in self.artists:
            if not figure_only or artist.axes is None:
                artist.draw(renderer)

    def on_draw(self, event):
        self.timer.stop()
        self.pending = self.full_pending = self.timer_running = False
        if not self.supports_blit:
            self.last_paint = time.perf_counter()
            return
        if self.canvas.is_saving():
            # Axes.draw includes animated children during export.
            self.invalidate()
            self.axes.draw(event.renderer)
            self.draw_overlays(event.renderer, figure_only=True)
        else:
            self.ui_background = self.canvas.copy_from_bbox(self.figure.bbox)
            self.ui_key = (tuple(self.figure.bbox.bounds), self.figure.dpi)
            self.axes.draw(event.renderer)
            self.plot_background = self.canvas.copy_from_bbox(self.figure.bbox)
            self.plot_key = self.view_key()
            self.draw_overlays(event.renderer)
        self.last_paint = time.perf_counter()

    def request(self, full=False, immediate=False):
        if self.closed:
            return
        self.pending = True
        self.full_pending |= full
        elapsed = time.perf_counter() - self.last_paint
        if immediate or not self.interactive or elapsed >= self.interval:
            self.flush()
        elif not self.timer_running:
            self.timer.interval = max(1, int((self.interval - elapsed) * 1000) + 1)
            self.timer_running = True
            self.timer.start()

    def flush(self):
        self.timer.stop()
        self.timer_running = False
        if self.closed or not self.pending:
            return
        self.pending = False
        if (self.full_pending or not self.supports_blit or self.ui_background is None
                or self.ui_key != (tuple(self.figure.bbox.bounds), self.figure.dpi)):
            self.full_pending = False
            self.canvas.draw()
            return
        renderer = self.canvas.get_renderer()
        if self.plot_background is None or self.plot_key != self.view_key():
            self.canvas.restore_region(self.ui_background)
            self.axes.draw(renderer)
            self.plot_background = self.canvas.copy_from_bbox(self.figure.bbox)
            self.plot_key = self.view_key()
        else:
            self.canvas.restore_region(self.plot_background)
        self.draw_overlays(renderer)
        self.canvas.blit(self.figure.bbox)
        self.last_paint = time.perf_counter()

    def close(self):
        self.closed = True
        self.pending = self.full_pending = self.timer_running = False
        self.timer.stop()
        self.canvas.mpl_disconnect(self.draw_connection)
        self.axes.set_animated(False)
        for artist in self.artists:
            artist.set_animated(False)
        self.invalidate()


class NativeTypeEntry:
    """Use Tk's actual text editor for IME and clipboard input on the Tk canvas."""

    def __init__(self, box, on_submit, on_focus, on_escape):
        import tkinter as tk

        self.box, self.on_submit = box, on_submit
        self.canvas = box.ax.figure.canvas.get_tk_widget()
        self.widget = tk.Entry(self.canvas, font=(box.text_disp.get_fontfamily()[0],
                                                  int(box.text_disp.get_fontsize())),
                               background='#f2f2f2', foreground='#202b35',
                               insertbackground='#202b35', relief='sunken', borderwidth=1,
                               exportselection=False)
        self.widget.insert(0, box.text)
        rect = box.ax.get_position()
        self.widget.place(relx=rect.x0, rely=1 - rect.y1,
                          relwidth=rect.width, relheight=rect.height)
        # All keyboard input for this field belongs to the native Entry.
        box.disconnect_events()
        self.widget.bind('<Return>', self.submit)
        self.widget.bind('<KP_Enter>', self.submit)
        self.widget.bind('<FocusOut>', self.submit)
        self.widget.bind('<FocusIn>', lambda event: on_focus())
        self.widget.bind('<Control-a>', self.select_all)
        self.widget.bind('<Control-A>', self.select_all)
        self.widget.bind('<Escape>', lambda event: self.escape(on_escape))

    def __getattr__(self, name):
        return getattr(self.box, name)

    @property
    def text(self):
        return self.widget.get()

    @property
    def capturekeystrokes(self):
        return self.widget.focus_get() == self.widget

    def set_val(self, value):
        value = str(value)
        if self.text != value:
            self.widget.delete(0, 'end')
            self.widget.insert(0, value)
        self.box.set_val(value)

    def set_display_value(self, value):
        value = str(value)
        if self.text != value:
            self.widget.delete(0, 'end')
            self.widget.insert(0, value)
        self.box.set_display_value(value)

    def begin_typing(self):
        self.widget.focus_set()

    def stop_typing(self):
        if self.capturekeystrokes:
            self.submit()
            self.canvas.focus_set()

    def submit(self, event=None):
        if self.widget.winfo_exists():
            self.on_submit(self.text)
        return 'break'

    def select_all(self, event=None):
        self.widget.selection_range(0, 'end')
        self.widget.icursor('end')
        return 'break'

    def escape(self, callback):
        self.stop_typing()
        callback()
        return 'break'


def configure_plotting():
    import matplotlib
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei',
                 'Microsoft YaHei', 'Noto Sans CJK JP']:
        if name in available:
            matplotlib.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    # These keys belong to the annotation controls rather than the figure toolbar.
    for key in ['keymap.quit', 'keymap.save', 'keymap.home', 'keymap.back', 'keymap.forward',
                'keymap.fullscreen']:
        matplotlib.rcParams[key] = []


class PoseAnnotator:
    def __init__(self, session_dirs, store, frames=5, default_length=1.8, start_session=None,
                 font_size=16, data_source=None, trailer_type=DEFAULT_TYPE):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.sessions = [Path(path) for path in session_dirs]
        self.store, self.frames, self.default_length = store, frames, default_length
        self.font_size = font_size
        self.default_type = self.active_type = normalize_type(trailer_type)
        self.trailer_type = self.default_type
        self.type_drafts, self.initial_types = {}, {}
        self.data_source = Path(data_source) if data_source is not None else (
            self.sessions[0].parent if len(self.sessions) > 1 else self.sessions[0])
        self.dataset_stores = {str(self.data_source.resolve()): store}
        self.dataset_stores.update({str(path.resolve()): store for path in self.sessions})
        self.index, self.state = None, None
        self.drafts, self.histories = {}, {}
        self.frame_overrides = {}
        if self.data_source.is_file():
            self.frame_overrides[str(self.data_source.parent.resolve())] = [self.data_source.resolve()]
        self.mode = 'position'
        self.session_filter, self.query, self.list_page = 'all', '', 0
        self.page_size, self.visible_indices = 8, []
        self.pan_origin = None
        self.zoom_origin, self.zoom_patch = None, None
        self.view_history = []
        self.dragging, self.syncing = None, False
        self.display_mode = None
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title('拖挂 x / y / heading 手工标注')
        self.ax = self.fig.add_axes([0.265, 0.27, 0.54, 0.48])
        self.fig.text(0.025, 0.955, '拖挂位姿手工标注', fontsize=font_size + 8, weight='bold')
        self.progress = self.fig.text(0.025, 0.925, '', fontsize=max(12, font_size - 2))
        self.source_text = self.fig.text(0.025, 0.895, '', fontsize=max(12, font_size - 3),
                                        color='#555555')
        self.live = self.fig.text(0.245, 0.16, '', fontsize=font_size + 1, weight='bold')
        self.state_label = self.fig.text(0.835, 0.785, '', fontsize=font_size, weight='bold')
        self.mode_label = self.fig.text(0.245, 0.785, '', fontsize=max(12, font_size - 1),
                                       color='#145b91', weight='bold')
        self.notice = self.fig.text(0.025, 0.045, '', fontsize=max(12, font_size - 2))
        self.output_text = self.fig.text(0.025, 0.012, '', fontsize=max(12, font_size - 4),
                                        color='#555555')
        self.fig.text(0.245, 0.08,
                      '左键位置 / 右键朝向 / Shift+左键框选 / 中键平移\n'
                      '滚轮缩放，Ctrl+滚轮微调角度    Enter 保存下一段    U 撤销',
                      fontsize=max(12, font_size - 3), linespacing=1.35)
        self.buttons, self.mode_buttons, self.filter_buttons = [], {}, {}

        def add_button(label, rect, callback, size=font_size):
            button = Button(self.fig.add_axes(rect), label, color='#edf2f7', hovercolor='#dce9f5')
            button.label.set_fontsize(size)
            button.on_clicked(callback)
            self.buttons.append(button)
            return button

        add_button('打开 PCD', [0.025, 0.82, 0.085, 0.055], self.open_pcd_dialog)
        add_button('选择目录', [0.12, 0.82, 0.085, 0.055], self.open_directory_dialog)
        self.search_box = AnnotationTextBox(self.fig.add_axes([0.075, 0.745, 0.095, 0.045]), '搜索 ', initial='')
        self.search_box.label.set_fontsize(max(12, font_size - 2))
        self.search_box.text_disp.set_fontsize(max(12, font_size - 2))
        self.search_box.on_text_change(self.search_changed)
        add_button('清空', [0.175, 0.745, 0.03, 0.045],
                   lambda _: self.search_box.set_val(''), max(12, font_size - 4))
        for i, (key, label) in enumerate([('all', '全部'), ('pending', '待确认'),
                                        ('confirmed', '已确认')]):
            self.filter_buttons[key] = add_button(label, [0.025 + i * 0.0625, 0.675, 0.055, 0.045],
                                                   lambda _, status=key: self.set_filter(status),
                                                   max(12, font_size - 2))
        self.list_ax = self.fig.add_axes([0.025, 0.245, 0.18, 0.42])
        self.list_ax.set_navigate(False)
        self.page_status = self.fig.text(0.025, 0.215, '', fontsize=max(12, font_size - 2))
        add_button('上一页', [0.025, 0.145, 0.085, 0.045], lambda _: self.turn_page(-1),
                   max(12, font_size - 2))
        add_button('下一页', [0.12, 0.145, 0.085, 0.045], lambda _: self.turn_page(1),
                   max(12, font_size - 2))
        add_button('恢复全图 F', [0.025, 0.08, 0.085, 0.045], self.fit_view,
                   max(12, font_size - 2))
        add_button('上一视图 V', [0.12, 0.08, 0.085, 0.045], self.previous_view,
                   max(12, font_size - 2))
        for i, (key, label) in enumerate([('position', '快捷标注  1'), ('heading', '调整朝向  2'),
                                        ('browse', '浏览点云  3'), ('zoom', '框选放大  4')]):
            self.mode_buttons[key] = add_button(label, [0.245 + i * 0.14, 0.82, 0.13, 0.055],
                                                 lambda _, mode=key: self.set_mode(mode))
        self.type_box = AnnotationTextBox(self.fig.add_axes([0.86, 0.82, 0.125, 0.055]), '车型 ',
                                initial=self.default_type)
        self.type_box.label.set_fontsize(font_size)
        self.type_box.text_disp.set_fontsize(font_size)
        self.type_box.on_submit(self.on_type)
        self.boxes = []
        for i, label in enumerate(['x (m)', 'y (m)', 'heading (°)', 'length (m)']):
            box = AnnotationTextBox(self.fig.add_axes([0.835, 0.67 - i * 0.105, 0.15, 0.057]),
                          label, initial='0')
            box.label.set_fontsize(font_size)
            box.label.set_position((0, 1.15))
            box.label.set_horizontalalignment('left')
            box.label.set_verticalalignment('bottom')
            box.text_disp.set_fontsize(font_size + 2)
            box.on_submit(lambda value, index=i: self.on_text(index, value))
            self.boxes.append(box)
        controls = [
            ('保存并下一段  Enter', [0.835, 0.275, 0.15, 0.055], self.save_next),
            ('上一段  B', [0.835, 0.21, 0.07, 0.055], lambda _: self.navigate(-1)),
            ('跳过  N', [0.915, 0.21, 0.07, 0.055], lambda _: self.navigate(1)),
            ('保存  S', [0.835, 0.145, 0.07, 0.055], self.save_current),
            ('撤销  U', [0.915, 0.145, 0.07, 0.055], self.undo),
            ('恢复本段初值  R', [0.835, 0.08, 0.15, 0.055], self.reset),
        ]
        for label, rect, callback in controls:
            add_button(label, rect, callback)
        if hasattr(self.fig.canvas, 'get_tk_widget'):
            self.type_box = NativeTypeEntry(self.type_box, self.on_type,
                                            self.begin_type_edit, self.annotation_mode)
        self.renderer = PlotRenderer(self.fig, self.ax)
        for name, callback in [('button_press_event', self.on_press),
                               ('motion_notify_event', self.on_motion),
                               ('button_release_event', self.on_release),
                               ('scroll_event', self.on_scroll),
                               ('resize_event', self.on_resize),
                               ('key_press_event', self.on_key),
                               ('close_event', self.on_close)]:
            self.fig.canvas.mpl_connect(name, callback)
        names = [path.name for path in self.sessions]
        if start_session is not None:
            if start_session not in names:
                raise ValueError('找不到采集段：%s' % start_session)
            start = names.index(start_session)
        else:
            start = next((i for i, name in enumerate(names) if name not in store.confirmed), 0)
        self.load_session(start)
        self.set_mode('position')

    @property
    def name(self):
        return self.sessions[self.index].name

    @property
    def session_id(self):
        return str(self.sessions[self.index].resolve())

    def choose_path(self, directory=False):
        import tkinter as tk
        from tkinter import filedialog

        parent = getattr(self.fig.canvas.manager, 'window', None)
        temporary = None
        if parent is None or not hasattr(parent, 'tk'):
            temporary = tk.Tk()
            temporary.withdraw()
            parent = temporary
        initial = self.data_source.parent if self.data_source.is_file() else self.data_source
        try:
            if directory:
                return filedialog.askdirectory(parent=parent, initialdir=str(initial),
                                               title='选择采集段目录或数据集根目录')
            return filedialog.askopenfilename(parent=parent, initialdir=str(initial),
                                               title='选择 PCD，预览其所在采集段',
                                               filetypes=[('PCD 点云', ('*.pcd', '*.PCD'))])
        finally:
            if temporary is not None:
                temporary.destroy()

    def open_pcd_dialog(self, event=None):
        self.open_dialog(False)

    def open_directory_dialog(self, event=None):
        self.open_dialog(True)

    def open_dialog(self, directory):
        from tkinter import TclError

        try:
            selected = self.choose_path(directory)
            if selected:
                self.load_source(Path(selected))
        except (OSError, ValueError, RuntimeError, TclError) as error:
            self.notice.set_text('打开失败：%s' % error)
            self.fig.canvas.draw_idle()

    def read_session_data(self, directory, shown=None):
        files = pcd_files(directory)
        if shown is None:
            shown = self.frame_overrides.get(str(directory.resolve()), files[:self.frames])
        clouds = [read_pcd_xyz(path) for path in shown]
        points = np.concatenate(clouds) if clouds else np.empty((0, 3))
        if not len(points):
            raise ValueError('%s 没有有效点，请检查 PCD' % directory)
        return files, shown, points

    def load_source(self, source):
        source = Path(source).expanduser().resolve()
        sessions = discover_sessions(source)
        folder = source.parent if source.is_file() else source
        known = [path.resolve() for path in self.sessions]
        if folder in known:
            index = known.index(folder)
            shown = [source] if source.is_file() else pcd_files(folder)[:self.frames]
            data = self.read_session_data(folder, shown)
            if source.is_file():
                self.frame_overrides[str(folder)] = shown
            else:
                self.frame_overrides.pop(str(folder), None)
            self.load_session(index, data)
        else:
            store = self.dataset_stores.get(str(source)) or self.dataset_stores.get(str(folder))
            if store is None:
                output, initial = default_label_paths(source)
                if initial is not None and not initial.is_file():
                    initial = None
                store = LabelStore(output, initial)
            index = next((i for i, path in enumerate(sessions)
                          if path.name not in store.confirmed), 0)
            shown = [source] if source.is_file() else pcd_files(sessions[index])[:self.frames]
            # Validate the new cloud before replacing the current dataset or store.
            data = self.read_session_data(sessions[index], shown)
            if isinstance(self.type_box, NativeTypeEntry):
                self.type_box.submit()
            self.drafts[self.session_id] = self.state.copy()
            self.type_drafts[self.session_id] = self.trailer_type
            self.sessions, self.store, self.data_source = sessions, store, source
            self.active_type = self.default_type
            self.dataset_stores[str(source)] = store
            self.dataset_stores.update({str(path.resolve()): store for path in sessions})
            self.index = None
            if source.is_file():
                self.frame_overrides[str(folder)] = shown
            else:
                for path in sessions:
                    self.frame_overrides.pop(str(path.resolve()), None)
            self.load_session(index, data)
        self.session_filter, self.query, self.list_page = 'all', '', 0
        self.search_box.set_val('')
        self.refresh_session_list(focus=True)
        self.set_mode('position')
        self.notice.set_text('已打开单帧预览，保存的标签应用于该文件所在整段。'
                             if source.is_file() else '已加载目录；选择左侧条目开始标注。')
        self.fig.canvas.draw_idle()

    def matching_indices(self):
        return [i for i, path in enumerate(self.sessions)
                if (self.query.casefold() in path.name.casefold()
                    or self.query.casefold() in self.session_type(path).casefold())
                and (self.session_filter == 'all'
                     or (path.name in self.store.confirmed) == (self.session_filter == 'confirmed'))]

    def refresh_session_list(self, focus=False):
        self.visible_indices = self.matching_indices()
        if focus and self.index in self.visible_indices:
            self.list_page = self.visible_indices.index(self.index) // self.page_size
        pages = max(1, (len(self.visible_indices) + self.page_size - 1) // self.page_size)
        self.list_page = min(max(0, self.list_page), pages - 1)
        self.list_ax.clear()
        self.list_ax.set(xlim=(0, 1), ylim=(0, self.page_size))
        self.list_ax.set_axis_off()
        self.list_ax.set_navigate(False)
        start = self.list_page * self.page_size
        for row, index in enumerate(self.visible_indices[start:start + self.page_size]):
            name = self.sessions[index].name
            confirmed = name in self.store.confirmed
            selected = index == self.index
            background = '#246da7' if selected else ('#f0f4f8' if row % 2 == 0 else '#fafcfe')
            y = self.page_size - row - 1
            self.list_ax.axhspan(y + 0.03, y + 0.97, color=background, linewidth=0)
            prefix = '[%s] ' % ('已' if confirmed else '待')
            text = self.list_ax.text(0.025, y + 0.5, prefix + name,
                                    fontsize=max(12, self.font_size - 2), va='center', clip_on=True,
                                    color='white' if selected else
                                    ('#237a3b' if confirmed else '#303b45'))
            renderer = self.fig.canvas.get_renderer()
            display, keep = name, len(name)
            while renderer.get_text_width_height_descent(
                    prefix + display, text.get_fontproperties(), False)[0] > self.list_ax.bbox.width * 0.94:
                keep -= 1
                if keep < 4:
                    display = '…'
                    break
                left = (keep - 1) // 2
                display = name[:left] + '…' + name[-(keep - 1 - left):]
            text.set_text(prefix + display)
        if self.visible_indices:
            self.page_status.set_text('第 %d / %d 页 · %d 段' % (
                self.list_page + 1, pages, len(self.visible_indices)))
        else:
            self.list_ax.text(0.03, self.page_size / 2, '没有匹配的采集段',
                              fontsize=max(12, self.font_size - 2))
            self.page_status.set_text('0 段，请调整搜索 / 筛选')
        for key, button in self.filter_buttons.items():
            self.style_button(button, key == self.session_filter)
        self.fig.canvas.draw_idle()

    def on_resize(self, event):
        if self.index is not None:
            self.refresh_session_list()

    def search_changed(self, value):
        self.query, self.list_page = value.strip(), 0
        if self.index is not None:
            self.refresh_session_list()

    def set_filter(self, status):
        self.session_filter, self.list_page = status, 0
        self.refresh_session_list()

    def turn_page(self, delta):
        self.list_page += delta
        self.refresh_session_list()

    @staticmethod
    def style_button(button, active):
        button.color = '#246da7' if active else '#edf2f7'
        button.hovercolor = '#1c5b8e' if active else '#dce9f5'
        button.ax.set_facecolor(button.color)
        button.label.set_color('white' if active else '#202b35')

    def update_mode_display(self):
        if self.display_mode == self.mode:
            return
        self.display_mode = self.mode
        names = {'position': '快捷标注：左键位置 / 右键朝向 / Shift 拖框',
                 'heading': '调整朝向：左键指向尾部 / 拖动',
                 'browse': '浏览点云：拖动平移 / 滚轮缩放',
                 'zoom': '框选放大：左键拖框 / 右键缩小 / 滚轮缩放'}
        self.mode_label.set_text('当前：' + names[self.mode])
        for key, button in self.mode_buttons.items():
            self.style_button(button, key == self.mode)
        self.renderer.invalidate()

    def set_mode(self, mode):
        if mode not in self.mode_buttons:
            raise ValueError('未知操作模式：%s' % mode)
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        self.release_toolbar()
        self.cancel_view_drag()
        self.mode, self.dragging = mode, None
        for box in self.boxes + [self.search_box, self.type_box]:
            box.stop_typing()
        if mode == 'browse' and toolbar:
            toolbar.pan()
        self.update_mode_display()
        self.notice.set_text({'position': '快捷标注：左键放挂钩，右键定朝向；Shift+拖框放大，滚轮缩放，中键拖动平移。',
                              'heading': '调整朝向模式：左键或右键指定尾部方向，滚轮微调角度。',
                              'browse': '浏览点云模式：拖动平移，滚轮缩放；按 4 框选放大，F 恢复全图。',
                              'zoom': '框选放大：左键拖出矩形；右键缩小、滚轮缩放；F 全图，V 上一视图。'}[mode])
        self.renderer.request(full=True)

    def cancel_view_drag(self):
        self.zoom_origin = None
        if self.zoom_patch is not None:
            self.zoom_patch.set_visible(False)
        if self.dragging in ['zoom', 'pan']:
            self.dragging = None

    def remember_view(self):
        view = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        if not self.view_history or not np.allclose(self.view_history[-1], view, rtol=0, atol=1e-12):
            self.view_history.append(view)
            del self.view_history[:-50]

    def previous_view(self, event=None):
        self.cancel_view_drag()
        current = (self.ax.get_xlim(), self.ax.get_ylim())
        while self.view_history:
            xlim, ylim = self.view_history.pop()
            if not np.allclose((xlim, ylim), current, rtol=0, atol=1e-12):
                self.ax.set(xlim=xlim, ylim=ylim)
                self.notice.set_text('已恢复上一视图；滚轮缩放，中键拖动平移。')
                self.renderer.request()
                return
        self.notice.set_text('当前采集段没有更早的视图；按 F 恢复全图。')
        self.renderer.request()

    def fit_view(self, event=None, remember=True):
        if remember:
            self.remember_view()
        self.cancel_view_drag()
        lower, upper = self.points[:, :2].min(axis=0), self.points[:, :2].max(axis=0)
        if (lower >= [-5.2, -1.2]).all() and (upper <= [0.2, 1.2]).all():
            self.ax.set(xlim=(-5.2, 0.2), ylim=(-1.2, 1.2))
        else:
            padding = np.maximum(upper - lower, 0.5) * 0.08
            self.ax.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]),
                        ylim=(lower[1] - padding[1], upper[1] + padding[1]))
        self.renderer.request()

    def scale_view(self, x, y, factor):
        self.remember_view()
        xlim, ylim = np.array(self.ax.get_xlim()), np.array(self.ax.get_ylim())
        self.ax.set(xlim=tuple(x + (xlim - x) * factor),
                    ylim=tuple(y + (ylim - y) * factor))
        self.renderer.request()

    def view_point(self, event):
        point = self.ax.transData.inverted().transform((event.x, event.y))
        if not np.isfinite(point).all():
            return None
        return np.clip(point, [min(self.ax.get_xlim()), min(self.ax.get_ylim())],
                       [max(self.ax.get_xlim()), max(self.ax.get_ylim())])

    def update_zoom_rectangle(self, point):
        lower, upper = np.minimum(self.zoom_origin, point), np.maximum(self.zoom_origin, point)
        self.zoom_patch.set_bounds(lower[0], lower[1], *(upper - lower))
        self.zoom_patch.set_visible(True)
        self.renderer.request()

    def finish_box_zoom(self, event):
        point, start = self.view_point(event), self.zoom_origin
        self.cancel_view_drag()
        if point is not None and start is not None:
            pixels = np.abs(self.ax.transData.transform(point) - self.ax.transData.transform(start))
            if np.all(pixels >= 5):
                self.remember_view()
                center, span = (start + point) / 2, np.abs(point - start)
                # Keep metres equally scaled on both axes and fill the point-cloud pane.
                rect = self.ax.get_position(original=True)
                ratio = rect.width * self.fig.get_figwidth() / (rect.height * self.fig.get_figheight())
                span = np.array([max(span[0], span[1] * ratio), max(span[1], span[0] / ratio)])
                self.ax.set(xlim=tuple(center[0] + np.array([-.5, .5]) * span[0]),
                            ylim=tuple(center[1] + np.array([-.5, .5]) * span[1]))
                self.notice.set_text('已放大框选区域；滚轮缩放，中键平移；F 全图，V 上一视图。')
        self.renderer.request(immediate=True)

    def initial_pose(self, name):
        row = self.store.confirmed.get(name) or self.store.initial.get(name)
        return pose_from_row(row, self.default_length) if row else np.array(
            [-2.2, 0.0, 0.0, self.default_length])

    def session_type(self, path):
        key = str(path.resolve())
        if self.index is not None and key == self.session_id:
            return self.trailer_type
        if key in self.type_drafts:
            return self.type_drafts[key]
        row = self.store.confirmed.get(path.name) or self.store.initial.get(path.name) or {}
        value = row_type(row)
        return value if value != DEFAULT_TYPE else self.active_type

    def initial_type(self):
        row = self.store.confirmed.get(self.name) or self.store.initial.get(self.name) or {}
        value = row_type(row)
        return value if value != DEFAULT_TYPE else self.initial_types.get(self.session_id, self.default_type)

    def current_is_saved(self):
        row = self.store.confirmed.get(self.name)
        return (row is not None and row_type(row) == self.trailer_type and np.allclose(
            self.state, pose_from_row(row, self.default_length), rtol=0, atol=1e-9))

    def load_session(self, index, data=None):
        directory = self.sessions[index]
        files, shown, points = self.read_session_data(directory) if data is None else data
        if self.index is not None:
            if isinstance(self.type_box, NativeTypeEntry):
                self.type_box.submit()
            self.drafts[self.session_id] = self.state.copy()
            self.type_drafts[self.session_id] = self.trailer_type
        self.index, self.points, self.n_frames = index, points, len(files)
        self.displayed_files = shown
        self.state = self.drafts.get(self.session_id, self.initial_pose(self.name)).copy()
        self.initial_types.setdefault(self.session_id, self.active_type)
        self.trailer_type = self.type_drafts.get(self.session_id, self.initial_type())
        self.histories.setdefault(self.session_id, [])
        self.dragging = None
        self.zoom_origin, self.view_history = None, []
        self.ax.clear()
        from matplotlib.patches import Rectangle
        self.zoom_patch = Rectangle((0, 0), 0, 0, linewidth=1.5, edgecolor='#145b91',
                                     facecolor=(.17, .55, .79, .12), visible=False, zorder=10)
        self.ax.add_patch(self.zoom_patch)
        if self.toolbar_active():
            self.ax.set_navigate_mode('PAN' if 'pan' in str(self.fig.canvas.toolbar.mode).lower()
                                      else 'ZOOM')
        self.ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.4,
                        color='#3678a8', rasterized=True)
        self.ax.set(xlim=(-5.2, 0.2), ylim=(-1.2, 1.2), xlabel='x (m)', ylabel='y (m)')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.tick_params(labelsize=max(12, self.font_size - 2))
        self.ax.xaxis.label.set_fontsize(self.font_size)
        self.ax.yaxis.label.set_fontsize(self.font_size)
        self.ax.grid(alpha=0.25)
        self.arrow = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                      arrowprops={'arrowstyle': '-|>', 'color': '#d53535', 'lw': 2})
        self.arrow.arrow_patch.set_clip_on(True)
        self.arrow.arrow_patch.set_clip_path(self.ax.patch)
        self.hook, = self.ax.plot([], [], 'o', color='#dc3131', markersize=10,
                                  markeredgecolor='white', zorder=5)
        self.tail, = self.ax.plot([], [], 'o', color='#1963bd', markersize=9,
                                  markeredgecolor='white', zorder=5)
        self.renderer.set_artists([self.arrow, self.hook, self.tail, self.zoom_patch,
                                   self.live, self.state_label, self.notice])
        self.notice.set_text('仅“保存”会确认本段；跳过不写入标签。关闭窗口前请保存当前修改。')
        self.fit_view(remember=False)
        self.redraw()
        self.refresh_session_list(focus=True)

    def redraw(self, sync=True, immediate=False):
        x, y, heading, length = self.state
        radians = np.radians(heading)
        back = np.array([x, y]) - length * np.array([np.cos(radians), np.sin(radians)])
        self.hook.set_data([x], [y])
        self.tail.set_data([back[0]], [back[1]])
        self.arrow.xy, self.arrow.xyann = (x, y), tuple(back)
        is_saved = self.current_is_saved()
        label = '已保存' if is_saved else '待确认 / 有修改'
        self.live.set_text('x = %.4f m     y = %.4f m     heading = %.3f°' % (x, y, heading))
        self.state_label.set_text(label)
        self.state_label.set_color('#237a3b' if is_saved else '#a64b0a')
        self.live.set_color('#237a3b' if is_saved else '#a64b0a')
        if sync:
            count = sum(path.name in self.store.confirmed for path in self.sessions)
            self.progress.set_text('%d / %d  %s    已确认 %d 段    显示 %d / %d 帧、%d 点' % (
                self.index + 1, len(self.sessions), self.name, count,
                len(self.displayed_files), self.n_frames, len(self.points)))
            selected = self.frame_overrides.get(self.session_id)
            source_folder = self.data_source.parent if self.data_source.is_file() else self.data_source
            self.source_text.set_text('数据：%s%s' % (source_folder,
                                     ' | 单帧预览：%s（整段标签）' % selected[0].name if selected else ''))
            self.output_text.set_text('输出：' + str(self.store.output))
            self.syncing = True
            try:
                for box, value in zip(self.boxes, self.state):
                    box.set_display_value('%.6f' % value)
                self.type_box.set_display_value(self.trailer_type)
            finally:
                self.syncing = False
        self.renderer.request(full=sync, immediate=immediate)

    def push_history(self):
        history = self.histories[self.session_id]
        history.append((self.state.copy(), self.trailer_type))
        del history[:-100]

    def change(self, pose, history=True, sync=True):
        pose = normalize_pose(pose)
        if np.array_equal(pose, self.state):
            return False
        if history:
            self.push_history()
        self.state = pose
        self.redraw(sync)
        return True

    def toolbar_active(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return bool(toolbar and toolbar.mode)

    def toolbar_view_mode(self):
        # Matplotlib calls pan mode "pan/zoom", so check for "pan" first.
        return 'browse' if 'pan' in str(self.fig.canvas.toolbar.mode).lower() else 'zoom'

    def release_toolbar(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar and toolbar.mode:
            if 'pan' in str(toolbar.mode).lower():
                toolbar.pan()
            elif 'zoom' in str(toolbar.mode).lower():
                toolbar.zoom()

    def annotation_mode(self, event=None):
        self.set_mode('position')
        self.notice.set_text('已恢复标注：左键位置、右键朝向，Shift+拖框放大；无需切换顶部模式。')
        self.renderer.request()

    def apply_pointer(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        pose = self.state.copy()
        if self.dragging == 'move':
            pose[:2] = [event.xdata, event.ydata]
        else:
            delta = pose[:2] - [event.xdata, event.ydata]
            if np.linalg.norm(delta) < 1e-6:
                return
            pose[2] = np.degrees(np.arctan2(delta[1], delta[0]))
        self.change(pose, history=False, sync=False)

    def on_press(self, event):
        if event.inaxes == self.list_ax and event.button == 1 and event.ydata is not None:
            row = int(self.page_size - event.ydata)
            position = self.list_page * self.page_size + row
            if 0 <= row < self.page_size and position < len(self.visible_indices):
                try:
                    self.load_session(self.visible_indices[position])
                except (OSError, ValueError) as error:
                    self.notice.set_text('读取失败：%s' % error)
                    self.fig.canvas.draw_idle()
            return
        if event.inaxes != self.ax or event.button not in [1, 2, 3]:
            return
        if event.button == 2:
            self.release_toolbar()
            self.cancel_view_drag()
            if event.dblclick:
                self.fit_view()
            else:
                self.remember_view()
                self.dragging = 'pan'
                self.pan_origin = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
                self.notice.set_text('中键拖动平移视图；松开后可继续左键位置、右键朝向。双击中键恢复全图。')
            self.renderer.request()
            return
        if event.button == 1 and event.dblclick and self.mode in ['browse', 'zoom']:
            self.fit_view()
            return
        if self.toolbar_active():
            self.remember_view()
            self.mode = self.toolbar_view_mode()
            self.dragging = None
            self.update_mode_display()
            self.notice.set_text('浏览点云中，标注暂停。按 Esc 恢复快捷标注，或按 1 / 2 切换。')
            self.renderer.request(full=True)
            return
        if self.mode == 'zoom' or (event.button == 1 and 'shift' in (event.key or '').split('+')):
            if event.button == 3:
                self.scale_view(event.xdata, event.ydata, 1.5)
            else:
                self.dragging = 'zoom'
                self.zoom_origin = np.array([event.xdata, event.ydata])
                self.update_zoom_rectangle(self.zoom_origin)
            return
        if self.mode == 'browse':
            self.remember_view()
            self.dragging = 'pan'
            self.pan_origin = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            self.renderer.request()
            return
        tail = np.array([self.tail.get_xdata()[0], self.tail.get_ydata()[0]])
        pixels = self.ax.transData.transform(tail)
        near_tail = np.linalg.norm(pixels - [event.x, event.y]) < 15
        rotate = self.mode == 'heading' or event.button == 3 or near_tail
        self.dragging = 'rotate' if rotate else 'move'
        self.push_history()
        self.apply_pointer(event)
        self.notice.set_text('旋转朝向：按住并拖动尾部手柄；也可右键单击尾部方向。'
                             if self.dragging == 'rotate' else
                             '移动挂钩：按住左键拖动，或在希望放置挂钩的目标位置单击。')
        self.renderer.request()

    def on_motion(self, event):
        if self.toolbar_active():
            self.cancel_view_drag()
            self.mode = self.toolbar_view_mode()
            self.dragging = None
            self.update_mode_display()
            return
        if self.dragging == 'zoom':
            point = self.view_point(event)
            if point is not None:
                self.update_zoom_rectangle(point)
        elif self.dragging == 'pan':
            x, y, xlim, ylim = self.pan_origin
            dx = (event.x - x) / self.ax.bbox.width * (xlim[1] - xlim[0])
            dy = (event.y - y) / self.ax.bbox.height * (ylim[1] - ylim[0])
            self.ax.set(xlim=(xlim[0] - dx, xlim[1] - dx),
                        ylim=(ylim[0] - dy, ylim[1] - dy))
            self.renderer.request()
        elif self.dragging:
            self.apply_pointer(event)

    def on_release(self, event):
        if self.dragging == 'zoom':
            self.finish_box_zoom(event)
        elif self.dragging:
            if self.dragging in ['move', 'rotate']:
                self.apply_pointer(event)
            elif self.dragging == 'pan':
                self.on_motion(event)
            self.dragging = None
            self.redraw(immediate=True)

    def on_scroll(self, event):
        if event.inaxes == self.list_ax:
            self.turn_page(-1 if event.step > 0 else 1)
            return
        if event.inaxes != self.ax:
            return
        modifiers = (event.key or '').split('+')
        fine_heading = ('control' in modifiers or 'ctrl' in modifiers)
        if (fine_heading and self.mode in ['position', 'heading'] and not self.toolbar_active()
                and self.dragging not in ['zoom', 'pan']):
            pose = self.state.copy()
            pose[2] += event.step * 0.1
            self.change(pose)
            return
        if self.toolbar_active():
            self.mode = self.toolbar_view_mode()
        if self.dragging in ['move', 'rotate']:
            self.redraw()
        self.cancel_view_drag()
        self.dragging = None
        self.update_mode_display()
        if event.xdata is None or event.ydata is None:
            return
        self.scale_view(event.xdata, event.ydata, 1.2 ** (-event.step))

    def on_text(self, index, value):
        if self.syncing or self.state is None:
            return
        try:
            pose = self.state.copy()
            pose[index] = float(value)
            if not self.change(pose):
                self.redraw()
        except ValueError as error:
            self.notice.set_text(str(error))
            self.redraw()

    def on_type(self, value):
        if self.syncing or self.state is None:
            return
        value = value.strip()
        if not value:
            self.notice.set_text('请输入车型名称，例如 cart_A；default 仅用于未补充车型的旧数据。')
            self.redraw()
            return
        if value != self.trailer_type:
            self.push_history()
            self.trailer_type = self.active_type = value
            self.type_drafts[self.session_id] = value
            self.redraw()
            self.refresh_session_list()

    def begin_type_edit(self):
        self.cancel_view_drag()
        self.dragging = None
        for box in self.boxes + [self.search_box]:
            box.stop_typing()
        self.notice.set_text('车型输入：Ctrl+A 全选，Ctrl+V 粘贴；回车应用，点击“保存”写入 CSV。')
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'escape':
            self.annotation_mode()
            return
        if any(box.capturekeystrokes for box in self.boxes + [self.search_box, self.type_box]):
            return
        key = event.key or ''
        if key in ['1', '2', '3', '4']:
            self.set_mode({'1': 'position', '2': 'heading', '3': 'browse', '4': 'zoom'}[key])
        elif key == 'f':
            self.fit_view()
        elif key == 'v':
            self.previous_view()
        elif key in ['enter', 'return']:
            self.save_next()
        elif key in ['s', 'ctrl+s', 'control+s']:
            self.save_current()
        elif key == 'n':
            self.navigate(1)
        elif key == 'b':
            self.navigate(-1)
        elif key in ['u', 'ctrl+z', 'control+z']:
            self.undo()
        elif key == 'r':
            self.reset()
        else:
            if self.mode in ['browse', 'zoom'] or self.dragging in ['pan', 'zoom']:
                return
            fine = 'shift+' in key or key in ['Q', 'E']
            base = key.replace('shift+', '').lower()
            pose = self.state.copy()
            distance, angle = (0.001, 0.1) if fine else (0.01, 1.0)
            if base in ['left', 'right']:
                pose[0] += distance * (-1 if base == 'left' else 1)
            elif base in ['down', 'up']:
                pose[1] += distance * (-1 if base == 'down' else 1)
            elif base in ['q', 'e']:
                pose[2] += angle * (-1 if base == 'q' else 1)
            else:
                return
            self.change(pose)

    def navigate(self, delta):
        matches = self.matching_indices()
        candidates = [index for index in matches if (index - self.index) * delta > 0]
        if not candidates:
            self.notice.set_text('已到筛选结果的首段 / 末段。可清空搜索或切换筛选，当前修改仍可保存。')
            self.fig.canvas.draw_idle()
            return
        index = candidates[0] if delta > 0 else candidates[-1]
        try:
            self.load_session(index)
        except (OSError, ValueError) as error:
            self.notice.set_text('读取失败：%s' % error)
            self.fig.canvas.draw_idle()

    def save_current(self, event=None):
        try:
            if isinstance(self.type_box, NativeTypeEntry):
                self.type_box.submit()
            x, y, heading, length = normalize_pose(self.state)
            row = dict(self.store.confirmed.get(self.name, {}))
            row.update(session=self.name, hook_x=float(x), hook_y=float(y), angle=float(heading),
                       length=float(length), n_frames=self.n_frames, n_points=len(self.points),
                       trailer_type=self.trailer_type, session_dir=self.session_id)
            radians = np.radians(heading)
            row.update(back_x=float(x - length * np.cos(radians)),
                       back_y=float(y - length * np.sin(radians)))
            self.store.save(row)
            self.active_type = self.trailer_type
            self.notice.set_text('已保存 %s；车型：%s。' % (self.name, self.trailer_type))
            self.redraw()
            self.refresh_session_list()
            return True
        except (OSError, ValueError) as error:
            self.notice.set_text('保存失败：%s' % error)
            self.fig.canvas.draw_idle()
            return False

    def save_next(self, event=None):
        if self.save_current():
            self.navigate(1)

    def undo(self, event=None):
        history = self.histories[self.session_id]
        if history:
            self.state, self.trailer_type = history.pop()
            self.type_drafts[self.session_id] = self.trailer_type
            self.redraw()
            self.refresh_session_list()

    def reset(self, event=None):
        self.push_history()
        self.state, self.trailer_type = self.initial_pose(self.name), self.initial_type()
        self.type_drafts[self.session_id] = self.trailer_type
        self.redraw()
        self.refresh_session_list()

    def on_close(self, event):
        self.renderer.close()
        if self.index is not None and not self.current_is_saved():
            print('当前段未保存的修改未写入 CSV；已确认标签已保存：%s' % self.store.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=DATA_ROOT / 'bag_combined',
                        help='数据集根目录、采集段目录或一个 PCD 文件')
    parser.add_argument('--initial_csv', type=Path, help='已有段级标签，仅作初值；默认使用 session_analysis.csv')
    parser.add_argument('--output', type=Path, help='输出 CSV；默认使用当前数据位置的 manual_pose_labels.csv')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--default_length', type=float, default=1.8)
    parser.add_argument('--trailer_type', default=DEFAULT_TYPE, help='未标注场景的默认车型名称')
    parser.add_argument('--font_size', type=int, default=16,
                        help='界面基础字号，默认 16；高分辨率屏幕可设为 20')
    parser.add_argument('--start_session', help='从指定段开始；默认从首个未确认段继续')
    parser.add_argument('--preview', type=Path, help='无窗口导出界面预览并退出，不确认或保存标签')
    args = parser.parse_args()
    if args.frames <= 0 or not np.isfinite(args.default_length) or args.default_length <= 0:
        parser.error('--frames 和 --default_length 必须为正数')
    if args.font_size < 10:
        parser.error('--font_size 不能小于 10')
    try:
        source = args.data_dir.expanduser().resolve()
        sessions = discover_sessions(source)
        default_output, default_initial = default_label_paths(source)
        output = args.output or default_output
        initial = args.initial_csv or default_initial
        if initial is not None and not initial.is_file() and args.initial_csv is None:
            initial = None
        store = LabelStore(output, initial)
        import matplotlib
        matplotlib.use('Agg' if args.preview else 'TkAgg')
        configure_plotting()
        app = PoseAnnotator(sessions, store, args.frames, args.default_length, args.start_session,
                            font_size=args.font_size, data_source=source, trailer_type=args.trailer_type)
        if args.preview:
            if args.preview.resolve() == output.resolve() or (
                    initial is not None and args.preview.resolve() == initial.resolve()):
                parser.error('预览图路径不能与标签 CSV 相同')
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            app.fig.savefig(str(args.preview), dpi=140)
            print('界面预览：%s（未写入标签）' % args.preview)
        else:
            import matplotlib.pyplot as plt
            plt.show()
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, '启动失败：%s\n' % error)


if __name__ == '__main__':
    main()
