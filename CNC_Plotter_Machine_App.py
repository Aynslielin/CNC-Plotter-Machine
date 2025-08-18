import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import os
import platform
import subprocess
from svgpathtools import svg2paths
import serial
import time
import threading
from concurrent.futures import ThreadPoolExecutor
# 去背
from rembg import remove
import io
#吉卜力（選用）
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None

# Logging 引入
import logging
import sys
import queue
from tkinter.constants import DISABLED, END, N, S, W, E
from tkinter.scrolledtext import ScrolledText

# 手動關閉 PyInstaller 啟動畫面
def close_splash():
    if hasattr(sys, '_MEIPASS'):
        try:
            import pyi_splash
            pyi_splash.close()
            logger.info("Splash screen closed manually")
        except ImportError:
            logger.warning("pyi_splash module not available, splash screen may not close properly")

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)  # 開發時輸出到控制台
    ]
)
logger = logging.getLogger(__name__)

__version__ = "1.0"  # 版本資訊

DRAW_MODES = [
    "輪廓描邊",
    "掃描填充",
    "Potrace SVG",
    "素描陰影填圖（Sketch Shading）",
    "層級線條陰影填圖（Layered Shading）"
]

# -------------------- 自訂 Logging Handler for Tkinter --------------------
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class ConsoleUi:
    def __init__(self, frame, log_queue):
        self.frame = frame
        self.log_queue = log_queue  # Store log_queue
        self.queue_handler = QueueHandler(self.log_queue)  # Create handler
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        self.scrolled_text = ScrolledText(frame, state=DISABLED, height=15, width=80)
        self.scrolled_text.pack(fill=tk.BOTH, expand=True)
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='black')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')
        self.scrolled_text.tag_config('CRITICAL', foreground='red', underline=1)
        self.frame.after(100, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state=DISABLED)
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get(block=False)
                self.display(record)
            except queue.Empty:
                break
        self.frame.after(100, self.poll_log_queue)

# -------------------- Path generation functions (unchanged) --------------------
# 輪廓描邊
def generate_paths_contour(img, scale, t1, t2, smooth_factor=0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, t1, t2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in contours:
        epsilon = smooth_factor * cv2.arcLength(cnt, True)
        smoothed = cv2.approxPolyDP(cnt, epsilon, True)
        if len(smoothed) < 5:
            continue
        path = [(pt[0][0] * scale, pt[0][1] * scale) for pt in smoothed]
        if path[0] != path[-1]:
            path.append(path[0])  # 回到起點
        paths.append(path)
    return paths

# 掃描填充
def generate_paths_hatch(img, scale, spacing=20, threshold=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape
    paths = []
    for y in range(0, h, spacing):
        in_line = False
        for x in range(w):
            if binary[y, x] == 255:
                if not in_line:
                    x_start = x
                    in_line = True
            else:
                if in_line:
                    x_end = x
                    paths.append([(x_start*scale, y*scale), (x_end*scale, y*scale)])
                    in_line = False
        if in_line:
            paths.append([(x_start*scale, y*scale), ((w-1)*scale, y*scale)])
    return paths

# Potrace 真實 (修改以支持打包)
def generate_paths_potrace_real(img, scale, threshold=127, temp_dir="temp", app=None):
    os.makedirs(temp_dir, exist_ok=True)
    bmp_path = os.path.join(temp_dir, "input.bmp")
    svg_path = os.path.join(temp_dir, "output.svg")

    # 動態獲取 Potrace 路徑
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        potrace_path = os.path.join(base_path, 'potrace', 'potrace.exe')
    else:
        potrace_path = 'potrace'

    # 檢查是否需要重新生成二值化圖像
    if app.last_potrace_threshold != threshold or app.last_binary_image is None or app.last_svg_paths is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        app.last_binary_image = binary
        app.last_potrace_threshold = threshold
        Image.fromarray(binary).save(bmp_path)
        try:
            logger.info(f"嘗試執行 Potrace: {potrace_path}")
            subprocess.run([potrace_path, "-s", bmp_path, "-o", svg_path], check=True)
            logger.info("Potrace 執行成功")
            app.last_svg_paths, _ = svg2paths(svg_path)
        except Exception as e:
            logger.error(f"Potrace 執行錯誤：{e}")
            return []
        finally:
            for path in [bmp_path, svg_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"無法刪除臨時檔案 {path}：{e}")
    else:
        logger.info("使用快取的二值化圖像和 SVG 路徑")

    paths = []
    potrace_scale_factor = scale * 0.1
    paths_data = app.last_svg_paths

    max_x, max_y = 0, 0
    all_points = []
    for path in paths_data:
        for seg in path:
            for t in np.linspace(0, 1, 3):
                pt = seg.point(t)
                all_points.append((pt.real, pt.imag))
    if all_points:
        max_x = max(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)

    for path in paths_data:
        sampled = []
        for seg in path:
            length = seg.length()
            num_samples = max(5, min(20, int(length / 10)))
            for t in np.linspace(0, 1, num_samples):
                pt = seg.point(t)
                fixed_x = pt.real
                fixed_y = max_y - pt.imag
                sampled.append((fixed_x * potrace_scale_factor, fixed_y * potrace_scale_factor))
        if len(sampled) >= 2:
            paths.append(sampled)

    return paths

# Sketch Shading
def generate_paths_sketch_shading(img, scale, block_size=40, spacing=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    paths = []
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            block = gray[y0:y0+block_size, x0:x0+block_size]
            if block.size == 0:
                continue
            avg = np.mean(block)
            directions = []
            if avg < 60:
                directions = ['h', 'v', 'd1', 'd2']
            elif avg < 100:
                directions = ['h', 'v', 'd1']
            elif avg < 150:
                directions = ['h', 'v']
            elif avg < 200:
                directions = ['h']
            for dir in directions:
                if dir == 'h':
                    for y in range(y0, y0 + block_size, spacing):
                        paths.append([(x0*scale, y*scale), ((x0+block_size)*scale, y*scale)])
                elif dir == 'v':
                    for x in range(x0, x0 + block_size, spacing):
                        paths.append([(x*scale, y0*scale), (x*scale, (y0+block_size)*scale)])
                elif dir == 'd1':
                    for d in range(-block_size, block_size, spacing):
                        points = []
                        for i in range(block_size):
                            x = x0 + i
                            y = y0 + i + d
                            if x0 <= x < x0+block_size and y0 <= y < y0+block_size:
                                points.append((x*scale, y*scale))
                        if len(points) >= 2:
                            paths.append(points)
                elif dir == 'd2':
                    for d in range(-block_size, block_size, spacing):
                        points = []
                        for i in range(block_size):
                            x = x0 + i
                            y = y0 + block_size - 1 - i + d
                            if x0 <= x < x0+block_size and y0 <= y < y0+block_size:
                                points.append((x*scale, y*scale))
                        if len(points) >= 2:
                            paths.append(points)
    return paths

# 層級線條陰影
def generate_paths_layered_shading(img, scale, thresholds=[50,100,150,200], spacings=[52,56,58,60],
                                 use_pos45=True, use_neg45=True, use_vert=True, use_horiz=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    paths = []

    for thresh, spacing in zip(thresholds, spacings):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        if use_horiz:
            for y in range(0, height, 1):
                if y % spacing != 0:
                    continue
                current_line = []
                for x in range(width):
                    if bw[y, x] == 255:
                        current_line.append((x*scale, y*scale))
                    elif current_line:
                        if len(current_line) > 1:
                            paths.append(current_line.copy())
                        current_line.clear()
                if current_line and len(current_line) > 1:
                    paths.append(current_line.copy())

        if use_vert:
            for x in range(0, width, 1):
                if x % spacing != 0:
                    continue
                current_line = []
                for y in range(height):
                    if bw[y, x] == 255:
                        current_line.append((x*scale, y*scale))
                    elif current_line:
                        if len(current_line) > 1:
                            paths.append(current_line.copy())
                        current_line.clear()
                if current_line and len(current_line) > 1:
                    paths.append(current_line.copy())

        if use_pos45:
            for d in range(-height, width):
                current_line = []
                for y in range(height):
                    x = d + y
                    if 0 <= x < width and bw[y, x] == 255 and (y - x) % spacing == 0:
                        current_line.append((x*scale, y*scale))
                    elif current_line:
                        if len(current_line) > 1:
                            paths.append(current_line.copy())
                        current_line.clear()
                if current_line and len(current_line) > 1:
                    paths.append(current_line.copy())

        if use_neg45:
            for d in range(0, height + width):
                current_line = []
                for y in range(height):
                    x = d - y
                    if 0 <= x < width and bw[y, x] == 255 and (y + x) % spacing == 0:
                        current_line.append((x*scale, y*scale))
                    elif current_line:
                        if len(current_line) > 1:
                            paths.append(current_line.copy())
                        current_line.clear()
                if current_line and len(current_line) > 1:
                    paths.append(current_line.copy())

    return paths

# -------------------- Preview helper --------------------
def preview_comparison(original_paths, optimized_paths, ax):
    """
    在同一張圖上比較原始路徑（灰）與優化後路徑（藍）。
    original_paths, optimized_paths: list of paths (each path = list of (x,y))
    """
    ax.clear()
    # 畫原始（灰）
    for path in original_paths:
        if len(path) < 2: 
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=0.4, color='dimgray', alpha=1)

    # 畫優化後（藍）
    for path in optimized_paths:
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=0.4, color='blue')

    ax.set_title("G-code Preview — Original(gray) vs Optimized(blue)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True)

def create_toggle_section(master, label_text):
    frame = tk.LabelFrame(master, text=label_text)
    return frame

# -------------------- Path optimization (unchanged) --------------------
def optimize_paths_greedy(paths, start_pos=(0.0, 0.0)):
    """
    使用貪婪最近鄰法對 paths（list of list of (x,y)）重新排序。
    若某段路徑的終點比較靠近當前位置，會將該段路徑反轉，使畫線從最近端開始。
    回傳新的 paths（淺拷貝）。
    """
    if not paths:
        return []

    # 建立一份可修改的索引列表
    remaining = list(range(len(paths)))
    optimized_order = []
    cur_x, cur_y = float(start_pos[0]), float(start_pos[1])

    # 為每個 path 計算 start 和 end
    starts = []
    ends = []
    for p in paths:
        starts.append(p[0])
        ends.append(p[-1])

    while remaining:
        best_idx = None
        best_dist = float('inf')
        best_reverse = False

        for idx in remaining:
            sx, sy = starts[idx]
            ex, ey = ends[idx]
            # 距離到起點與終點
            d_start = (cur_x - sx)**2 + (cur_y - sy)**2
            d_end = (cur_x - ex)**2 + (cur_y - ey)**2
            if d_start < best_dist:
                best_dist = d_start
                best_idx = idx
                best_reverse = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = idx
                best_reverse = True

        # 將選到的路徑加入序列，並視情況反轉
        chosen = paths[best_idx]
        if best_reverse:
            chosen = list(reversed(chosen))
        optimized_order.append(chosen)
        # 更新當前位置為新路徑的最後一點
        cur_x, cur_y = optimized_order[-1][-1]
        remaining.remove(best_idx)

    return optimized_order

# -------------------- Main App --------------------
class GCodeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("G-code Transform Tool (with Path Optimization)")
        
        self.last_potrace_threshold = None
        self.last_binary_image = None
        self.last_svg_paths = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.is_processing = False
        self.debounce_job = None
        self.debounce_delay = 300
        
        self.img = None
        self.captured_frame = None
        self.video = None

        self.scale = 0.3
        self.t1 = 100
        self.t2 = 200
        self.auto_scale = tk.BooleanVar(value=True)
        self.draw_mode = tk.StringVar(value=DRAW_MODES[0])
        self.max_width_mm = 240.0   # x最大寬度 (mm)
        self.max_height_mm = 160.0  # y最大高度 (mm)
        self.gemini_api_key = tk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))

        # 新增：下筆/抬筆命令變數
        self.pen_down_cmd = "M3 S090"
        self.pen_up_cmd = "M3 S060"

        # Logging queue 和 handler
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, padx=10, fill="y")
        preview_frame = tk.Frame(root)
        preview_frame.pack(side=tk.LEFT, padx=10)
        layered_param_frame = tk.Frame(root)
        layered_param_frame.pack(side=tk.LEFT, padx=10, fill="y")

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview_frame)
        self.canvas.get_tk_widget().pack()

        tk.Button(control_frame, text="選擇圖片", command=self.load_image).pack(pady=5)
        tk.Button(control_frame, text="開啟攝像頭", command=self.open_camera).pack(pady=5)
        tk.Button(control_frame, text="拍照並預覽", command=self.capture_and_preview).pack(pady=5)
        tk.Button(control_frame, text="轉換成吉卜力風格(Need Gemini API Key)", command=self.convert_to_ghibli).pack(pady=5)
        tk.Button(control_frame, text="移除背景", command=self.remove_background).pack(pady=5)
        tk.Button(control_frame, text="匯出 G-code", command=self.save_gcode).pack(pady=5)
        tk.Button(control_frame, text="🛠 開啟 GRBL 控制器", command=self.open_grbl_window).pack(pady=5)

        # 新增：下筆抬筆設定按鈕
        tk.Button(control_frame, text="下筆抬筆與範圍設定", command=self.open_pen_settings_window).pack(pady=5)

        # 新增：查看日誌按鈕
        tk.Button(control_frame, text="查看日誌", command=self.open_log_viewer).pack(pady=5)

        # 新增：關於按鈕
        tk.Button(control_frame, text="關於", command=self.show_about).pack(pady=5)

        tk.Label(control_frame, text="Gemini API Key").pack(pady=5)
        self.api_key_entry = tk.Entry(control_frame, textvariable=self.gemini_api_key, show="*")
        self.api_key_entry.pack(pady=5)
        tk.Button(control_frame, text="設定 API Key", command=self.set_gemini_api_key).pack(pady=5)
        
        tk.Label(control_frame, text="畫線方式").pack()
        mode_selector = ttk.Combobox(control_frame, values=DRAW_MODES, textvariable=self.draw_mode, state="readonly", width=25)
        mode_selector.pack()
        mode_selector.bind("<<ComboboxSelected>>", self.on_mode_change)

        self.mode_warning_label = tk.Label(control_frame, text="", wraplength=200, fg="red")
        self.mode_warning_label.pack(pady=5)

        self.contour_frame = create_toggle_section(control_frame, "輪廓描邊參數")
        self.hatch_frame = create_toggle_section(control_frame, "掃描填充參數")
        self.sketch_frame = create_toggle_section(control_frame, "素描陰影參數")
        self.potrace_frame = create_toggle_section(control_frame, "Potrace 門檻值參數")
        self.layered_frame = create_toggle_section(layered_param_frame, "層級線條陰影參數")

        tk.Label(self.hatch_frame, text="二值化門檻值").pack()
        self.hatch_threshold_slider = tk.Scale(self.hatch_frame, from_=0, to=255, orient="horizontal", resolution=1, command=self.update_preview)
        self.hatch_threshold_slider.set(127)
        self.hatch_threshold_slider.pack()

        # 新增：掃描填充的間距拉桿
        tk.Label(self.hatch_frame, text="線條間距").pack()
        self.hatch_spacing_slider = tk.Scale(self.hatch_frame, from_=1, to=100, orient="horizontal", resolution=1, command=self.update_preview)
        self.hatch_spacing_slider.set(20)  # 預設值與 generate_paths_hatch 一致
        self.hatch_spacing_slider.pack()

        tk.Label(self.potrace_frame, text="二值化門檻值").pack()
        self.potrace_threshold_slider = tk.Scale(self.potrace_frame, from_=0, to=255, orient="horizontal", resolution=1, command=self.update_preview)
        self.potrace_threshold_slider.set(127)
        self.potrace_threshold_slider.pack()

        tk.Label(self.contour_frame, text="邊緣門檻值1").pack()
        self.slider_t1 = tk.Scale(self.contour_frame, from_=0, to=255, orient="horizontal", command=self.update_preview)
        self.slider_t1.set(self.t1)
        self.slider_t1.pack()

        tk.Label(self.contour_frame, text="邊緣門檻值2").pack()
        self.slider_t2 = tk.Scale(self.contour_frame, from_=0, to=255, orient="horizontal", command=self.update_preview)
        self.slider_t2.set(self.t2)
        self.slider_t2.pack()

        self.block_size_var = tk.StringVar(value="40")
        tk.Label(self.sketch_frame, text="素描區塊大小(5~50)").pack()
        self.block_size_entry = tk.Entry(self.sketch_frame, textvariable=self.block_size_var, width=10)
        self.block_size_entry.pack()

        self.spacing_var = tk.StringVar(value="40")
        tk.Label(self.sketch_frame, text="素描線條間距(1~50)").pack()
        self.spacing_entry = tk.Entry(self.sketch_frame, textvariable=self.spacing_var, width=10)
        self.spacing_entry.pack()

        tk.Button(self.sketch_frame, text="確認參數設定", command=self.update_preview).pack(pady=5)

        self.threshold_vars = []
        self.spacing_vars = []
        default_thresholds = [50, 100, 150, 200]
        default_spacings = [52, 56, 58, 60]

        for i in range(4):
            tk.Label(self.layered_frame, text=f"層級 {i+1} 門檻值(0~255)").pack()
            var_t = tk.StringVar(value=str(default_thresholds[i]))
            entry_t = tk.Entry(self.layered_frame, textvariable=var_t, width=10)
            entry_t.pack()
            self.threshold_vars.append(var_t)

            tk.Label(self.layered_frame, text=f"層級 {i+1} 行距(0~60)").pack()
            var_s = tk.StringVar(value=str(default_spacings[i]))
            entry_s = tk.Entry(self.layered_frame, textvariable=var_s, width=10)
            entry_s.pack()
            self.spacing_vars.append(var_s)

        tk.Button(self.layered_frame, text="確認參數設定", command=self.update_preview).pack(pady=5)

        self.use_horiz = tk.BooleanVar(value=False)
        self.use_vert = tk.BooleanVar(value=False)
        self.use_pos45 = tk.BooleanVar(value=True)
        self.use_neg45 = tk.BooleanVar(value=True)

        tk.Checkbutton(self.layered_frame, text="水平線條", variable=self.use_horiz, command=self.update_preview).pack()
        tk.Checkbutton(self.layered_frame, text="垂直線條", variable=self.use_vert, command=self.update_preview).pack()
        tk.Checkbutton(self.layered_frame, text="正斜線（／）", variable=self.use_pos45, command=self.update_preview).pack()
        tk.Checkbutton(self.layered_frame, text="反斜線（\\）", variable=self.use_neg45, command=self.update_preview).pack()

        tk.Label(control_frame, text="縮放比例").pack()
        self.slider_scale = tk.Scale(control_frame, from_=0.001, to=0.5, resolution=0.001, orient="horizontal", command=self.update_preview)
        self.slider_scale.set(self.scale)
        self.slider_scale.pack()

        self.checkbox = tk.Checkbutton(control_frame, text="🔁 自動縮放至最大尺寸", variable=self.auto_scale, command=self.update_preview)
        self.checkbox.pack(pady=5)

        self.optimize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="啟用路徑優化 (匯出時)", variable=self.optimize_var).pack(pady=2)
        tk.Button(control_frame, text="預覽優化路徑", command=self.apply_optimize_and_preview).pack(pady=2)
        tk.Button(control_frame, text="預覽原始路徑", command=self.restore_original_paths).pack(pady=2)

        self.size_label = tk.Label(control_frame, text="目前輸出尺寸：-- mm x -- mm")
        self.size_label.pack(pady=5)

        self.current_paths = []
        self.current_paths_raw = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 初始化時呼叫 on_mode_change 以顯示預設模式的參數框架
        self.on_mode_change()

        # 檢查 Potrace 可用性
        self.check_potrace_availability()

        # 檢查 rembg
        if remove is None:
            logger.warning("rembg 模組未安裝，去背功能不可用")
            messagebox.showwarning("警告", "rembg 模組未安裝，去背功能不可用")
        

    def check_potrace_availability(self):
        try:
            if getattr(sys, 'frozen', False):
                potrace_path = os.path.join(sys._MEIPASS, 'potrace', 'potrace.exe')
            else:
                potrace_path = 'potrace'
            subprocess.run([potrace_path, "--version"], capture_output=True, check=True)
            logger.info("Potrace 可用")
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            logger.error(f"Potrace 不可用：{e}")
            self.mode_warning_label.config(text="Potrace 未安裝或路徑錯誤，'Potrace SVG' 模式不可用。請下載 potrace.exe 並放置於 dist/potrace 資料夾。", fg="red")
            # 禁用模式
            self.draw_mode.set(DRAW_MODES[0])  # 預設回輪廓描邊
            mode_selector = self.root.nametowidget(".!frame.!combobox")  # 假設 combobox 在 frame 中
            mode_selector['values'] = [m for m in DRAW_MODES if m != "Potrace SVG"]

    def open_log_viewer(self):
        log_window = tk.Toplevel(self.root)
        log_window.title("日誌查看器")
        ConsoleUi(log_window, self.log_queue)  # Pass log_queue to ConsoleUi

    def show_about(self):
        about_text = f"版本: {__version__}\n開發者: 林昕瑋、陳柔吟\n依賴: OpenCV, Tkinter, Matplotlib, NumPy, Pillow, svgpathtools, pyserial, rembg, google-generativeai\nPotrace: 需下載 exe 並包含在打包中\n如有問題，請檢查 app.log"
        messagebox.showinfo("關於", about_text)
        
    def on_mode_change(self, *args):
        mode = self.draw_mode.get()
        for frame in [self.contour_frame, self.hatch_frame, self.sketch_frame, self.potrace_frame]:
            frame.pack_forget()
        self.layered_frame.pack_forget()

        if mode == "輪廓描邊":
            self.contour_frame.pack(pady=5, fill="x")
            self.mode_warning_label.config(text="")
            self.update_preview()
        elif mode == "掃描填充":
            self.hatch_frame.pack(pady=5, fill="x")
            self.mode_warning_label.config(text="")
            self.update_preview()
        elif mode == "Potrace SVG":
            self.potrace_frame.pack(pady=5, fill="x")
            self.mode_warning_label.config(text="")
            self.update_preview()
        elif mode == "素描陰影填圖（Sketch Shading）":
            self.sketch_frame.pack(pady=5, fill="x")
            self.mode_warning_label.config(
                text="當切換到「素描陰影填圖」或「層級線條陰影填圖」模式時，預覽不會立即顯示，需點擊「確認參數設定」按鈕以使用當前輸入框中的參數生成路徑。按下按鈕後，運轉需要一段時間，請耐心等待，謝謝。"
            )
        elif mode == "層級線條陰影填圖（Layered Shading）":
            self.layered_frame.pack(in_=self.layered_frame.master, pady=5, fill="x")
            self.mode_warning_label.config(
                text="當切換到「素描陰影填圖」或「層級線條陰影填圖」模式時，預覽不會立即顯示，需點擊「確認參數設定」按鈕以使用當前輸入框中的參數生成路徑。按下按鈕或勾選後，運轉需要一段時間，請耐心等待，謝謝。"
            )

    def set_gemini_api_key(self):
        api_key = self.gemini_api_key.get().strip()
        if not api_key:
            messagebox.showwarning("無 API Key", "請輸入有效的 Gemini API Key")
            return
        messagebox.showinfo("成功", "Gemini API Key 已設定")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path:
            self.img = cv2.imread(path)
            # 重置 Potrace 快取
            self.last_potrace_threshold = None
            self.last_binary_image = None
            self.last_svg_paths = None
            self.update_preview()

    def open_camera(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            messagebox.showerror("錯誤", "無法開啟攝像頭")
            self.video = None
            return
        self.show_video_frame()

    def show_video_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.captured_frame = frame
            cv2.imshow("Camera", frame)
            self.root.after(30, self.show_video_frame)
        else:
            self.video.release()
            cv2.destroyAllWindows()

    def capture_and_preview(self):
        if self.video is None or not self.video.isOpened():
            messagebox.showwarning("無攝像頭", "請先開啟攝像頭")
            return
        if self.captured_frame is not None:
            self.img = self.captured_frame.copy()
            if self.video:
                self.video.release()
                cv2.destroyAllWindows()
            # 生成時間戳記作為預設檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"captured_{timestamp}.jpg"
            # 提示使用者選擇拍攝圖片的儲存位置
            filename = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG 檔案", "*.jpg")],
                title="儲存拍攝圖片",
                initialfile=default_filename
            )
            if not filename:
                return
            try:
                cv2.imwrite(filename, self.img)
                logger.info(f"[OK] 拍照並儲存為 {filename}")
                # 重置 Potrace 快取
                self.last_potrace_threshold = None
                self.last_binary_image = None
                self.last_svg_paths = None
                self.update_preview()
                self.show_image_in_window(filename, "拍攝圖片預覽")
            except Exception as e:
                logger.error(f"拍照儲存失敗：{e}")
                messagebox.showerror("錯誤", "拍照儲存失敗")
        
    def convert_to_ghibli(self):
        if self.img is None:
            messagebox.showwarning("無圖片", "請先載入或拍攝圖片")
            return
        if genai is None:
            messagebox.showwarning("缺少套件", "google.genai 套件未安裝或無法使用")
            return
        try:
            api_key = self.gemini_api_key.get().strip()
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("請輸入 Gemini API Key 或設置 GEMINI_API_KEY 環境變數")
            client = genai.Client(api_key=api_key)
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            from io import BytesIO
            temp_buffer = BytesIO()
            pil_img.save(temp_buffer, format="JPEG")
            image_data = temp_buffer.getvalue()
            prompt = "Transform this image into a Studio Ghibli-style illustration with vibrant colors, soft lighting, detailed backgrounds, and a whimsical, hand-drawn aesthetic."
            image_part = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            contents = [prompt, image_part]
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            if not hasattr(response, 'candidates') or not response.candidates:
                raise ValueError("API 回應中無有效內容")
            image_found = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    image_found = True
                    result_image_data = part.inline_data.data
                    result_image = Image.open(BytesIO(result_image_data))
                    # 生成時間戳記作為預設檔案名稱
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"ghibli_{timestamp}.jpg"
                    # 提示使用者選擇吉卜力風格圖片的儲存位置
                    filename = filedialog.asksaveasfilename(
                        defaultextension=".jpg",
                        filetypes=[("JPEG 檔案", "*.jpg")],
                        title="儲存吉卜力風格圖片",
                        initialfile=default_filename
                    )
                    if not filename:
                        return
                    result_image.save(filename)
                    logger.info(f"[OK] 吉卜力風格圖片儲存為 {filename}")
                    transformed_np = np.array(result_image)
                    transformed_cv = cv2.cvtColor(transformed_np, cv2.COLOR_RGB2BGR)
                    self.img = transformed_cv
                    # 重置 Potrace 快取
                    self.last_potrace_threshold = None
                    self.last_binary_image = None
                    self.last_svg_paths = None
                    self.update_preview()
                    self.show_image_in_window(filename, "吉卜力風格圖片預覽")
                    break
            if not image_found:
                raise ValueError("回應中未找到圖片內容")
        except Exception as e:
            error_str = str(e)
            logger.error(f"[!] 吉卜力風格轉換失敗：{e}")
            if "503 UNAVAILABLE" in error_str and "overloaded" in error_str:
                messagebox.showerror("模型過載", "Gemini API 模型目前負載過高，請稍後再試。\n\n提示：您可以在非尖峰時段再次嘗試，或等待幾分鐘後重試。")
            else:
                messagebox.showerror("錯誤", f"無法轉換為吉卜力風格：{e}")
                        
    def remove_background(self):
        if self.img is None:
            messagebox.showwarning("無圖片", "請先載入圖片或拍照")
            return
        image_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        try:
            bg_removed = remove(pil_image)
        except Exception as e:
            logger.error(f"[!] 去背失敗：{e}")
            messagebox.showerror("錯誤", "去背處理失敗")
            return
        bg_removed = bg_removed.convert("RGBA")
        white_bg = Image.new("RGBA", bg_removed.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, bg_removed)
        composite_rgb = composite.convert("RGB")
        open_cv_img = np.array(composite_rgb)
        open_cv_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2BGR)
        # 生成時間戳記作為預設檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"bg_removed_{timestamp}.jpg"
        # 提示使用者選擇去背圖片的儲存位置
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG 檔案", "*.jpg")],
            title="儲存去背圖片",
            initialfile=default_filename
        )
        if not filename:
            return
        cv2.imwrite(filename, open_cv_img)
        logger.info(f"[OK] 去背圖片儲存為 {filename}")
        self.img = open_cv_img
        # 重置 Potrace 快取
        self.last_potrace_threshold = None
        self.last_binary_image = None
        self.last_svg_paths = None
        self.update_preview()
        self.show_image_in_window(filename, "去背圖片預覽")
        
    def update_preview(self, *args):
        if self.debounce_job:
            self.root.after_cancel(self.debounce_job)
        self.debounce_job = self.root.after(self.debounce_delay, self._do_update_preview)

    def _do_update_preview(self):
        if self.img is None or self.is_processing:
            return
        self.is_processing = True
        try:
            self.t1 = self.slider_t1.get()
            self.t2 = self.slider_t2.get()
            scale = self.slider_scale.get()
            img_h, img_w = self.img.shape[:2]

            max_scale_x = self.max_width_mm / img_w
            max_scale_y = self.max_height_mm / img_h
            allowed_max_scale = min(max_scale_x, max_scale_y)

            if self.auto_scale.get():
                scale = allowed_max_scale
                self.slider_scale.set(scale)
            elif scale > allowed_max_scale:
                scale = allowed_max_scale
                self.slider_scale.set(scale)
                messagebox.showwarning("縮放限制", f"超出範圍，調整為 {scale:.2f}")

            self.scale = scale
            width_mm = img_w * scale
            height_mm = img_h * scale
            self.size_label.config(text=f"目前輸出尺寸：{width_mm:.1f} mm x {height_mm:.1f} mm")

            mode = self.draw_mode.get()
            for frame in [self.contour_frame, self.hatch_frame, self.sketch_frame, self.potrace_frame]:
                frame.pack_forget()
            self.layered_frame.pack_forget()

            if mode == "輪廓描邊":
                self.contour_frame.pack(pady=5, fill="x")
                self.current_paths = generate_paths_contour(self.img, scale, self.t1, self.t2)
                self.current_paths_raw = [list(p) for p in self.current_paths]
                preview_comparison(self.current_paths_raw, self.current_paths, self.ax)
                self.canvas.draw()
            elif mode == "掃描填充":
                self.hatch_frame.pack(pady=5, fill="x")
                hatch_threshold = self.hatch_threshold_slider.get()
                hatch_spacing = self.hatch_spacing_slider.get()
                if hatch_spacing <= 0:
                    messagebox.showwarning("無效輸入", "線條間距必須為正整數")
                    return
                self.current_paths = generate_paths_hatch(self.img, scale, spacing=hatch_spacing, threshold=hatch_threshold)
                self.current_paths_raw = [list(p) for p in self.current_paths]
                preview_comparison(self.current_paths_raw, self.current_paths, self.ax)
                self.canvas.draw()
            elif mode == "Potrace SVG":
                self.potrace_frame.pack(pady=5, fill="x")
                potrace_threshold = self.potrace_threshold_slider.get()
                self.executor.submit(self.async_generate_potrace, self.img, scale, potrace_threshold)
            elif mode == "素描陰影填圖（Sketch Shading）":
                self.sketch_frame.pack(pady=5, fill="x")
                try:
                    block_size = int(self.block_size_var.get())
                    spacing = int(self.spacing_var.get())
                    if block_size <= 0 or spacing <= 0:
                        messagebox.showwarning("無效輸入", "素描區塊大小和線條間距必須為正整數")
                        return
                    self.current_paths = generate_paths_sketch_shading(self.img, scale, block_size=block_size, spacing=spacing)
                    self.current_paths_raw = [list(p) for p in self.current_paths]
                    preview_comparison(self.current_paths_raw, self.current_paths, self.ax)
                    self.canvas.draw()
                except ValueError:
                    messagebox.showwarning("無效輸入", "請輸入有效的整數值")
                    return
            elif mode == "層級線條陰影填圖（Layered Shading）":
                self.layered_frame.pack(in_=self.layered_frame.master, pady=5, fill="x")
                try:
                    thresholds = [int(var.get()) for var in self.threshold_vars]
                    spacings = [int(var.get()) for var in self.spacing_vars]
                    if any(t < 0 for t in thresholds) or any(s <= 0 for s in spacings):
                        messagebox.showwarning("無效輸入", "門檻值必須為非負整數，行距必須為正整數")
                        return
                    use_p45 = self.use_pos45.get()
                    use_n45 = self.use_neg45.get()
                    use_v = self.use_vert.get()
                    use_h = self.use_horiz.get()
                    self.current_paths = generate_paths_layered_shading(
                        self.img, scale,
                        thresholds=thresholds,
                        spacings=spacings,
                        use_pos45=use_p45, use_neg45=use_n45,
                        use_vert=use_v, use_horiz=use_h
                    )
                    self.current_paths_raw = [list(p) for p in self.current_paths]
                    preview_comparison(self.current_paths_raw, self.current_paths, self.ax)
                    self.canvas.draw()
                except ValueError:
                    messagebox.showwarning("無效輸入", "請輸入有效的整數值")
                    return
        finally:
            self.is_processing = False
            self.debounce_job = None

    def async_generate_potrace(self, img, scale, threshold):
        try:
            paths = generate_paths_potrace_real(img, scale, threshold=threshold, app=self)
            self.current_paths = paths
            self.current_paths_raw = [list(p) for p in paths]
            self.root.after(0, lambda: self.update_canvas(paths))
        finally:
            self.is_processing = False

    def update_canvas(self, paths):
        preview_comparison(self.current_paths_raw, paths, self.ax)
        self.canvas.draw()

    def apply_optimize_and_preview(self):
        if not hasattr(self, 'current_paths_raw') or not self.current_paths_raw:
            messagebox.showwarning("無路徑", "目前沒有可以優化的路徑，請先產生路徑")
            return
        optimized = optimize_paths_greedy(self.current_paths_raw, start_pos=(0.0, 0.0))
        preview_comparison(self.current_paths_raw, optimized, self.ax)
        self.canvas.draw()
        self.current_paths = [list(p) for p in optimized]

    def restore_original_paths(self):
        if self.current_paths_raw is None:
            return
        self.current_paths = [list(p) for p in self.current_paths_raw]
        preview_comparison(self.current_paths_raw, [], self.ax)
        self.canvas.draw()

    def save_gcode(self):
        if not hasattr(self, 'current_paths') or self.img is None:
            return
        # 生成時間戳記作為預設檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_gcode_filename = f"gcode_{timestamp}.gcode"
        # 提示使用者選擇 G-code 檔案儲存位置
        filepath = filedialog.asksaveasfilename(
            defaultextension=".gcode",
            filetypes=[("G-code 檔案", "*.gcode")],
            title="儲存 G-code 檔案",
            initialfile=default_gcode_filename
        )
        if not filepath:
            return
        # 提示使用者選擇預覽圖片儲存位置
        default_preview_filename = f"gcode_{timestamp}_preview.png"
        preview_image_file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 檔案", "*.png")],
            title="儲存 G-code 預覽圖片",
            initialfile=default_preview_filename
        )
        if not preview_image_file:
            return
        paths_to_save = self.current_paths
        if self.optimize_var.get() and self.current_paths:
            paths_to_save = optimize_paths_greedy(self.current_paths, start_pos=(0.0, 0.0))
        with open(filepath, "w") as f:
            f.write("G21\nG90\nG92 X0 Y0\nG1 F500\n")
            for path in paths_to_save:
                if not path:
                    continue
                f.write(f"G0 X{path[0][0]:.2f} Y{path[0][1]:.2f}\n")
                f.write(f"{self.pen_down_cmd}\n")  # 使用自訂下筆命令
                for x, y in path[1:]:
                    f.write(f"G1 X{x:.2f} Y{y:.2f}\n")
                f.write(f"{self.pen_up_cmd}\n")  # 使用自訂抬筆命令
            f.write("G0 X0 Y0\n")
        self.fig.savefig(preview_image_file, dpi=300)
        logger.info(f"[OK] G-code 儲存：{filepath}")
        logger.info(f"[OK] 預覽圖儲存：{preview_image_file}")
        self.show_image_in_window(preview_image_file, "G-code 預覽圖片")
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", filepath.replace("/", "\\")])
            elif platform.system() == "Darwin":
                subprocess.run(["open", "-R", filepath])
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", os.path.dirname(filepath)])
        except Exception as e:
            logger.error(f"[!] 開啟資料夾失敗：{e}")
            
    def open_grbl_window(self):
        grbl_window = tk.Toplevel(self.root)
        grbl_app = GCodeSenderGUI(grbl_window, pen_down_cmd=self.pen_down_cmd, pen_up_cmd=self.pen_up_cmd)

    def on_closing(self):
        if self.video:
            self.video.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        os._exit(0)

    def show_image_in_window(self, image_path, title):
        preview_window = tk.Toplevel(self.root)
        preview_window.title(title)
        pil_img = Image.open(image_path)
        pil_img.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(pil_img)
        label = tk.Label(preview_window, image=photo)
        label.image = photo
        label.pack()

    # 新增：開啟下筆抬筆設定視窗
    def open_pen_settings_window(self):
        pen_window = tk.Toplevel(self.root)
        pen_window.title("下筆抬筆與範圍設定")

        # 下筆命令
        tk.Label(pen_window, text="下筆命令 (例如 M3 S090):").pack(pady=5)
        down_entry = tk.Entry(pen_window)
        down_entry.insert(0, self.pen_down_cmd)
        down_entry.pack(pady=5)

        # 抬筆命令
        tk.Label(pen_window, text="抬筆命令 (例如 M3 S060):").pack(pady=5)
        up_entry = tk.Entry(pen_window)
        up_entry.insert(0, self.pen_up_cmd)
        up_entry.pack(pady=5)

        # 新增：最大寬度設定
        tk.Label(pen_window, text="最大寬度 (mm, 例如 240.0):").pack(pady=5)
        max_width_entry = tk.Entry(pen_window)
        max_width_entry.insert(0, str(self.max_width_mm))
        max_width_entry.pack(pady=5)

        # 新增：最大高度設定
        tk.Label(pen_window, text="最大高度 (mm, 例如 160.0):").pack(pady=5)
        max_height_entry = tk.Entry(pen_window)
        max_height_entry.insert(0, str(self.max_height_mm))
        max_height_entry.pack(pady=5)

        def save_pen_settings():
            # 儲存下筆/抬筆命令
            new_down = down_entry.get().strip()
            new_up = up_entry.get().strip()
            if not new_down or not new_up:
                messagebox.showwarning("無效輸入", "請輸入有效的下筆/抬筆命令")
                return

            # 儲存最大寬度和高度
            try:
                new_max_width = float(max_width_entry.get().strip())
                new_max_height = float(max_height_entry.get().strip())
                if new_max_width <= 0 or new_max_height <= 0:
                    messagebox.showwarning("無效輸入", "最大寬度和高度必須為正數")
                    return
            except ValueError:
                messagebox.showwarning("無效輸入", "請輸入有效的數字作為最大寬度和高度")
                return

            # 更新屬性
            self.pen_down_cmd = new_down
            self.pen_up_cmd = new_up
            self.max_width_mm = new_max_width
            self.max_height_mm = new_max_height

            # 更新預覽以反映新的範圍限制
            self.update_preview()

            logger.info(f"更新範圍設定: max_width={new_max_width}, max_height={new_max_height}")
            messagebox.showinfo("成功", "下筆抬筆設定與範圍已更新")
            pen_window.destroy()

        tk.Button(pen_window, text="確認", command=save_pen_settings).pack(pady=5)

# -------------------- GRBL Sender classes --------------------
class GRBLSender:
    def __init__(self, port='COM5', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.gcode_lines = []
        self.current_line = 0
        self.is_sending = False
        self.is_paused = False
        self.stop_signal = False

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        self.ser.write(b"\x18")
        self.ser.flush()
        time.sleep(2)

        logger.info("正在等待 GRBL 啟動訊息...")
        init_message = ""
        for _ in range(20):
            init_response = self.ser.readline().decode(errors='ignore').strip()
            logger.info(f"<< {init_response}")
            if init_response.startswith('Grbl'):
                init_message = init_response
                logger.info("GRBL 連線成功")
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                return init_message
        raise Exception("未收到 GRBL 啟動訊息")

    def get_settings(self):
        if not self.ser or not self.ser.is_open:
            raise Exception("GRBL 未連線")
        self.ser.write(b"$$\n")
        self.ser.flush()
        settings = {}
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line:
                break
            if line.startswith('$'):
                parts = line.split('=')
                if len(parts) >= 2:
                    key = parts[0]
                    value_comment = '='.join(parts[1:])
                    value_parts = value_comment.split(' (')
                    value = value_parts[0]
                    comment = value_parts[1].rstrip(')') if len(value_parts) > 1 else ''
                    settings[key] = {'value': value, 'comment': comment}
        return settings

    def load_gcode(self, filepath):
        with open(filepath, 'r') as f:
            self.gcode_lines = [line.strip() for line in f if line.strip()]
        self.current_line = 0

    def send_gcode_thread(self, callback=None):
        def sender():
            self.is_sending = True
            self.stop_signal = False

            while self.ser.in_waiting:
                line = self.ser.readline()
                logger.info(f"<< 清除前殘留: {line.decode(errors='ignore').strip()}")

            if self.current_line < len(self.gcode_lines):
                line = self.gcode_lines[self.current_line]
                logger.info(f">> (預備第一行) {line}")
                self.ser.write((line + '\n').encode())
                self.ser.flush()

                while True:
                    response = self.ser.readline().decode(errors='ignore').strip()
                    if response:
                        logger.info(f"<< {response}")
                        if response.startswith('ok') or response.startswith('error'):
                            break
                    else:
                        time.sleep(0.01)

                self.current_line += 1
                if callback:
                    callback(self.current_line, len(self.gcode_lines))

                logger.info("[OK] 第一行已完成，準備傳輸後續")
                time.sleep(2.5)

            while self.current_line < len(self.gcode_lines) and not self.stop_signal:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                line = self.gcode_lines[self.current_line]
                logger.info(f">> {line}")
                self.ser.write((line + '\n').encode())
                self.ser.flush()

                while True:
                    response = self.ser.readline().decode(errors='ignore').strip()
                    if response:
                        logger.info(f"<< {response}")
                        if response.startswith('ok') or response.startswith('error'):
                            break
                    else:
                        time.sleep(0.01)

                self.current_line += 1
                if callback:
                    callback(self.current_line, len(self.gcode_lines))

            self.is_sending = False
            logger.info("傳輸完成或中止")

        threading.Thread(target=sender).start()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.stop_signal = True
        self.is_paused = False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

class GCodeSenderGUI:
    def __init__(self, root, pen_down_cmd="M3 S090", pen_up_cmd="M3 S060"):
        self.move_job = None
        self.move_direction = None
        self.sender = GRBLSender()
        self.root = root
        self.root.title("GRBL-Servo G-code 傳輸器")
        self.grbl_version = ""

        # 新增：儲存下筆/抬筆命令
        self.pen_down_cmd = pen_down_cmd
        self.pen_up_cmd = pen_up_cmd

        tk.Label(root, text="COM Port:").grid(row=0, column=0)
        self.port_entry = tk.Entry(root)
        self.port_entry.insert(0, "COM5")
        self.port_entry.grid(row=0, column=1)

        self.connect_btn = tk.Button(root, text="連接 GRBL", command=self.connect_grbl)
        self.connect_btn.grid(row=0, column=2)

        self.status_label = tk.Label(root, text="尚未連接", fg="gray")
        self.status_label.grid(row=0, column=3, padx=(10, 0))

        tk.Label(root, text="手動控制：").grid(row=4, column=0, pady=(10, 0))

        self.up_btn = tk.Button(root, text=f"抬筆 ({self.pen_up_cmd})", command=self.pen_up)
        self.up_btn.grid(row=5, column=0)

        self.down_btn = tk.Button(root, text=f"下筆 ({self.pen_down_cmd})", command=self.pen_down)
        self.down_btn.grid(row=5, column=1)

        self.reset_origin_btn = tk.Button(root, text="重設原點 (G92 X0 Y0)", command=self.reset_origin)
        self.reset_origin_btn.grid(row=5, column=3)

        self.home_btn = tk.Button(root, text="歸零 (G0 X0 Y0)", command=self.go_home)
        self.home_btn.grid(row=5, column=2)

        tk.Label(root, text="手動移動：").grid(row=6, column=0, pady=(10, 0))

        self.x_pos_btn = tk.Button(root, text="X+")
        self.x_pos_btn.grid(row=7, column=0)
        self.x_pos_btn.bind("<ButtonPress>", lambda e: self.start_move("X+"))
        self.x_pos_btn.bind("<ButtonRelease>", self.stop_move)

        self.x_neg_btn = tk.Button(root, text="X-")
        self.x_neg_btn.grid(row=7, column=1)
        self.x_neg_btn.bind("<ButtonPress>", lambda e: self.start_move("X-"))
        self.x_neg_btn.bind("<ButtonRelease>", self.stop_move)

        self.y_pos_btn = tk.Button(root, text="Y+")
        self.y_pos_btn.grid(row=7, column=2)
        self.y_pos_btn.bind("<ButtonPress>", lambda e: self.start_move("Y+"))
        self.y_pos_btn.bind("<ButtonRelease>", self.stop_move)

        self.y_neg_btn = tk.Button(root, text="Y-")
        self.y_neg_btn.grid(row=7, column=3)
        self.y_neg_btn.bind("<ButtonPress>", lambda e: self.start_move("Y-"))
        self.y_neg_btn.bind("<ButtonRelease>", self.stop_move)

        self.load_btn = tk.Button(root, text="載入 G-code", command=self.load_gcode)
        self.load_btn.grid(row=1, column=0)

        self.start_btn = tk.Button(root, text="開始傳送", command=self.start_send)
        self.start_btn.grid(row=1, column=1)

        self.pause_btn = tk.Button(root, text="暫停", command=self.pause_send)
        self.pause_btn.grid(row=1, column=2)

        self.resume_btn = tk.Button(root, text="繼續", command=self.resume_send)
        self.resume_btn.grid(row=1, column=3)

        self.stop_btn = tk.Button(root, text="停止", command=self.stop_send)
        self.stop_btn.grid(row=1, column=4)

        self.settings_btn = tk.Button(root, text="GRBL設定", command=self.open_settings_window)
        self.settings_btn.grid(row=2, column=0, pady=5)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=5, pady=10)

        self.progress_label = tk.Label(root, text="尚未開始")
        self.progress_label.grid(row=4, column=0, columnspan=5)

    def connect_grbl(self):
        try:
            self.sender.port = self.port_entry.get()
            self.grbl_version = self.sender.connect()
            messagebox.showinfo("成功", "GRBL 已連線")
            self.sender.ser.write(b"G92 X0 Y0\n")
            self.status_label.config(text="已連線", fg="green")
        except Exception as e:
            messagebox.showerror("錯誤", f"無法連線：{e}")
            self.status_label.config(text="連線失敗", fg="red")

    def open_settings_window(self):
        if not self.sender.ser or not self.sender.ser.is_open:
            messagebox.showwarning("未連接", "請先連接 GRBL")
            return
        settings_window = tk.Toplevel(self.root)
        GRBLSettingsGUI(settings_window, self.sender, self.grbl_version)

    def pen_up(self):
        if self.sender.ser and self.sender.ser.is_open:
            self.sender.ser.write((self.pen_up_cmd + "\n").encode())  # 使用自訂抬筆命令

    def pen_down(self):
        if self.sender.ser and self.sender.ser.is_open:
            self.sender.ser.write((self.pen_down_cmd + "\n").encode())  # 使用自訂下筆命令

    def reset_origin(self):
        if self.sender.ser and self.sender.ser.is_open:
            self.sender.ser.write(b"G92 X0 Y0\n")

    def go_home(self):
        if self.sender.ser and self.sender.ser.is_open:
            self.sender.ser.write(b"G0 X0 Y0\n")

    def repeat_move(self):
        if self.move_direction and self.sender.ser and self.sender.ser.is_open:
            cmd = {
                "X+": b"G91\nG0 X1\nG90\n",
                "X-": b"G91\nG0 X-1\nG90\n",
                "Y+": b"G91\nG0 Y1\nG90\n",
                "Y-": b"G91\nG0 Y-1\nG90\n"
            }.get(self.move_direction)
            if cmd:
                self.sender.ser.write(cmd)
                self.move_job = self.root.after(100, self.repeat_move)

    def start_move(self, direction):
        self.move_direction = direction
        self.repeat_move()

    def stop_move(self, event=None):
        if self.move_job:
            self.root.after_cancel(self.move_job)
            self.move_job = None
        self.move_direction = None

    def load_gcode(self):
        filepath = filedialog.askopenfilename(filetypes=[("G-code files", "*.gcode *.nc *.txt")])
        if filepath:
            self.sender.load_gcode(filepath)
            self.progress["maximum"] = len(self.sender.gcode_lines)
            self.progress_label.config(text=f"已載入 {len(self.sender.gcode_lines)} 行")

    def update_progress(self, current, total):
        self.progress["value"] = current
        if current >= total:
            self.progress_label.config(text="✅ 傳輸完成")
        else:
            self.progress_label.config(text=f"傳送中：{current} / {total} 行")

    def start_send(self):
        if not self.sender.gcode_lines:
            messagebox.showwarning("未載入", "請先載入 G-code 檔案")
            return
        self.sender.send_gcode_thread(callback=self.update_progress)

    def pause_send(self):
        self.sender.pause()
        self.progress_label.config(text="已暫停")

    def resume_send(self):
        self.sender.resume()
        self.progress_label.config(text="繼續傳送中")

    def stop_send(self):
        self.sender.stop()
        self.progress["value"] = 0
        self.progress_label.config(text="❌ 已中止並歸零")
        self.pen_up()
        self.go_home()

class GRBLSettingsGUI:
    def __init__(self, root, sender, grbl_version):
        self.sender = sender
        self.root = root
        self.root.title("GRBL 設定工具")
        self.grbl_version = grbl_version
        self.settings_entries = {}
        self.settings = {}

        tk.Label(root, text=f"GRBL 版本: {self.grbl_version}").grid(row=0, column=0, columnspan=2, pady=5)

        try:
            self.settings = self.sender.get_settings()
            row = 1
            for key, data in sorted(self.settings.items(), key=lambda x: int(x[0][1:])):
                label = tk.Label(root, text=f"{key} ({data['comment']}):")
                label.grid(row=row, column=0, sticky="w")
                entry = tk.Entry(root)
                entry.insert(0, data['value'])
                entry.grid(row=row, column=1)
                self.settings_entries[key] = entry
                row += 1
        except Exception as e:
            messagebox.showerror("錯誤", f"無法讀取設定：{e}")

        self.save_btn = tk.Button(root, text="儲存資料設定", command=self.save_settings)
        self.save_btn.grid(row=row, column=0, pady=5)

        self.load_btn = tk.Button(root, text="載入設定檔", command=self.load_settings_file)
        self.load_btn.grid(row=row, column=1, pady=5)

        self.send_btn = tk.Button(root, text="傳輸設定", command=self.send_settings)
        self.send_btn.grid(row=row+1, column=0, columnspan=2, pady=5)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
        self.progress.grid(row=row+2, column=0, columnspan=2, pady=10)

        self.progress_label = tk.Label(root, text="尚未開始")
        self.progress_label.grid(row=row+3, column=0, columnspan=2)

    def save_settings(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("G-code files", "*.gcode")], initialfile="grbl_settings.gcode")
        if not filepath:
            return
        with open(filepath, "w") as f:
            for key, entry in self.settings_entries.items():
                value = entry.get().strip()
                if value:
                    f.write(f"{key}={value}\n")
        messagebox.showinfo("成功", f"設定已儲存至 {filepath}")

    def load_settings_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("G-code files", "*.gcode *.nc *.txt")])
        if not filepath:
            return
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip().startswith('$')]
        for line in lines:
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0]
                value = parts[1]
                if key in self.settings_entries:
                    self.settings_entries[key].delete(0, tk.END)
                    self.settings_entries[key].insert(0, value)
        messagebox.showinfo("成功", "設定檔已載入至輸入框")

    def refresh_settings(self):
        try:
            self.settings = self.sender.get_settings()
            for key, entry in self.settings_entries.items():
                if key in self.settings:
                    entry.delete(0, tk.END)
                    entry.insert(0, self.settings[key]['value'])
            self.progress_label.config(text="✅ 傳輸並重新讀取完成")
        except Exception as e:
            self.progress_label.config(text="❌ 重新讀取設定失敗")
            messagebox.showerror("錯誤", f"無法重新讀取設定：{e}")

    def send_settings(self):
        if messagebox.askyesno("確認", "確定要傳輸這些設定給 GRBL 嗎？這可能改變機器行為。"):
            settings_lines = []
            for key, entry in self.settings_entries.items():
                value = entry.get().strip()
                if value and value != self.settings.get(key, {}).get('value', ''):
                    settings_lines.append(f"{key}={value}")
            if not settings_lines:
                messagebox.showinfo("無變更", "沒有變更的設定")
                return
            self.sender.gcode_lines = settings_lines
            self.sender.current_line = 0
            self.progress["maximum"] = len(settings_lines)
            self.sender.send_gcode_thread(callback=self.update_progress)

    def update_progress(self, current, total):
        self.progress["value"] = current
        if current >= total:
            self.progress_label.config(text="✅ 傳輸完成，正在重新讀取設定...")
            self.root.after(1000, self.refresh_settings)  # 延遲1秒後重新讀取，確保傳輸完成
        else:
            self.progress_label.config(text=f"傳送中：{current} / {total} 行")

# -------------------- 主程式入口 --------------------
if __name__ == "__main__":
    close_splash()
    root = tk.Tk()
    app = GCodeApp(root)
    root.mainloop()