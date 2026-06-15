# -*- coding: utf-8 -*-
"""
深度学习考试题库 —— 全中文互动学习工具
支持三种模式：学习模式 / 练习模式 / 考试模式
自动识别选择题、填空题、简答题、代码分析题
题目来源于 .md 题库文件
"""
import os
import sys
import re
import random
import json
import html as html_mod


import markdown
from PyQt5.QtWidgets import (
    QColorDialog,
    QApplication, QMainWindow, QSplitter, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QAction, QMessageBox,
    QToolBar, QPushButton, QLabel, QComboBox,
    QButtonGroup, QRadioButton, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView
# ==================== 数据结构 ====================
# ==================== 主题系统（继承自 MyFirstQt.py） ====================
THEMES = {
    "亮色": {
        "bg": "#f0f4f8", "card_bg": "#fff", "text": "#263238", "sub": "#546e7a",
        "border": "#e0e0e0", "accent": "#1a56db", "accent2": "#4caf50",
        "code_bg": "#e8eaed", "pre_bg": "#1e1e1e", "pre_text": "#d4d4d4",
        "table_h": "#e3f2fd", "block_bg": "#fff8e1", "block_border": "#ff9800",
    },
    "暗色": {
        "bg": "#1a1a2e", "card_bg": "#202040", "text": "#e0e0e0", "sub": "#a0a0b0",
        "border": "#333355", "accent": "#5b9df5", "accent2": "#66bb6a",
        "code_bg": "#2a2a4a", "pre_bg": "#0d0d1a", "pre_text": "#c8c8e0",
        "table_h": "#2a2a4a", "block_bg": "#2a2a1a", "block_border": "#cc8800",
    },
    "护眼": {
        "bg": "#f5f0e8", "card_bg": "#fefcf7", "text": "#4a3b2c", "sub": "#6d5d4b",
        "border": "#d4c5b0", "accent": "#6b4226", "accent2": "#4a8c3f",
        "code_bg": "#f0e6d3", "pre_bg": "#3d3226", "pre_text": "#e0d5c0",
        "table_h": "#f0e6d3", "block_bg": "#fcf3e0", "block_border": "#cc8800",
    },
}
# 存储用户自定义主题（背景色+文字色）
CUSTOM_THEME = {"bg": "", "text": ""}


def build_css(theme_name="亮色"):
    """根据主题名生成CSS（含动画）"""
    t = THEMES.get(theme_name, THEMES["亮色"])
    # 允许自定义颜色覆盖
    bg = CUSTOM_THEME["bg"] or t["bg"]
    text = CUSTOM_THEME["text"] or t["text"]
    return f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
    background: {bg}; color: {text}; padding: 16px 20px 40px; line-height: 1.75;
    transition: background 0.4s ease, color 0.4s ease;
}}
h1 {{ font-size: 1.7em; color: {t["accent"]}; margin-bottom: 14px; border-left: 4px solid {t["accent"]}; padding-left: 14px; animation: slideIn 0.4s ease; }}
h2 {{ font-size: 1.3em; color: {t["accent"]}; margin: 24px 0 10px; border-bottom: 1px solid {t["border"]}; padding-bottom: 6px; }}
h3 {{ font-size: 1.1em; color: {t["sub"]}; margin: 16px 0 6px; }}
p  {{ margin: 8px 0; }}
code {{ background: {t["code_bg"]}; color: #d63384; padding: 2px 6px; border-radius: 3px; font-family: "Fira Code","Cascadia Code",Consolas,monospace; font-size: 0.9em; }}
pre {{ background: {t["pre_bg"]}; color: {t["pre_text"]}; padding: 16px; border-radius: 10px; overflow-x: auto; margin: 10px 0; line-height: 1.5; }}
pre code {{ background: none; color: inherit; padding: 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid {t["border"]}; padding: 8px 12px; text-align: left; }}
th {{ background: {t["table_h"]}; color: {t["accent"]}; font-weight: 600; }}
tr:nth-child(even) {{ background: {t["card_bg"]}; }}
blockquote {{ border-left: 4px solid {t["block_border"]}; padding: 8px 16px; margin: 10px 0; background: {t["block_bg"]}; color: {text}; border-radius: 0 8px 8px 0; }}
img {{ max-width: 100%; border-radius: 8px; }}

/* === 动画（令人眼前一亮的效果） === */
@keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(18px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes slideIn   {{ from {{ opacity: 0; transform: translateX(-16px); }} to {{ opacity: 1; transform: translateX(0); }} }}
@keyframes shake     {{ 0%,100% {{ transform: translateX(0); }} 20%,60% {{ transform: translateX(-5px); }} 40%,80% {{ transform: translateX(5px); }} }}
@keyframes bounceIn  {{ 0% {{ transform: scale(0); opacity: 0; }} 60% {{ transform: scale(1.2); }} 100% {{ transform: scale(1); opacity: 1; }} }}
@keyframes glowPulse {{ 0%,100% {{ box-shadow: 0 0 4px {t["accent"]}33; }} 50% {{ box-shadow: 0 0 16px {t["accent"]}77; }} }}

/* === 题目卡片 === */
.question-card {{
    background: {t["card_bg"]}; border-radius: 16px; padding: 22px 24px;
    margin: 16px 0; box-shadow: 0 3px 16px rgba(0,0,0,0.06);
    border-left: 5px solid {t["accent"]}; transition: all 0.3s ease;
    animation: fadeInUp 0.4s ease both; position: relative;
}}
.question-card:nth-child(2) {{ animation-delay: 0.04s; }}
.question-card:nth-child(3) {{ animation-delay: 0.08s; }}
.question-card:nth-child(4) {{ animation-delay: 0.12s; }}
.question-card:nth-child(5) {{ animation-delay: 0.16s; }}
.question-card:hover {{ box-shadow: 0 6px 24px rgba(0,0,0,0.10); transform: translateY(-2px); }}
.question-card.correct {{ border-left-color: {t["accent2"]}; background: linear-gradient(to right, {t["card_bg"]} 70%, #e8f5e9); }}
.question-card.correct::after {{ content: '\\2713'; position: absolute; top: 12px; right: 18px; font-size: 2em; color: {t["accent2"]}; animation: bounceIn 0.5s ease; }}
.question-card.wrong   {{ border-left-color: #f44336; background: linear-gradient(to right, {t["card_bg"]} 70%, #ffebee); animation: shake 0.5s ease; }}
.question-card.wrong::after   {{ content: '\\2717'; position: absolute; top: 12px; right: 18px; font-size: 2em; color: #f44336; animation: bounceIn 0.5s ease; }}
.question-card.skipped {{ border-left-color: #ff9800; }}

.q-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
.q-badge  {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.78em; font-weight: 700; color: #fff; letter-spacing: 0.5px; }}
.q-badge.choice {{ background: linear-gradient(135deg, #1976d2, #42a5f5); }}
.q-badge.blank  {{ background: linear-gradient(135deg, #2e7d32, #66bb6a); }}
.q-badge.short  {{ background: linear-gradient(135deg, #e65100, #ff9800); }}
.q-badge.code   {{ background: linear-gradient(135deg, #6a1b9a, #ab47bc); }}
.q-badge.calc   {{ background: linear-gradient(135deg, #b71c1c, #ef5350); }}
.q-title {{ font-weight: 600; font-size: 1.02em; color: {text}; flex: 1; min-width: 200px; }}
.q-code-block {{ background: {t["pre_bg"]}; color: {t["pre_text"]}; padding: 14px 18px; border-radius: 10px; overflow-x: auto; margin: 10px 0; font-family: "Fira Code","Cascadia Code",monospace; font-size: 0.85em; line-height: 1.5; white-space: pre; }}

/* === 选项（涟漪动画） === */
.option-group {{ margin: 6px 0 12px; }}
.option-item {{
    display: block; padding: 12px 18px; margin: 7px 0;
    border: 2px solid {t["border"]}; border-radius: 12px;
    cursor: pointer; transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    font-size: 0.96em; position: relative; overflow: hidden;
}}
.option-item::before {{
    content: ''; position: absolute; top: 50%; left: 50%; width: 0; height: 0;
    border-radius: 50%; background: {t["accent"]}19;
    transform: translate(-50%,-50%); transition: width 0.6s, height 0.6s;
}}
.option-item:active::before {{ width: 300px; height: 300px; }}
.option-item:hover {{ border-color: {t["accent"]}88; background: {t["accent"]}0a; transform: translateX(4px); box-shadow: 0 2px 8px {t["accent"]}22; }}
.option-item input {{ display: none; }}
.option-item.selected {{ border-color: {t["accent"]}; background: linear-gradient(135deg, {t["accent"]}18, {t["accent"]}08); font-weight: 600; box-shadow: 0 0 10px {t["accent"]}33; }}
.option-item.reveal-correct {{ border-color: {t["accent2"]} !important; background: linear-gradient(135deg, #e8f5e9, #c8e6c9) !important; animation: glowPulse 1.5s ease infinite; }}
.option-item.reveal-wrong   {{ border-color: #f44336 !important; background: #ffebee !important; text-decoration: line-through; }}
.option-item.disabled {{ pointer-events: none; opacity: 0.85; }}
.option-label {{ display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%; background: {t["code_bg"]}; text-align: center; font-weight: 700; margin-right: 12px; color: {t["accent"]}; transition: all 0.25s; }}
.option-item.selected .option-label {{ background: {t["accent"]}; color: #fff; transform: scale(1.1); }}
.option-item.reveal-correct .option-label {{ background: {t["accent2"]} !important; color: #fff; }}

/* === 填空 === */
.blank-group {{ margin: 8px 0 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
.blank-input {{ width: 160px; padding: 9px 14px; border: 2px solid {t["border"]}; border-radius: 8px; font-size: 0.94em; transition: all 0.25s; outline: none; font-family: inherit; background: {t["card_bg"]}; color: {text}; }}
.blank-input:focus {{ border-color: {t["accent"]}; box-shadow: 0 0 0 3px {t["accent"]}26; }}
.blank-input.correct {{ border-color: {t["accent2"]} !important; background: #e8f5e9; animation: glowPulse 1.5s ease infinite; }}
.blank-input.wrong   {{ border-color: #f44336 !important; background: #ffebee; }}
.blank-label {{ display: inline-block; min-width: 28px; color: {t["sub"]}; font-size: 0.85em; font-weight: 600; }}

/* === 答案区 === */
.answer-box {{ margin-top: 16px; padding: 16px 20px; border-radius: 12px; display: none; transition: all 0.35s ease; }}
.answer-box.show {{ display: block; animation: fadeInUp 0.35s ease; }}
.answer-box.correct-answer {{ background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 1px solid #a5d6a7; }}
.answer-box.wrong-answer   {{ background: linear-gradient(135deg, #ffebee, #ffcdd2); border: 1px solid #ef9a9a; }}
.answer-box.normal-answer  {{ background: {t["card_bg"]}; border: 1px solid {t["border"]}; }}
.answer-label {{ font-weight: 700; margin-bottom: 8px; font-size: 1.02em; }}
.answer-label.correct {{ color: #2e7d32; }}
.answer-label.wrong   {{ color: #c62828; }}
.answer-text {{ color: {t["sub"]}; line-height: 1.8; }}

/* === 按钮 === */
.btn {{ display: inline-block; padding: 10px 24px; border: none; border-radius: 10px; font-size: 0.92em; font-weight: 600; cursor: pointer; transition: all 0.25s cubic-bezier(0.4,0,0.2,1); font-family: inherit; margin: 4px; position: relative; overflow: hidden; }}
.btn::after {{ content: ''; position: absolute; top: 50%; left: 50%; width: 0; height: 0; border-radius: 50%; background: rgba(255,255,255,0.3); transform: translate(-50%,-50%); transition: width 0.6s, height 0.6s; }}
.btn:active::after {{ width: 300px; height: 300px; }}
.btn-primary {{ background: linear-gradient(135deg, {t["accent"]}, #4285f4); color: #fff; box-shadow: 0 3px 10px {t["accent"]}4d; }}
.btn-primary:hover {{ box-shadow: 0 6px 20px {t["accent"]}73; transform: translateY(-1px); }}
.btn-success {{ background: linear-gradient(135deg, #2e7d32, #43a047); color: #fff; box-shadow: 0 3px 10px #2e7d324d; }}
.btn-success:hover {{ transform: translateY(-1px); }}
.btn-warning {{ background: linear-gradient(135deg, #e65100, #ff9800); color: #fff; }}
.btn-outline {{ background: {t["card_bg"]}; color: {t["accent"]}; border: 2px solid {t["accent"]}; }}
.btn-outline:hover {{ background: {t["accent"]}0d; }}
.btn-small {{ padding: 5px 14px; font-size: 0.8em; }}

/* === 计分栏（粘性定位） === */
.score-bar {{ position: sticky; top: 0; z-index: 100; background: {t["card_bg"]}; border-radius: 16px; padding: 16px 22px; margin-bottom: 18px; box-shadow: 0 3px 16px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
.score-stat {{ text-align: center; transition: transform 0.3s; }}
.score-stat:hover {{ transform: scale(1.08); }}
.score-num {{ font-size: 1.8em; font-weight: 800; background: linear-gradient(135deg, {t["accent"]}, #42a5f5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
.score-label {{ font-size: 0.72em; color: {t["sub"]}; letter-spacing: 1px; }}
.score-bar .btn {{ margin-left: auto; }}

/* === 搜索 === */
.search-box {{ width: 100%; padding: 8px 14px; border: 2px solid {t["border"]}; border-radius: 8px; font-size: 0.88em; outline: none; margin-bottom: 8px; transition: all 0.25s; font-family: inherit; background: {t["card_bg"]}; color: {text}; }}
.search-box:focus {{ border-color: {t["accent"]}; box-shadow: 0 0 0 3px {t["accent"]}1a; }}

@media (max-width: 700px) {{ body {{ padding: 10px; }} .question-card {{ padding: 14px; }} .blank-input {{ width: 110px; }} }}
"""

# 保留原静态CSS引用（兼容旧代码），实际渲染时由 build_css() 动态生成
CSS_STYLE = build_css("亮色")


# ==================== 公式保护 ====================
_MATH_PLACEHOLDERS = []

def _protect_math(text: str) -> str:
    """把 $$...$$ 和 $...$ 替换为占位符"""
    _MATH_PLACEHOLDERS.clear()
    def _repl(m):
        _MATH_PLACEHOLDERS.append(m.group(0))
        return '\x00MATH' + str(len(_MATH_PLACEHOLDERS)-1) + '\x00'
    text = re.sub(r'\$\$[\s\S]*?\$\$', _repl, text)
    text = re.sub(r'\$[^$\n]+?\$', _repl, text)
    return text

def _restore_math(html: str) -> str:
    """将占位符还原为原始公式"""
    for i, m in enumerate(_MATH_PLACEHOLDERS):
        html = html.replace('\x00MATH' + str(i) + '\x00', m)
    return html


# ==================== HTML 生成器 ====================
class ExamBank(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dir = os.getcwd()
        self.current_file = None
        
        
        
        
        self.current_theme = "亮色"

        self.initUI()
        self.load_md_list(self.current_dir)

    # ---------- UI 搭建 ----------
    def initUI(self):
        self.setWindowTitle("深度学习 考试题库 —— 互动练习系统")
        self.setGeometry(80, 60, 1400, 900)

        # 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        act_open = QAction("打开文件夹...", self)
        act_open.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open)
        act_file = QAction("打开单个文件...", self)
        act_file.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_file)
        file_menu.addSeparator()
        act_exit = QAction("退出", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        help_menu = menubar.addMenu("帮助")
        act_about = QAction("关于", self)
        act_about.triggered.connect(lambda: QMessageBox.about(self, "关于",
            "深度学习考试题库 v1.0\n\n"
            "支持三种模式：\n"
            "  📖 学习模式 — 阅读内容、折叠答案\n"
            "  ✍️ 练习模式 — 逐题作答、自动判对错\n"
            "  📝 考试模式 — 限时作答、统一评分\n\n"
            "选择题点击选项即可选择\n填空题在输入框中填写\n提交后自动显示正确答案"))
        help_menu.addAction(act_about)

        # 中央布局
        main_splitter = QSplitter(Qt.Horizontal)

        # === 左侧面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 8, 4, 8)

        

        # 主题选择（继承自 MyFirstQt.py 的三主题）
        theme_label = QLabel("主题：")
        theme_label.setStyleSheet("font-weight:600; font-size:13px; margin-top:6px;")
        left_layout.addWidget(theme_label)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        self.theme_combo.currentTextChanged.connect(self._on_theme_change)
        self.theme_combo.setStyleSheet("QComboBox{padding:4px 8px;border:1px solid #ddd;border-radius:6px;font-size:12px;}")
        left_layout.addWidget(self.theme_combo)

        # 自定义颜色按钮
        custom_layout = QHBoxLayout()
        btn_bg = QPushButton("自定义背景")
        btn_bg.setStyleSheet(self._btn_style())
        btn_bg.clicked.connect(self._custom_bg)
        btn_text = QPushButton("自定义文字")
        btn_text.setStyleSheet(self._btn_style())
        btn_text.clicked.connect(self._custom_text)
        btn_reset = QPushButton("重置")
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self._reset_theme)
        custom_layout.addWidget(btn_bg)
        custom_layout.addWidget(btn_text)
        custom_layout.addWidget(btn_reset)
        left_layout.addLayout(custom_layout)

        # 文件列表
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_clicked)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0; border-radius: 8px;
                background: #fafafa; font-size: 13px; padding: 4px;
            }
            QListWidget::item {
                padding: 6px 10px; border-radius: 4px; margin: 1px 0;
            }
            QListWidget::item:hover { background: #e3f2fd; }
            QListWidget::item:selected { background: #1a73e8; color: #fff; }
        """)
        left_layout.addWidget(self.file_list)

        # 底部操作
        bottom_btns = QHBoxLayout()
        btn_refresh = QPushButton("刷新列表")
        btn_refresh.clicked.connect(lambda: self.load_md_list(self.current_dir))
        btn_refresh.setStyleSheet(self._btn_style())
        bottom_btns.addWidget(btn_refresh)
        left_layout.addLayout(bottom_btns)

        main_splitter.addWidget(left_panel)

        # === 右侧 Web 预览 ===
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("border: none; background: #f0f2f5;")
        main_splitter.addWidget(self.web_view)

        main_splitter.setSizes([300, 1100])
        self.setCentralWidget(main_splitter)

        self.statusBar().showMessage("就绪 —— 选择左侧文件开始学习")

    def _btn_style(self):
        return """
            QPushButton {
                background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px;
                padding: 4px 12px; font-size: 12px; font-weight: 600;
            }
            QPushButton:hover { background: #e3f2fd; border-color: #90caf9; }
            QPushButton:checked { background: #1a73e8; color: #fff; border-color: #1a73e8; }
        """

    # ---------- MD 加载 ----------
    def load_md_list(self, directory):
        self.file_list.clear()
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "错误", f"文件夹不存在：{directory}")
            return
        self.current_dir = directory
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.md')])
        for f in files:
            item = QListWidgetItem(f)
            item.setData(Qt.UserRole, os.path.join(directory, f))
            self.file_list.addItem(item)
        if files:
            self.file_list.setCurrentRow(0)
            self.on_file_clicked(self.file_list.currentItem())

    def open_file_dialog(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择 Markdown 文件", self.current_dir, "Markdown (*.md);;所有文件 (*)")
        if fp:
            self.current_dir = os.path.dirname(fp)
            self.load_md_list(self.current_dir)
            self.load_and_render(fp)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", self.current_dir)
        if folder:
            self.load_md_list(folder)

    def on_file_clicked(self, item):
        md_path = item.data(Qt.UserRole)
        self.load_and_render(md_path)

    def load_and_render(self, md_path):
        """读取文件、解析、渲染"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
        except Exception as e:
            self.web_view.setHtml(f"<p style='color:red;padding:20px;'>读取失败：{e}</p>")
            return

        self.current_file = md_path

        # 保护公式 → Markdown渲染 → 还原公式
        protected = _protect_math(md_text)
        html_body = markdown.markdown(protected, extensions=['extra', 'toc', 'nl2br', 'fenced_code', 'codehilite'])
        html_body = _restore_math(html_body)
        css = build_css(self.current_theme)
        full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$']]}},svg:{{fontCache:'global'}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs2015.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>document.addEventListener('DOMContentLoaded',function(){{hljs.highlightAll();}});</script>
</head><body>{html_body}</body></html>"""
        self.web_view.setHtml(full, QUrl.fromLocalFile(os.path.dirname(md_path) + '/'))
        self.statusBar().showMessage(f"{os.path.basename(md_path)} | {self.current_theme}主题")

    def _on_theme_change(self, name):
        """主题下拉框切换"""
        if name not in THEMES:
            return
        self.current_theme = name
        CUSTOM_THEME["bg"] = ""
        CUSTOM_THEME["text"] = ""
        if self.current_file:
            self.load_and_render(self.current_file)

    def _custom_bg(self):
        """自定义背景颜色（继承自 MyFirstQt.py 的 QColorDialog 功能）"""
        color = QColorDialog.getColor()
        if color.isValid():
            CUSTOM_THEME["bg"] = color.name()
            if self.current_file:
                self.load_and_render(self.current_file)
            self.statusBar().showMessage(f"自定义背景: {color.name()}")

    def _custom_text(self):
        """自定义文字颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            CUSTOM_THEME["text"] = color.name()
            if self.current_file:
                self.load_and_render(self.current_file)
            self.statusBar().showMessage(f"自定义文字: {color.name()}")

    def _reset_theme(self):
        """重置自定义颜色为当前主题默认值"""
        CUSTOM_THEME["bg"] = ""
        CUSTOM_THEME["text"] = ""
        if self.current_file:
            self.load_and_render(self.current_file)
        self.statusBar().showMessage(f"已重置为 {self.current_theme} 默认色")


# ==================== 启动入口 ====================
def main():
    # 设置 Qt 平台插件路径（解决 Windows 平台插件找不到的问题）
    import PyQt5
    pyqt5_dir = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(pyqt5_dir, 'Qt', 'plugins')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))
    app.setStyle("Fusion")

    # 全局调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f5f5f5"))
    palette.setColor(QPalette.WindowText, QColor("#333333"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f0f2f5"))
    palette.setColor(QPalette.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText, QColor("#333333"))
    palette.setColor(QPalette.Highlight, QColor("#1a73e8"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = ExamBank()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
