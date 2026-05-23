# -*- coding: utf-8 -*-
"""
优雅 Markdown 阅读器 —— 原版保留
功能：主题切换（亮色/暗色/护眼）、自定义颜色、文件夹浏览
"""
import os
import sys
import markdown
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSplitter,
                             QListWidget, QListWidgetItem, QVBoxLayout,
                             QWidget, QFileDialog, QAction, QColorDialog,
                             QMessageBox, QToolBar, QComboBox)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView


# ---------- 预设主题 ----------
THEMES = {
    "亮色 (浅灰)": {
        "body_bg": "#ffffff",
        "text_color": "#333333",
        "code_bg": "#f6f8fa",
        "border_color": "#e1e4e8",
        "link_color": "#0366d6",
        "blockquote_border": "#dfe2e5",
        "table_header_bg": "#f2f2f2"
    },
    "暗色 (深邃)": {
        "body_bg": "#1e1e1e",
        "text_color": "#e0e0e0",
        "code_bg": "#2d2d2d",
        "border_color": "#3c3c3c",
        "link_color": "#79c0ff",
        "blockquote_border": "#3c3c3c",
        "table_header_bg": "#2d2d2d"
    },
    "护眼 (米黄)": {
        "body_bg": "#faf0e6",
        "text_color": "#4a3b2c",
        "code_bg": "#f5e6d3",
        "border_color": "#d4c5b0",
        "link_color": "#8b5a2b",
        "blockquote_border": "#d4c5b0",
        "table_header_bg": "#e6d6c0"
    }
}

def generate_css(theme):
    """根据主题字典生成 CSS 样式"""
    return f"""
    body {{
        background-color: {theme["body_bg"]};
        color: {theme["text_color"]};
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        max-width: 1000px;
        margin: 2rem auto;
        padding: 0 20px;
        line-height: 1.6;
    }}
    h1, h2, h3 {{
        border-bottom: 1px solid {theme["border_color"]};
        padding-bottom: 0.3em;
    }}
    code {{
        background-color: {theme["code_bg"]};
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: 'Fira Code', 'Cascadia Code', monospace;
        color: {theme["text_color"]};
    }}
    pre {{
        background-color: {theme["code_bg"]};
        padding: 1em;
        border-radius: 8px;
        overflow-x: auto;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}
    a {{
        color: {theme["link_color"]};
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    blockquote {{
        border-left: 4px solid {theme["blockquote_border"]};
        margin: 0;
        padding-left: 1em;
        color: {theme["text_color"]};
        opacity: 0.8;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 1px solid {theme["border_color"]};
        padding: 8px;
    }}
    th {{
        background-color: {theme["table_header_bg"]};
    }}
    img {{
        max-width: 100%;
    }}
    """

# HTML 模板（包含 MathJax + highlight.js）
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Markdown Preview</title>
    <style>{style}</style>
    <!-- highlight.js 代码高亮 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <!-- MathJax 数学公式 -->
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }},
            svg: {{ fontCache: 'global' }}
        }};
    </script>
</head>
<body>
{content}
</body>
</html>
"""


class MarkdownReader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme_name = "亮色 (浅灰)"  # 默认主题
        self.current_dir = os.getcwd()           # 当前浏览的文件夹
        self.initUI()
        self.load_md_list(self.current_dir)

    def initUI(self):
        self.setWindowTitle("📚 优雅 Markdown 阅读器")
        self.setGeometry(100, 100, 1300, 850)

        # ---------- 菜单栏 ----------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")

        open_file_action = QAction("打开文件...", self)
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

        open_folder_action = QAction("打开文件夹...", self)
        open_folder_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 主题菜单
        theme_menu = menubar.addMenu("主题")

        # 预设主题
        for theme_name in THEMES.keys():
            action = QAction(theme_name, self)
            action.triggered.connect(lambda checked, name=theme_name: self.apply_theme(name))
            theme_menu.addAction(action)

        theme_menu.addSeparator()
        custom_bg_action = QAction("自定义背景颜色...", self)
        custom_bg_action.triggered.connect(self.custom_bg_color)
        theme_menu.addAction(custom_bg_action)

        custom_text_action = QAction("自定义文字颜色...", self)
        custom_text_action.triggered.connect(self.custom_text_color)
        theme_menu.addAction(custom_text_action)

        # 工具栏（快速主题切换）
        toolbar = QToolBar("工具栏")
        self.addToolBar(toolbar)
        theme_combo = QComboBox()
        theme_combo.addItems(list(THEMES.keys()))
        theme_combo.currentTextChanged.connect(self.apply_theme)
        toolbar.addWidget(theme_combo)

        # ---------- 中央布局：左侧文件列表，右侧预览 ----------
        splitter = QSplitter(Qt.Horizontal)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_clicked)
        splitter.addWidget(self.file_list)

        self.web_view = QWebEngineView()
        splitter.addWidget(self.web_view)

        splitter.setSizes([300, 1000])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("就绪")

    def load_md_list(self, directory):
        """加载指定目录下所有 .md 文件"""
        self.file_list.clear()
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "错误", f"文件夹不存在：{directory}")
            return
        self.current_dir = directory
        files = [f for f in os.listdir(directory) if f.lower().endswith('.md')]
        files.sort()
        for f in files:
            item = QListWidgetItem(f)
            item.setData(Qt.UserRole, os.path.join(directory, f))
            self.file_list.addItem(item)
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
            self.on_file_clicked(self.file_list.currentItem())
        else:
            self.web_view.setHtml("<p>当前文件夹没有 .md 文件，请通过「打开文件夹」选择其他目录。</p>")

    def open_file_dialog(self):
        """打开单个 Markdown 文件（不依赖左侧列表）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Markdown 文件", self.current_dir, "Markdown (*.md);;所有文件 (*)"
        )
        if file_path:
            self.render_md_file(file_path)
            # 可选：将当前目录切换到文件所在目录
            self.current_dir = os.path.dirname(file_path)
            self.load_md_list(self.current_dir)   # 刷新左侧列表

    def open_folder_dialog(self):
        """打开文件夹并列出其中的 .md 文件"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", self.current_dir)
        if folder:
            self.load_md_list(folder)

    def render_md_file(self, md_path):
        """直接渲染一个 .md 文件（不在左侧列表时也可以用）"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            self.web_view.setHtml(f"<p>读取失败：{e}</p>")
            return
        html_body = markdown.markdown(md_content, extensions=['extra', 'toc', 'nl2br'])
        current_css = generate_css(THEMES[self.current_theme_name])
        full_html = HTML_TEMPLATE.format(style=current_css, content=html_body)
        self.web_view.setHtml(full_html, QUrl.fromLocalFile(os.path.dirname(md_path) + '/'))
        self.statusBar().showMessage(f"查看：{os.path.basename(md_path)}")

    def on_file_clicked(self, item):
        md_path = item.data(Qt.UserRole)
        self.render_md_file(md_path)

    def apply_theme(self, theme_name):
        """应用预设主题"""
        if theme_name not in THEMES:
            return
        self.current_theme_name = theme_name
        # 刷新当前显示的页面
        if self.file_list.currentItem():
            self.on_file_clicked(self.file_list.currentItem())
        else:
            # 如果没有文件，也更新一个占位提示的样式
            placeholder_html = f"<html><head><style>{generate_css(THEMES[theme_name])}</style></head><body><p>选择左侧文件开始阅读</p></body></html>"
            self.web_view.setHtml(placeholder_html)

    def custom_bg_color(self):
        """自定义背景颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            # 基于当前主题，只修改背景色
            new_theme = THEMES[self.current_theme_name].copy()
            new_theme["body_bg"] = color.name()
            # 临时应用（不存入预设主题）
            self._apply_temp_theme(new_theme)

    def custom_text_color(self):
        """自定义文字颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            new_theme = THEMES[self.current_theme_name].copy()
            new_theme["text_color"] = color.name()
            self._apply_temp_theme(new_theme)

    def _apply_temp_theme(self, theme_dict):
        """临时使用一个主题（不改变 current_theme_name）"""
        if self.file_list.currentItem():
            md_path = self.file_list.currentItem().data(Qt.UserRole)
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                html_body = markdown.markdown(md_content, extensions=['extra', 'toc', 'nl2br'])
                css = generate_css(theme_dict)
                full_html = HTML_TEMPLATE.format(style=css, content=html_body)
                self.web_view.setHtml(full_html, QUrl.fromLocalFile(os.path.dirname(md_path) + '/'))
            except:
                pass
        self.statusBar().showMessage("已应用临时颜色（未保存为主题）")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))   # 界面字体
    viewer = MarkdownReader()
    viewer.show()
    sys.exit(app.exec_())
