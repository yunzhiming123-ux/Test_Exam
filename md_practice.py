# -*- coding: utf-8 -*-
"""
MD 互动练习器 —— Markdown 题库交互式学习工具
支持三种模式（学习/练习/考试）+ 三种主题（亮色/暗色/护眼）
自动识别选择题、填空题、简答题、代码分析题
题目来源于 .md 题库文件
"""
import os, sys, re, json, html as html_mod
from enum import Enum

import markdown
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QAction, QMessageBox,
    QPushButton, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView


# ==================== 题型枚举 ====================
class QType(Enum):
    CHOICE = "选择题"
    BLANK  = "填空题"
    SHORT  = "简答题"
    CODE   = "代码分析"
    CALC   = "计算题"


# ==================== 题目数据结构 ====================
class Question:
    """单道题目，所有字段用 __slots__ 节省内存"""
    __slots__ = ('qid', 'qtype', 'title', 'options', 'answer_letter', 'answer', 'explanation', 'code')
    def __init__(self):
        self.qid = ""              # 题目编号（如 "1" 或 "1-1"）
        self.qtype = QType.SHORT  # 题型
        self.title = ""           # 题干文本
        self.options = []         # [(label, text), ...]  仅选择题有
        self.answer_letter = ""   # 选择题正确选项字母（A/B/C/D）
        self.answer = ""          # 完整答案文本
        self.explanation = ""     # 答案解析（同 answer，备用）
        self.code = ""            # 关联的代码块


# ==================== MD 解析器 ====================
class MDParser:
    """
    从 .md 文件中提取题目和普通内容区块。
    支持三种格式：
      1. **题目1（选择题）：...        —— 新格式（文件08风格）
      2. **Q1.** ...                   —— 旧格式（文件01-07、10-12风格）
      3. ### 题目 X-Y                  —— 标题格式（文件09风格）
    """

    # 正则：匹配 "**题目N（题型）：..."  或  "**题目N**（题型）：..."
    RE_QHEADER = re.compile(r'^\*?\*?(?:题目|Q)(\d+)\s*[））]?\s*(?:[（(](.+?)[）)])?\s*[:：]?\s*(.*?)$', re.I)
    # 正则：旧格式 "**Q1.** 题目内容"
    RE_LEGACY  = re.compile(r'^\*\*Q(\d+)\.?\*\*\s*(.*)')
    # 正则：标题格式 "### 题目 1-1"
    RE_HEADING = re.compile(r'^#{1,3}\s*题目\s*(\d+[-–]\d+)\s*$')
    # 正则：匹配行首的单选项 "A. xxx"
    RE_OPTION  = re.compile(r'^([A-Da-d])[\.\、\s）)]\s*(.+)')
    # 正则：匹配同行多选项  "A. xxx  B. yyy  C. zzz"（修复Bug：原先版本只能解析同行第一个选项）
    RE_OPTION_INLINE = re.compile(r'([A-D])[\.\、\s）)]\s*(.+?)(?=\s+[A-D][\.\、\s）)]|$)', re.I)
    # 正则：填空题横线（___ 或 \underline{}）
    RE_BLANK   = re.compile(r'(_{2,}|\\underline\{[^}]*\})')
    # 正则：<details> 答案块的起止
    RE_DETAILS_S = re.compile(r'<details[^>]*>')
    RE_DETAILS_E = re.compile(r'</details>')
    # 正则：从答案文本中提取正确选项字母（"A4. B。" → "B"）
    RE_ANSWER_LETTER = re.compile(r'A\d+\s*[\.\、：:]\s*([A-Da-d])[\.\、。）]', re.I)
    # 正则：独立行的选项字母（"B"）
    RE_ANSWER_STANDALONE = re.compile(r'(?:^|\n)\s*([A-Da-d])\s*[\.\、。）]?\s*$', re.M)
    # 正则：**答案N** ...  格式
    RE_ANSWER_AN = re.compile(r'^\*?\*?[A答案](\d+)\**\s*[\.\、：:）)]?\s*(.+)', re.I)

    @classmethod
    def parse(cls, md_text: str) -> tuple:
        """
        解析 md 文本，返回 (blocks, questions)
        blocks[i] = {"type":"md"|"question", "content":str, "question":Question|None}
        """
        blocks, questions = [], []
        lines = md_text.split('\n')
        i, N = 0, len(lines)

        while i < N:
            line = lines[i]

            # 格式1: ### 题目 X-Y
            m = cls.RE_HEADING.match(line.strip())
            if m:
                q = cls._parse_heading(lines, i, m)
                if q:
                    questions.append(q); blocks.append({"type":"question","question":q})
                    i = cls._next_q(lines, i); continue

            # 格式2: **Q1.** ...
            m = cls.RE_LEGACY.match(line.strip())
            if m:
                q = cls._parse_legacy(lines, i, m)
                if q:
                    questions.append(q); blocks.append({"type":"question","question":q})
                    i = cls._next_q(lines, i); continue

            # 格式3: **题目1（选择题）：...
            m = cls.RE_QHEADER.match(line.strip())
            if m:
                q = cls._parse_block(lines, i, m)
                if q:
                    questions.append(q); blocks.append({"type":"question","question":q})
                    i = cls._next_q(lines, i); continue

            # 普通 Markdown 内容块
            acc = []
            while i < N:
                l = lines[i]
                if cls.RE_QHEADER.match(l.strip()): break
                if cls.RE_LEGACY.match(l.strip()): break
                if cls.RE_HEADING.match(l.strip()): break
                acc.append(l); i += 1
            if acc:
                blocks.append({"type":"md","content":'\n'.join(acc),"question":None})

        return blocks, questions

    # ---- 工具方法 ----
    @staticmethod
    def _clean_title(t: str) -> str:
        """去除标题前后的 ** 标记、题型括号等"""
        t = re.sub(r'^\*{1,2}', '', t).strip()
        t = re.sub(r'^\*{1,2}', '', t).strip()
        t = re.sub(r'^[）)]\s*', '', t)
        t = re.sub(r'^[（(][^）)]*[）)]\s*[:：]?\s*', '', t)
        return t

    @classmethod
    def _next_q(cls, lines, start):
        """找到下一个题目的起始行号"""
        for j in range(start + 1, len(lines)):
            l = lines[j].strip()
            if cls.RE_QHEADER.match(l): return j
            if cls.RE_LEGACY.match(l): return j
            if cls.RE_HEADING.match(l): return j
        return len(lines)

    @classmethod
    def _extract_options(cls, body_lines):
        """
        从题目附近的文本行中提取选项。
        优先匹配同一行多选项（如 "A. xxx  B. yyy"），
        再回退到行首单选项匹配。
        """
        options = []
        for line in body_lines:
            s = line.strip()
            if not s: continue
            if s.startswith('```') or s.startswith('<details') or s.startswith('---'):
                continue
            # 先尝试同行多选项（修复Bug：旧版只能识别行首第一个选项）
            inline = cls.RE_OPTION_INLINE.findall(s)
            if inline and len(inline) >= 2:
                for lbl, txt in inline:
                    options.append((lbl.upper(), txt.strip()))
            else:
                m = cls.RE_OPTION.match(s)
                if m:
                    options.append((m.group(1).upper(), m.group(2).strip()))
        return options

    @classmethod
    def _extract_answer(cls, lines, from_idx, qid=""):
        """
        从 <details> 块中提取答案文本和正确选项字母。
        修复Bug：原先答案字母从全文 charAt(0) 取，导致"**A4.** B"取到A而非B。
        """
        answer_text = ""; answer_letter = ""
        details_lines = []; in_d = False

        # 1. 搜索 <details> 块
        for j in range(from_idx, min(from_idx + 120, len(lines))):
            l = lines[j]
            if cls.RE_DETAILS_S.search(l):
                in_d = True; details_lines = []; continue
            if in_d:
                if cls.RE_DETAILS_E.search(l): break
                details_lines.append(l)

        if details_lines:
            full = '\n'.join(details_lines)
            clean = re.sub(r'<[^>]+>', '', full)       # 去除 HTML 标签
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)  # 去除 ** 加粗标记
            answer_text = clean.strip()

            # 从 "A4. B。" 格式提取字母
            m_letter = cls.RE_ANSWER_LETTER.search(clean)
            if m_letter: answer_letter = m_letter.group(1).upper()
            # 从独立行 "B" 提取字母
            if not answer_letter:
                m_letter = cls.RE_ANSWER_STANDALONE.search(clean)
                if m_letter: answer_letter = m_letter.group(1).upper()
            # 宽松匹配："答案： C" 格式
            if not answer_letter:
                m_letter = re.search(r'(?:答案[：:]\s*)?\b([A-Da-d])\b', clean)
                if m_letter: answer_letter = m_letter.group(1).upper()

        # 2. 如果没找到 <details>，尝试 **答案N** 格式
        if not answer_text:
            for j in range(from_idx, min(from_idx + 80, len(lines))):
                l = lines[j]; am = cls.RE_ANSWER_AN.match(l.strip())
                if am:
                    answer_text = am.group(2).strip()
                    expl = [answer_text]
                    for k in range(j+1, min(j+20, len(lines))):
                        nl = lines[k].strip()
                        if nl == '---' or cls.RE_QHEADER.match(nl): break
                        if nl: expl.append(nl)
                    answer_text = '\n'.join(expl)
                    m_l = cls.RE_ANSWER_LETTER.search(answer_text)
                    if m_l: answer_letter = m_l.group(1).upper()
                    break

        return answer_text, answer_letter

    # ---- 三种格式的解析器 ----
    @classmethod
    def _parse_heading(cls, lines, start, m):
        """解析 ### 题目 X-Y 格式（文件09）"""
        q = Question(); q.qid = m.group(1); q.qtype = QType.CODE
        parts = []
        for j in range(start + 1, min(start + 40, len(lines))):
            l = lines[j]
            if cls.RE_HEADING.match(l.strip()): break
            if re.match(r'^#{1,3}\s*参考', l.strip()): break
            if l.strip().startswith('<details'): break
            parts.append(l)
        body = '\n'.join(parts)
        q.title = cls._clean_title(parts[0].strip()[:120] if parts else "")
        cm = re.search(r'```(?:python)?\s*\n(.*?)```', body, re.DOTALL)
        if cm: q.code = cm.group(1).strip()
        q.answer, q.answer_letter = cls._extract_answer(lines, start)
        q.explanation = q.answer
        return q

    @classmethod
    def _parse_block(cls, lines, start, m):
        """解析 **题目N（题型）：...  格式（文件08）"""
        qnum, type_hint, title_part = m.group(1), m.group(2), cls._clean_title(m.group(3))
        q = Question(); q.qid = qnum; q.title = title_part

        # 显式题型标记
        if type_hint:
            t = type_hint.replace(" ","")
            if "选择" in t: q.qtype = QType.CHOICE
            elif "填空" in t: q.qtype = QType.BLANK
            elif "简答" in t: q.qtype = QType.SHORT
            elif "代码" in t: q.qtype = QType.CODE
            elif "计算" in t: q.qtype = QType.CALC

        body_lines = []
        for j in range(start + 1, min(start + 50, len(lines))):
            l = lines[j]
            if cls.RE_QHEADER.match(l.strip()): break
            if cls.RE_LEGACY.match(l.strip()): break
            if re.match(r'^\*\*题目\s*\d+', l.strip()): break
            body_lines.append(l)

        q.options = cls._extract_options(body_lines)
        body_text = '\n'.join(body_lines)

        # 自动推断题型（没有显式标记时）
        if (not type_hint) or q.qtype == QType.SHORT:
            if len(q.options) >= 2: q.qtype = QType.CHOICE
            elif cls.RE_BLANK.search(body_text) or cls.RE_BLANK.search(q.title):
                q.qtype = QType.BLANK
            elif any('```' in l for l in body_lines): q.qtype = QType.CODE

        cm = re.search(r'```(?:python)?\s*\n(.*?)```', body_text, re.DOTALL)
        if cm: q.code = cm.group(1).strip()
        q.answer, q.answer_letter = cls._extract_answer(lines, start, qnum)
        q.explanation = q.answer
        return q

    @classmethod
    def _parse_legacy(cls, lines, start, m):
        """解析旧格式 **Q1.** ... （文件01-07、10-12）"""
        qnum = m.group(1)
        title_part = cls._clean_title(m.group(2) if m.lastindex >= 2 else "")
        q = Question(); q.qid = qnum; q.title = title_part

        body_lines = [title_part]
        for j in range(start + 1, min(start + 40, len(lines))):
            l = lines[j]
            if cls.RE_QHEADER.match(l.strip()): break
            if cls.RE_LEGACY.match(l.strip()): break
            if l.strip() == '---': break
            if cls.RE_DETAILS_S.search(l) or l.strip().startswith('<details'): break
            body_lines.append(l)

        q.options = cls._extract_options(body_lines)
        body_text = '\n'.join(body_lines)

        if len(q.options) >= 2: q.qtype = QType.CHOICE
        elif cls.RE_BLANK.search(body_text): q.qtype = QType.BLANK
        elif any('```' in l for l in body_lines): q.qtype = QType.CODE
        else: q.qtype = QType.SHORT

        cm = re.search(r'```(?:python)?\s*\n(.*?)```', body_text, re.DOTALL)
        if cm: q.code = cm.group(1).strip()
        q.answer, q.answer_letter = cls._extract_answer(lines, start, qnum)
        q.explanation = q.answer
        return q


# ==================== 三套主题（从 MyFirstQt.py 继承并扩展） ====================
THEMES = {
    "亮色": {
        "bg": "#f0f4f8", "card_bg": "#fff", "text": "#263238", "sub": "#546e7a",
        "border": "#e0e0e0", "accent": "#1a56db", "accent2": "#4caf50",
        "code_bg": "#e8eaed", "pre_bg": "#1e1e1e", "pre_text": "#d4d4d4",
        "table_h": "linear-gradient(135deg,#e3f2fd,#bbdefb)",
        "blockquote_bg": "#fff8e1", "blockquote_border": "#ff9800",
    },
    "暗色": {
        "bg": "#1a1a2e", "card_bg": "#202040", "text": "#e0e0e0", "sub": "#a0a0b0",
        "border": "#333355", "accent": "#5b9df5", "accent2": "#66bb6a",
        "code_bg": "#2a2a4a", "pre_bg": "#0d0d1a", "pre_text": "#c8c8e0",
        "table_h": "linear-gradient(135deg,#2a2a4a,#333366)",
        "blockquote_bg": "#2a2a1a", "blockquote_border": "#cc8800",
    },
    "护眼": {
        "bg": "#f5f0e8", "card_bg": "#fefcf7", "text": "#4a3b2c", "sub": "#6d5d4b",
        "border": "#d4c5b0", "accent": "#6b4226", "accent2": "#4a8c3f",
        "code_bg": "#f0e6d3", "pre_bg": "#3d3226", "pre_text": "#e0d5c0",
        "table_h": "linear-gradient(135deg,#f0e6d3,#e6d6c0)",
        "blockquote_bg": "#fcf3e0", "blockquote_border": "#cc8800",
    },
}


def build_css(theme_name="亮色"):
    """根据主题名生成完整的 CSS（含动画）。"""
    t = THEMES.get(theme_name, THEMES["亮色"])
    return f"""
*{{margin:0;padding:0;box-sizing:border-box}}
body{{
    font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif;
    background:{t["bg"]};color:{t["text"]};padding:16px 20px 40px;line-height:1.75;
    transition:all .4s ease;
}}
h1{{font-size:1.7em;color:{t["accent"]};margin-bottom:14px;border-left:4px solid {t["accent"]};padding-left:14px;animation:slideLeft .5s ease}}
h2{{font-size:1.3em;color:{t["accent"]};margin:24px 0 10px;border-bottom:1px solid {t["border"]};padding-bottom:6px}}
h3{{font-size:1.1em;color:{t["sub"]};margin:16px 0 6px}}
p{{margin:8px 0}}
code{{background:{t["code_bg"]};color:#d63384;padding:2px 6px;border-radius:3px;font-family:"Fira Code","Cascadia Code",Consolas,monospace;font-size:.9em}}
pre{{background:{t["pre_bg"]};color:{t["pre_text"]};padding:16px;border-radius:10px;overflow-x:auto;margin:10px 0;line-height:1.5}}
pre code{{background:none;color:inherit;padding:0}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
th,td{{border:1px solid {t["border"]};padding:8px 12px;text-align:left}}
th{{background:{t["table_h"]};color:{t["accent"]};font-weight:600}}
tr:nth-child(even){{background:{t["card_bg"]}}}
blockquote{{border-left:4px solid {t["blockquote_border"]};padding:8px 16px;margin:10px 0;background:{t["blockquote_bg"]};color:{t["text"]};border-radius:0 8px 8px 0}}
img{{max-width:100%;border-radius:8px}}

/* 动画定义 */
@keyframes fadeInUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes slideLeft{{from{{opacity:0;transform:translateX(-20px)}}to{{opacity:1;transform:translateX(0)}}}}
@keyframes shake{{0%,100%{{transform:translateX(0)}}10%,30%,50%,70%,90%{{transform:translateX(-4px)}}20%,40%,60%,80%{{transform:translateX(4px)}}}}
@keyframes checkBounce{{0%{{transform:scale(0);opacity:0}}50%{{transform:scale(1.3)}}100%{{transform:scale(1);opacity:1}}}}
@keyframes borderGlow{{0%,100%{{box-shadow:0 0 4px {t["accent"]}33}}50%{{box-shadow:0 0 18px {t["accent"]}88}}}}

/* 题目卡片 */
.question-card{{
    background:{t["card_bg"]};border-radius:16px;padding:22px 24px;
    margin:16px 0;box-shadow:0 3px 20px rgba(0,0,0,.06);
    border-left:5px solid {t["accent"]};transition:all .35s ease;
    animation:fadeInUp .45s ease both;position:relative;
}}
.question-card:nth-child(2){{animation-delay:.05s}}
.question-card:nth-child(3){{animation-delay:.1s}}
.question-card:nth-child(4){{animation-delay:.15s}}
.question-card:nth-child(5){{animation-delay:.2s}}
.question-card:hover{{box-shadow:0 6px 28px rgba(0,0,0,.1);transform:translateY(-2px)}}
.question-card.correct{{border-left-color:{t["accent2"]};background:linear-gradient(to right,{t["card_bg"]} 70%,#e8f5e9)}}
.question-card.correct::after{{content:'\\2713';position:absolute;top:10px;right:16px;font-size:2em;color:{t["accent2"]};animation:checkBounce .5s ease}}
.question-card.wrong{{border-left-color:#f44336;background:linear-gradient(to right,{t["card_bg"]} 70%,#ffebee);animation:shake .5s ease}}
.question-card.wrong::after{{content:'\\2717';position:absolute;top:10px;right:16px;font-size:2em;color:#f44336;animation:checkBounce .5s ease}}

.q-header{{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}}
.q-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.78em;font-weight:700;color:#fff;letter-spacing:.5px}}
.q-badge.choice{{background:linear-gradient(135deg,#1976d2,#42a5f5)}}
.q-badge.blank{{background:linear-gradient(135deg,#2e7d32,#66bb6a)}}
.q-badge.short{{background:linear-gradient(135deg,#e65100,#ff9800)}}
.q-badge.code{{background:linear-gradient(135deg,#6a1b9a,#ab47bc)}}
.q-badge.calc{{background:linear-gradient(135deg,#b71c1c,#ef5350)}}
.q-title{{font-weight:600;font-size:1.02em;color:{t["text"]};flex:1;min-width:200px}}
.q-code-block{{background:{t["pre_bg"]};color:{t["pre_text"]};padding:14px 18px;border-radius:10px;overflow-x:auto;margin:10px 0;font-family:"Fira Code","Cascadia Code",monospace;font-size:.85em;line-height:1.5;white-space:pre}}

/* 选择题选项 */
.option-group{{margin:6px 0 12px}}
.option-item{{
    display:block;padding:12px 18px;margin:7px 0;
    border:2px solid {t["border"]};border-radius:12px;
    cursor:pointer;transition:all .25s cubic-bezier(.4,0,.2,1);
    font-size:.96em;position:relative;overflow:hidden;
}}
.option-item::before{{content:'';position:absolute;top:50%;left:50%;width:0;height:0;border-radius:50%;background:{t["accent"]}19;transform:translate(-50%,-50%);transition:width .6s,height .6s}}
.option-item:active::before{{width:300px;height:300px}}
.option-item:hover{{border-color:{t["accent"]}88;background:{t["accent"]}0d;transform:translateX(4px);box-shadow:0 2px 8px {t["accent"]}26}}
.option-item input{{display:none}}
.option-item.selected{{border-color:{t["accent"]};background:linear-gradient(135deg,{t["accent"]}18,{t["accent"]}0d);font-weight:600;box-shadow:0 2px 12px {t["accent"]}33}}
.option-item.reveal-correct{{border-color:{t["accent2"]}!important;background:linear-gradient(135deg,#e8f5e9,#c8e6c9)!important;animation:borderGlow 1.5s ease infinite}}
.option-item.reveal-wrong{{border-color:#f44336!important;background:#ffebee!important;text-decoration:line-through}}
.option-item.disabled{{pointer-events:none;opacity:.85}}
.option-label{{display:inline-block;width:30px;height:30px;line-height:30px;border-radius:50%;background:{t["code_bg"]};text-align:center;font-weight:700;margin-right:12px;color:{t["accent"]};transition:all .25s}}
.option-item.selected .option-label{{background:{t["accent"]};color:#fff;transform:scale(1.1)}}
.option-item.reveal-correct .option-label{{background:{t["accent2"]}!important;color:#fff}}

/* 填空题输入框 */
.blank-group{{margin:8px 0 12px;display:flex;flex-wrap:wrap;align-items:center;gap:6px}}
.blank-input{{width:160px;padding:9px 14px;border:2px solid {t["border"]};border-radius:8px;font-size:.94em;transition:all .25s;outline:none;font-family:inherit;background:{t["card_bg"]};color:{t["text"]}}}
.blank-input:focus{{border-color:{t["accent"]};box-shadow:0 0 0 3px {t["accent"]}26}}
.blank-input.correct{{border-color:{t["accent2"]}!important;background:#e8f5e9;animation:borderGlow 1.5s ease infinite}}
.blank-input.wrong{{border-color:#f44336!important;background:#ffebee}}
.blank-label{{display:inline-block;min-width:28px;color:{t["sub"]};font-size:.85em;font-weight:600}}

/* 答案区 */
.answer-box{{margin-top:16px;padding:16px 20px;border-radius:12px;display:none;transition:all .35s ease}}
.answer-box.show{{display:block;animation:fadeInUp .35s ease}}
.answer-box.correct-answer{{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border:1px solid #a5d6a7}}
.answer-box.wrong-answer{{background:linear-gradient(135deg,#ffebee,#ffcdd2);border:1px solid #ef9a9a}}
.answer-box.normal-answer{{background:{t["card_bg"]};border:1px solid {t["border"]}}}
.answer-label{{font-weight:700;margin-bottom:8px;font-size:1.02em}}
.answer-label.correct{{color:#2e7d32}}
.answer-label.wrong{{color:#c62828}}
.answer-text{{color:{t["sub"]};line-height:1.8}}

/* 通用按钮 */
.btn{{display:inline-block;padding:10px 24px;border:none;border-radius:10px;font-size:.92em;font-weight:600;cursor:pointer;transition:all .25s;font-family:inherit;margin:4px;position:relative;overflow:hidden}}
.btn::after{{content:'';position:absolute;top:50%;left:50%;width:0;height:0;border-radius:50%;background:rgba(255,255,255,.3);transform:translate(-50%,-50%);transition:width .6s,height .6s}}
.btn:active::after{{width:300px;height:300px}}
.btn-primary{{background:linear-gradient(135deg,{t["accent"]},#4285f4);color:#fff;box-shadow:0 3px 10px {t["accent"]}4d}}
.btn-primary:hover{{box-shadow:0 6px 20px {t["accent"]}73;transform:translateY(-1px)}}
.btn-success{{background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff}}
.btn-success:hover{{transform:translateY(-1px)}}
.btn-outline{{background:{t["card_bg"]};color:{t["accent"]};border:2px solid {t["accent"]}}}
.btn-outline:hover{{background:{t["accent"]}0d}}
.btn-small{{padding:5px 14px;font-size:.8em}}

/* 顶部计分栏（粘性定位） */
.score-bar{{position:sticky;top:0;z-index:100;background:{t["card_bg"]};border-radius:16px;padding:16px 22px;margin-bottom:18px;box-shadow:0 3px 20px rgba(0,0,0,.08);display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.score-stat{{text-align:center;transition:transform .3s}}
.score-stat:hover{{transform:scale(1.08)}}
.score-num{{font-size:1.8em;font-weight:800;background:linear-gradient(135deg,{t["accent"]},#42a5f5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.score-label{{font-size:.72em;color:{t["sub"]};text-transform:uppercase;letter-spacing:1px}}
.score-bar .btn{{margin-left:auto}}

/* 搜索框 */
.search-box{{width:100%;padding:8px 14px;border:2px solid {t["border"]};border-radius:8px;font-size:.88em;outline:none;margin-bottom:8px;transition:all .25s;font-family:inherit;background:{t["card_bg"]};color:{t["text"]}}}
.search-box:focus{{border-color:{t["accent"]};box-shadow:0 0 0 3px {t["accent"]}1a}}

@media(max-width:700px){{body{{padding:8px}}.question-card{{padding:14px}}.blank-input{{width:110px}}}}
"""

# ==================== JavaScript 逻辑 ====================
JS = r"""
var currentMode='learn';
var answers={};
var scores={answered:0,correct:0};
var examSubmitted=false;

/* 选择题：选中选项 */
function selectChoice(qid,label){
    if(examSubmitted||currentMode==='learn')return;
    var card=document.getElementById('card-'+qid);
    card.querySelectorAll('.option-item').forEach(function(item){
        item.classList.remove('selected');
        if(item.getAttribute('data-label')===label)item.classList.add('selected');
    });
    answers[qid]={type:'choice',value:label};
    updateScore();
}

/* 填空题：收集输入 */
function onBlankChange(qid){
    if(examSubmitted||currentMode==='learn')return;
    var card=document.getElementById('card-'+qid);
    var vals=[];
    card.querySelectorAll('.blank-input').forEach(function(inp){vals.push(inp.value.trim());});
    answers[qid]={type:'blank',value:vals};
    updateScore();
}

/* 单题提交：比对学生答案与正确答案 */
function submitAnswer(qid){
    if(examSubmitted)return;
    var ad=answers[qid];
    if(!ad){alert('请先作答！');return;}
    var ok=false;
    var correct=window.__correctAnswers__[qid]||'';
    if(ad.type==='choice'){
        /* 直接比对字母（修复Bug：旧版用charAt(0)取"**A4.** B"的'A'而非'B'） */
        ok=(ad.value.toUpperCase()===correct.toUpperCase());
    }else if(ad.type==='blank'){
        var ua=ad.value;
        var ca=correct.split('|').map(function(s){return s.trim().toLowerCase().replace(/\s/g,'');});
        ok=true;
        for(var i=0;i<Math.max(ua.length,ca.length);i++){
            var u=(ua[i]||'').toLowerCase().replace(/\s/g,'');
            var c=ca[i]||'';
            if(u!==c&&c&&u){ok=false;break}
        }
    }
    revealAnswer(qid,ok);
    /* 练习模式计分：只计算首次提交 */
    if(currentMode==='practice'&&!(qid in (window.__submitted__||{}))){
        window.__submitted__=window.__submitted__||{};
        window.__submitted__[qid]=true;
        scores.answered++;
        if(ok)scores.correct++;
        updateScore();
    }
}

/* 显示答案：高亮正确/错误选项、显示正确答案 */
function revealAnswer(qid,isCorrect){
    var card=document.getElementById('card-'+qid);
    var ab=card.querySelector('.answer-box');
    var lbl=ab.querySelector('.answer-label');
    var items=card.querySelectorAll('.option-item');
    var inputs=card.querySelectorAll('.blank-input');
    var correct=window.__correctAnswers__[qid]||'';
    ab.classList.add('show');
    items.forEach(function(it){it.classList.add('disabled');});
    inputs.forEach(function(inp){inp.disabled=true;});

    /* 高亮正确的选项 */
    var cc=correct.toUpperCase().charAt(0);
    items.forEach(function(it){
        if(it.getAttribute('data-label').toUpperCase()===cc)it.classList.add('reveal-correct');
        if(it.classList.contains('selected')&&it.getAttribute('data-label').toUpperCase()!==cc)
            it.classList.add('reveal-wrong');
    });

    /* 高亮填空答案 */
    var cvals=correct.split('|').map(function(s){return s.trim();});
    inputs.forEach(function(inp,idx){
        var u=(inp.value||'').trim().toLowerCase().replace(/\s/g,'');
        var c=(cvals[idx]||'').trim().toLowerCase().replace(/\s/g,'');
        if(c&&u===c)inp.classList.add('correct');
        else if(c&&u)inp.classList.add('wrong');
        inp.value=inp.value+'  ['+(cvals[idx]||'?')+']';
    });

    if(isCorrect){
        card.classList.add('correct');
        ab.className='answer-box show correct-answer';
        lbl.className='answer-label correct';
        lbl.innerHTML='<span style=font-size:1.3em>✓</span> 回答正确！';
    }else{
        card.classList.add('wrong');
        ab.className='answer-box show wrong-answer';
        lbl.className='answer-label wrong';
        lbl.innerHTML='<span style=font-size:1.3em>✗</span> 回答错误';
    }
    card.scrollIntoView({behavior:'smooth',block:'center'});
}

/* 学习模式：折叠/展开答案 */
function toggleAnswer(qid){
    var ab=document.getElementById('ans-'+qid);
    var btn=document.getElementById('btn-'+qid);
    if(ab.classList.contains('show')){ab.classList.remove('show');btn.textContent='显示答案';}
    else{ab.classList.add('show');btn.textContent='隐藏答案';ab.scrollIntoView({behavior:'smooth',block:'center'});}
}

/* 更新计分面板数字 */
function updateScore(){
    var ae=document.getElementById('stat-answered');
    var ce=document.getElementById('stat-correct');
    var te=document.getElementById('stat-total');
    if(ae)ae.textContent=Object.keys(answers).length;
    if(ce)ce.textContent=scores.correct;
    if(te)te.textContent=window.__totalQuestions__||0;
}

/* 考试模式：统一提交 */
function submitExam(){
    if(examSubmitted)return;
    if(!confirm('确定要提交试卷吗？提交后无法修改。'))return;
    examSubmitted=true;
    var total=window.__totalQuestions__||0;
    var correct=0;
    var corrects=window.__correctAnswers__||{};
    for(var qid in corrects){
        var ad=answers[qid];
        var ca=corrects[qid]||'';
        if(ad&&ad.type==='choice'){
            if(ad.value.toUpperCase()===ca.toUpperCase())correct++;
        }else if(ad&&ad.type==='blank'){
            var ua=ad.value;
            var ci=ca.split('|').map(function(s){return s.trim().toLowerCase().replace(/\s/g,'');});
            var ok=true;
            for(var i=0;i<Math.max(ua.length,ci.length);i++){
                var u=(ua[i]||'').toLowerCase().replace(/\s/g,'');
                var c=ci[i]||'';
                if(u!==c&&c&&u){ok=false;break}
            }
            if(ok)correct++;
        }
    }
    /* 显示所有答案 */
    Object.keys(corrects).forEach(function(qid){
        var ad=answers[qid];
        var isC=false;
        if(ad&&ad.type==='choice')isC=(ad.value.toUpperCase()===corrects[qid].toUpperCase());
        revealAnswer(qid,isC);
    });
    var score=total>0?Math.round(correct/total*100):0;
    var gb=document.getElementById('grade-bar');
    if(gb){
        gb.innerHTML='<div style=text-align:center;padding:30px;animation:fadeInUp .6s ease>'+
            '<div style=font-size:4em;font-weight:800;background:linear-gradient(135deg,#1a56db,#42a5f5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text>'+score+'</div>'+
            '<div style=font-size:1.3em;color:#78909c;margin-top:4px>分</div>'+
            '<div style=margin-top:12px;font-size:1.1em;color:#546e7a>答对 <b>'+correct+'</b> / 共 <b>'+total+'</b> 题</div>'+
            '<div style=margin-top:18px;font-size:1.15em;font-weight:600;color:'+
              (score>=90?'#2e7d32':score>=70?'#1a56db':score>=60?'#e65100':'#c62828')+'">'+
              (score>=90?'非常优秀！':score>=80?'表现很好！':score>=70?'不错！继续加油':score>=60?'还需努力':score>=40?'需要多加练习':'要好好复习了')+'</div></div>';
    }
    window.scrollTo(0,0);
}

/* 考试模式：随机打乱选项顺序 */
function shuffleOptions(){
    document.querySelectorAll('.option-group').forEach(function(g){
        var its=Array.prototype.slice.call(g.children);
        for(var i=its.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1));g.appendChild(its[j]);}
    });
}

/* 搜索功能 */
function searchQuestions(){
    var q=document.getElementById('search-input').value.toLowerCase();
    document.querySelectorAll('.question-card').forEach(function(c){
        if(!q||c.textContent.toLowerCase().indexOf(q)>=0)c.style.display='';
        else c.style.display='none';
    });
}

/* 键盘快捷键：N = 下一题 */
document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
    if(e.key==='n'||e.key==='N'){
        var cards=document.querySelectorAll('.question-card');
        for(var i=0;i<cards.length;i++){
            var r=cards[i].getBoundingClientRect();
            if(r.bottom>50){cards[i].scrollIntoView({behavior:'smooth',block:'center'});break}
        }
    }
});
"""


# ==================== HTML 生成器 ====================
class HTMLGen:
    """根据解析结果和当前主题/模式生成完整 HTML"""

    @staticmethod
    def _base(body, extra_js, theme_name):
        css = build_css(theme_name)
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MD 互动练习器</title>
<style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$']]}},svg:{{fontCache:'global'}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs2015.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>document.addEventListener('DOMContentLoaded',function(){{hljs.highlightAll();}});</script>
</head>
<body>{body}
<script>{JS}</script><script>{extra_js}</script>
</body></html>"""

    @staticmethod
    def generate(blocks, questions, mode="learn", file_name="", theme="亮色"):
        """主入口：根据模式分发到对应的渲染方法"""
        if mode == "learn": return HTMLGen._learn(blocks, questions, file_name, theme)
        elif mode == "practice": return HTMLGen._practice(questions, file_name, theme)
        elif mode == "exam": return HTMLGen._exam(questions, file_name, theme)
        return ""

    @staticmethod
    def _question_card(q, mode="learn"):
        """生成单道题目的 HTML 卡片"""
        badge_map = {
            QType.CHOICE: ("choice","选择题"), QType.BLANK:("blank","填空题"),
            QType.SHORT:("short","简答题"), QType.CODE:("code","代码分析"), QType.CALC:("calc","计算题")
        }
        bc, bt = badge_map.get(q.qtype, ("short", "题目"))
        p = []
        right_extra = ' position:relative' if mode != "learn" else ''
        p.append(f'<div class="question-card" id="card-{q.qid}" style="{right_extra}">')
        p.append(f'<div class="q-header"><span class="q-badge {bc}">{bt} #{q.qid}</span>')
        p.append(f'<span class="q-title">{html_mod.escape(q.title)}</span></div>')
        if q.code: p.append(f'<div class="q-code-block">{html_mod.escape(q.code)}</div>')
        if q.qtype == QType.CHOICE: p.append(HTMLGen._choice_html(q, mode))
        elif q.qtype == QType.BLANK: p.append(HTMLGen._blank_html(q, mode))
        p.append(HTMLGen._answer_html(q, mode))
        p.append('</div>')
        return '\n'.join(p)

    @staticmethod
    def _choice_html(q, mode):
        """选择题的选项 HTML"""
        p = ['<div class="option-group">']
        for lbl, txt in q.options:
            sl, st = html_mod.escape(lbl), html_mod.escape(txt)
            click = f"selectChoice('{q.qid}','{sl}')" if mode != "learn" else ""
            p.append(f'<label class="option-item" data-label="{sl}" onclick="{click}"><span class="option-label">{sl}</span>{st}</label>')
        p.append('</div>')
        if mode == "practice":
            p.append(f'<button class="btn btn-primary" onclick="submitAnswer(\'{q.qid}\')">提交答案</button>')
        return '\n'.join(p)

    @staticmethod
    def _blank_html(q, mode):
        """填空题的输入框 HTML"""
        bc = len(re.findall(r'(_{2,}|\\underline\{[^}]*\})', q.title)) or 1
        p = ['<div class="blank-group">']
        for i in range(bc):
            dis = 'disabled' if mode == "learn" else ''
            chg = f"onBlankChange('{q.qid}')" if mode != "learn" else ""
            p.append(f'<span class="blank-label">({i+1})</span>')
            p.append(f'<input type="text" class="blank-input" placeholder="输入答案...{i+1}" oninput="{chg}" {dis}/>')
        p.append('</div>')
        if mode == "practice":
            p.append(f'<button class="btn btn-primary" onclick="submitAnswer(\'{q.qid}\')">提交答案</button>')
        return '\n'.join(p)

    @staticmethod
    def _answer_html(q, mode):
        """答案区域 HTML"""
        ah = html_mod.escape(q.explanation or q.answer or "暂无答案").replace('\n','<br>')
        if mode == "learn":
            return (f'<button class="btn btn-outline btn-small" id="btn-{q.qid}" onclick="toggleAnswer(\'{q.qid}\')">显示答案</button>\n'
                    f'<div class="answer-box normal-answer" id="ans-{q.qid}"><div class="answer-label">参考答案：</div><div class="answer-text">{ah}</div></div>')
        else:
            return f'<div class="answer-box normal-answer"><div class="answer-label">参考答案：</div><div class="answer-text">{ah}</div></div>'

    # ---- 三种模式 ----
    @staticmethod
    def _learn(blocks, questions, fn, theme):
        parts = []
        for blk in blocks:
            if blk["type"] == "md":
                parts.append(markdown.markdown(blk["content"], extensions=['extra','toc','nl2br','fenced_code','codehilite']))
            elif blk["type"] == "question":
                parts.append(HTMLGen._question_card(blk["question"], "learn"))
        body = '<div style="margin-bottom:12px"><input type="text" id="search-input" class="search-box" placeholder="搜索题目..." oninput="searchQuestions()"></div>\n'
        body += '\n'.join(parts) or '<p style="text-align:center;padding:40px">该文件中未检测到题目</p>'
        cd = {q.qid: (q.answer_letter if q.qtype == QType.CHOICE else q.answer) for q in questions}
        return HTMLGen._base(body, f"""window.__totalQuestions__={len(questions)};
window.__correctAnswers__={json.dumps(cd, ensure_ascii=False)};currentMode='learn';""", theme)

    @staticmethod
    def _practice(questions, fn, theme):
        cards = [HTMLGen._question_card(q, "practice") for q in questions]
        body = '<div class="score-bar" id="score-panel">'
        body += '<div class="score-stat"><div class="score-num" id="stat-answered">0</div><div class="score-label">已答</div></div>'
        body += f'<div class="score-stat"><div class="score-num" id="stat-total">{len(questions)}</div><div class="score-label">总题数</div></div>'
        body += '<div class="score-stat"><div class="score-num" id="stat-correct">0</div><div class="score-label">答对</div></div></div>\n'
        body += '\n'.join(cards) or '<p style="text-align:center;padding:40px">未检测到题目</p>'
        cd = {q.qid: (q.answer_letter if q.qtype == QType.CHOICE else q.answer) for q in questions}
        return HTMLGen._base(body, f"""window.__totalQuestions__={len(questions)};
window.__correctAnswers__={json.dumps(cd, ensure_ascii=False)};window.__submitted__={{}};currentMode='practice';""", theme)

    @staticmethod
    def _exam(questions, fn, theme):
        cards = [HTMLGen._question_card(q, "exam") for q in questions]
        body = '<div class="score-bar" id="grade-bar">'
        body += '<div class="score-stat"><div class="score-num" id="stat-answered">0</div><div class="score-label">已答</div></div>'
        body += f'<div class="score-stat"><div class="score-num" id="stat-total">{len(questions)}</div><div class="score-label">总题数</div></div>'
        body += '<div class="score-stat"><div class="score-num" id="timer-display">--:--</div><div class="score-label">用时</div></div>'
        body += '<button class="btn btn-success" onclick="submitExam()" id="submit-exam-btn">提交试卷</button></div>\n'
        body += '\n'.join(cards) or '<p style="text-align:center;padding:40px">未检测到题目</p>'
        cd = {q.qid: (q.answer_letter if q.qtype == QType.CHOICE else q.answer) for q in questions}
        return HTMLGen._base(body, f"""window.__totalQuestions__={len(questions)};
window.__correctAnswers__={json.dumps(cd, ensure_ascii=False)};currentMode='exam';
var es=Date.now();
setInterval(function(){{if(examSubmitted)return;var e=Math.floor((Date.now()-es)/1000);var m=Math.floor(e/60);var s=e%60;
var el=document.getElementById('timer-display');if(el)el.textContent=(m<10?'0':'')+m+':'+(s<10?'0':'')+s;}},1000);
shuffleOptions();""", theme)


# ==================== 主窗口 ====================
class MDPracticeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cur_dir = os.getcwd()     # 当前浏览目录
        self.cur_file = None           # 当前打开的文件路径
        self.blocks = []               # 当前已解析的内容块
        self.questions = []            # 当前已解析的题目列表
        self.mode = "learn"            # 当前模式: learn / practice / exam
        self.theme = "亮色"            # 当前主题: 亮色 / 暗色 / 护眼
        self.cache = {}                # 文件解析缓存，避免重复解析

        self._ui()
        self._load_list(self.cur_dir)

    # ---- UI 搭建 ----
    def _ui(self):
        self.setWindowTitle("MD 互动练习器")
        self.setGeometry(80, 60, 1450, 920)

        # 菜单栏
        mb = self.menuBar()
        fm = mb.addMenu("文件(&F)")
        fm.addAction(QAction("打开文件夹...", self, triggered=self._open_folder))
        fm.addAction(QAction("打开文件...", self, triggered=self._open_file))
        fm.addSeparator()
        fm.addAction(QAction("退出(&Q)", self, triggered=self.close))
        hm = mb.addMenu("帮助(&H)")
        hm.addAction(QAction("关于", self, triggered=lambda: QMessageBox.about(self, "关于",
            "MD 互动练习器 v2.0\n\n将 Markdown 题库变成可交互的练习工具。\n\n"
            "三种模式：学习 | 练习 | 考试\n三种主题：亮色 | 暗色 | 护眼\n\n快捷键: N=下一题")))

        sp = QSplitter(Qt.Horizontal)

        # --- 左侧面板 ---
        lp = QWidget()
        ll = QVBoxLayout(lp)
        ll.setContentsMargins(4, 8, 4, 8)

        # 模式切换
        ml = QHBoxLayout()
        ml.addWidget(QLabel("模式："))
        self._bts = {}
        for k, txt in [("learn","学习"), ("practice","练习"), ("exam","考试")]:
            b = QPushButton(txt)
            b.setCheckable(True); b.setFixedHeight(34); b.setStyleSheet(self._bs())
            b.clicked.connect(lambda _,m=k: self._switch(m))
            ml.addWidget(b); self._bts[k] = b
        self._bts["learn"].setChecked(True)
        ll.addLayout(ml)

        # 主题切换（继承自 MyFirstQt.py 的功能）
        tl = QHBoxLayout()
        tl.addWidget(QLabel("主题："))
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(list(THEMES.keys()))
        self._theme_combo.currentTextChanged.connect(self._on_theme_change)
        self._theme_combo.setStyleSheet("QComboBox{padding:4px 8px;border:1px solid #ddd;border-radius:6px;font-size:12px}")
        tl.addWidget(self._theme_combo)
        ll.addLayout(tl)

        # 文件列表
        self._fl = QListWidget()
        self._fl.itemClicked.connect(self._on_click)
        self._fl.setStyleSheet("""
            QListWidget{border:1px solid #e0e0e0;border-radius:8px;background:#fafafa;font-size:13px;padding:4px}
            QListWidget::item{padding:8px 12px;border-radius:4px;margin:1px 0}
            QListWidget::item:hover{background:#e3f2fd}QListWidget::item:selected{background:#1a56db;color:#fff}
        """)
        ll.addWidget(self._fl)

        rb = QPushButton("刷新列表")
        rb.clicked.connect(lambda: self._load_list(self.cur_dir))
        rb.setStyleSheet(self._bs())
        ll.addWidget(rb)

        sp.addWidget(lp)

        # --- 右侧 Web 预览 ---
        self._wv = QWebEngineView()
        self._wv.setStyleSheet("border:none;background:#e8edf5")
        sp.addWidget(self._wv)
        sp.setSizes([280, 1170])
        self.setCentralWidget(sp)
        self.statusBar().showMessage("就绪 - 选择左侧文件开始")

    def _bs(self):
        """按钮样式"""
        return """QPushButton{background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:4px 14px;font-size:12px;font-weight:600}
        QPushButton:hover{background:#e3f2fd;border-color:#90caf9}
        QPushButton:checked{background:#1a56db;color:#fff;border-color:#1a56db}"""

    # ---- 文件列表管理 ----
    def _load_list(self, d):
        self._fl.clear()
        if not os.path.isdir(d): QMessageBox.warning(self, "错误", f"文件夹不存在：{d}"); return
        self.cur_dir = d
        fs = sorted([f for f in os.listdir(d) if f.lower().endswith('.md')])
        for f in fs:
            it = QListWidgetItem(f); it.setData(Qt.UserRole, os.path.join(d, f))
            self._fl.addItem(it)
        if fs: self._fl.setCurrentRow(0); self._on_click(self._fl.currentItem())

    def _open_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择 Markdown 文件", self.cur_dir, "Markdown (*.md);;所有文件 (*)")
        if fp: self.cur_dir = os.path.dirname(fp); self._load_list(self.cur_dir); self._render_file(fp)

    def _open_folder(self):
        fd = QFileDialog.getExistingDirectory(self, "选择文件夹", self.cur_dir)
        if fd: self._load_list(fd)

    def _on_click(self, item):
        self._render_file(item.data(Qt.UserRole))

    # ---- 文件加载与渲染 ----
    def _render_file(self, fp):
        try:
            with open(fp, 'r', encoding='utf-8') as f: txt = f.read()
        except Exception as e:
            self._wv.setHtml(f"<p style='color:red;padding:20px'>读取失败：{e}</p>"); return
        self.cur_file = fp
        # 缓存机制：避免重复解析
        if fp in self.cache:
            self.blocks, self.questions = self.cache[fp]
        else:
            self.blocks, self.questions = MDParser.parse(txt)
            self.cache[fp] = (self.blocks, self.questions)

        if not self.questions:
            h = markdown.markdown(txt, extensions=['extra','toc','nl2br','fenced_code','codehilite'])
            css = build_css(self.theme)
            self._wv.setHtml(f"<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><style>{css}</style></head><body>{h}</body></html>",
                             QUrl.fromLocalFile(os.path.dirname(fp)+'/'))
            self.statusBar().showMessage(f"[无题目] {os.path.basename(fp)}")
        else:
            self._render()
        mn = {"learn":"学习","practice":"练习","exam":"考试"}
        self.statusBar().showMessage(f"{os.path.basename(fp)} | {len(self.questions)}题 | {mn[self.mode]} | {self.theme}主题")

    def _render(self):
        if not self.cur_file: return
        try:
            h = HTMLGen.generate(self.blocks, self.questions, self.mode,
                                 os.path.basename(self.cur_file), self.theme)
            self._wv.setHtml(h, QUrl.fromLocalFile(os.path.dirname(self.cur_file)+'/'))
        except Exception as e:
            self._wv.setHtml(f"<p style='color:red;padding:20px'>渲染出错：{e}</p>")

    # ---- 模式/主题切换 ----
    def _switch(self, m):
        self.mode = m
        for k, b in self._bts.items(): b.setChecked(k == m)
        if self.cur_file: self._render_file(self.cur_file)
        mn = {"learn":"学习","practice":"练习","exam":"考试"}
        self.statusBar().showMessage(f"已切换到{mn[m]}模式 | {self.theme}主题")

    def _on_theme_change(self, name):
        """主题下拉框变化时重新渲染（继承自 MyFirstQt.py 的主题切换功能）"""
        if name not in THEMES: return
        self.theme = name
        if self.cur_file: self._render_file(self.cur_file)


# ==================== 启动入口 ====================
def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))
    app.setStyle("Fusion")
    # 全局调色板
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f5f7fa")); pal.setColor(QPalette.WindowText, QColor("#263238"))
    pal.setColor(QPalette.Base, QColor("#ffffff")); pal.setColor(QPalette.AlternateBase, QColor("#f0f4f8"))
    pal.setColor(QPalette.Button, QColor("#ffffff")); pal.setColor(QPalette.ButtonText, QColor("#37474f"))
    pal.setColor(QPalette.Highlight, QColor("#1a56db")); pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    w = MDPracticeApp(); w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
