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
from datetime import datetime
from enum import Enum
from collections import defaultdict

import markdown
from PyQt5.QtWidgets import (
    QColorDialog,
    QApplication, QMainWindow, QSplitter, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QAction, QMessageBox,
    QToolBar, QPushButton, QLabel, QComboBox, QFrame, QCheckBox, QProgressBar,
    QButtonGroup, QRadioButton, QSizePolicy, QSpacerItem, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView
try:
    from PyQt5.QtWebChannel import QWebChannel
    HAS_WEBCHANNEL = True
except ImportError:
    HAS_WEBCHANNEL = False


# ==================== 数据结构 ====================
class QuestionType(Enum):
    CHOICE = "选择题"
    BLANK  = "填空题"
    SHORT  = "简答题"
    CODE   = "代码分析题"
    CALC   = "计算题"


class Question:
    """单道题目"""
    __slots__ = ('qid', 'display_id', 'qtype', 'title', 'options', 'answer_letter', 'answer', 'explanation', 'code')
    def __init__(self):
        self.qid = ""            # 全局唯一ID（计数器生成）
        self.display_id = ""     # 原始题号（显示用，如"Q1"）
        self.qtype = QuestionType.SHORT
        self.title = ""
        self.options = []        # [(label, text), ...]  选择题的选项
        self.answer_letter = ""  # 选择题正确选项字母 (A/B/C/D)
        self.answer = ""         # 完整答案文本
        self.explanation = ""    # 答案解析
        self.code = ""           # 关联的代码块


# ==================== MD 解析器 ====================
class MDParser:
    """解析 .md 文件，提取题目列表和普通内容区块"""

    # 匹配题型标记：题目N（选择题）
    RE_QHEADER = re.compile(
        r'^\*?\*?(?:题目|Q)(\d+)\s*[））]?\s*(?:（(.+?)）)?\s*(.*?)$',
        re.IGNORECASE
    )
    # 匹配选项 A. / B. / C. / D.
    RE_OPTION  = re.compile(r'^([A-Da-d])[\.\、\s）)]\s*(.+)')
    # 匹配同一行的多个选项："A. xxx  B. yyy  C. zzz"（修复Bug：旧版只能识别行首第一个）
    RE_OPTION_INLINE = re.compile(r'([A-D])[\.\、\s）)]\s*(.+?)(?=\s+[A-D][\.\、\s）)]|$)', re.I)
    # 匹配填空题的横线
    RE_BLANK   = re.compile(r'(_{2,}|\\underline\{[^}]*\}|\$\$?\s*\\underline\{[^}]*\})')
    # 匹配答案区域
    RE_DETAILS_START = re.compile(r'<details[^>]*>')
    RE_DETAILS_END   = re.compile(r'</details>')
    RE_ANSWER_PREFIX = re.compile(r'^\*?\*?(?:A|答案)(\d+)\**\s*[\.\、：:）)]?\s*(.+)', re.IGNORECASE)
    # 从答案中提取正确选项字母："A4. B。" → "B"（修复Bug：旧版 charAt(0) 取 'A' 而非 'B'）
    RE_ANSWER_LETTER = re.compile(r'A\d+\s*[\.\、：:]\s*([A-Da-d])[\.\、。）]', re.I)
    RE_ANSWER_STANDALONE = re.compile(r'(?:^|\n)\s*([A-Da-d])\s*[\.\、。）]?\s*$', re.M)

    @classmethod
    def parse(cls, md_text: str, file_name: str = "") -> tuple:
        """
        返回 (blocks: list[dict], questions: list[Question])
        block = {"type": "md"|"question", "content": str, "question": Question|None}
        """
        blocks = []
        questions = []
        lines = md_text.split('\n')
        N = len(lines)
        i = 0
        global_qid = 0  # 全局唯一计数器（修复Bug：旧版按section重复Q1导致DOM id冲突）

        while i < N:
            line = lines[i]

            # 检测 ### 题目 X-Y 格式 (如文件09)
            hm3 = re.match(r'^#{1,3}\s*题目\s*(\d+[-–]\d+)\s*$', line.strip())
            if hm3:
                q = cls._parse_heading_question(lines, i, hm3)
                if q:
                    global_qid += 1
                    q.display_id = q.qid   # 保留原始编号用于显示
                    q.qid = str(global_qid) # 全局唯一ID
                    questions.append(q)
                    blocks.append({"type": "question", "question": q})
                    i = cls._find_question_end(lines, i)
                    continue

            # 先检测旧格式: **Q1.** ...  (必须在 RE_QHEADER 之前，因为 RE_QHEADER 会误匹配)
            qm = re.match(r'^\*\*Q(\d+)\.?\*\*\s*(.*)', line.strip())
            if qm:
                q = cls._parse_legacy_question(lines, i, qm)
                if q:
                    global_qid += 1
                    q.display_id = f"Q{q.qid}"  # 原始题号用于显示
                    q.qid = str(global_qid)       # 全局唯一ID
                    questions.append(q)
                    blocks.append({"type": "question", "question": q})
                    i = cls._find_legacy_end(lines, i)
                    continue

            # 检测题型标记行: "题目1（选择题）：..."  "Q1（填空题）..."
            m = cls.RE_QHEADER.match(line.strip())
            if m:
                q = cls._parse_question_block(lines, i, m)
                if q:
                    global_qid += 1
                    q.display_id = f"#{q.qid}"  # 原始题号
                    q.qid = str(global_qid)      # 全局唯一ID
                    questions.append(q)
                    blocks.append({"type": "question", "question": q})
                    i = cls._find_question_end(lines, i)  # 跳到题目块结束
                    continue

            # 普通markdown内容：累积到下一个题目或文件结束
            md_lines = []
            while i < N:
                l = lines[i]
                if cls.RE_QHEADER.match(l.strip()):
                    break
                if re.match(r'^\*\*Q\d+\.?\*\*\s*', l.strip()):
                    break
                if re.match(r'^#{1,3}\s*题目\s*\d+', l.strip()):
                    break
                md_lines.append(l)
                i += 1
            if md_lines:
                blocks.append({"type": "md", "content": '\n'.join(md_lines), "question": None})

        return blocks, questions

    @staticmethod
    def _clean_title(t: str) -> str:
        """清理标题：去除 ** 标记、题型前缀等"""
        t = re.sub(r'^\*{1,2}', '', t).strip()
        t = re.sub(r'^\*{1,2}', '', t).strip()
        t = re.sub(r'^[）)]\s*', '', t)
        # 去除开头的题型标记如 （选择题） （填空题）
        t = re.sub(r'^[（(][^）)]*[）)]\s*[:：]?\s*', '', t)
        return t

    @classmethod
    def _parse_heading_question(cls, lines, start_idx, header_match):
        """解析 ### 题目 X-Y 格式"""
        qnum = header_match.group(1)
        q = Question()
        q.qid = qnum
        q.qtype = QuestionType.CODE  # 默认代码分析题

        # 收集标题后的内容作为题目描述
        title_parts = []
        for j in range(start_idx + 1, min(start_idx + 30, len(lines))):
            l = lines[j]
            if re.match(r'^#{1,3}\s*题目', l.strip()):
                break
            if re.match(r'^#{1,3}\s*参考', l.strip()):
                break
            if l.strip().startswith('<details'):
                break
            # 检查是否是问题行
            q_match = re.match(r'^\*\*问题[：:]?\*\*\s*$', l.strip())
            if q_match:
                # 下一行往往是问题内容
                continue
            title_parts.append(l)

        q.title = cls._clean_title('\n'.join(title_parts[:3]).strip()[:120])  # 取前120字符作为标题

        # 检测代码块
        body_text = '\n'.join(title_parts)
        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', body_text, re.DOTALL)
        if code_match:
            q.code = code_match.group(1).strip()

        # 提取答案
        q.answer, q.answer_letter = cls._extract_answer(lines, start_idx, legacy=True)
        q.explanation = q.answer
        return q

    @classmethod
    def _parse_question_block(cls, lines, start_idx, header_match):
        qnum = header_match.group(1)
        type_hint = header_match.group(2)
        title_part = cls._clean_title(header_match.group(3))

        q = Question()
        q.qid = qnum
        q.title = title_part

        # 判断题型
        if type_hint:
            type_hint = type_hint.replace(" ", "")
            if "选择" in type_hint:
                q.qtype = QuestionType.CHOICE
            elif "填空" in type_hint:
                q.qtype = QuestionType.BLANK
            elif "简答" in type_hint:
                q.qtype = QuestionType.SHORT
            elif "代码" in type_hint:
                q.qtype = QuestionType.CODE
            elif "计算" in type_hint:
                q.qtype = QuestionType.CALC

        # 扫描后续行，收集选项和答案
        body_lines = []
        has_options = False
        for j in range(start_idx + 1, min(start_idx + 50, len(lines))):
            l = lines[j]
            if cls.RE_QHEADER.match(l.strip()):
                break
            if re.match(r'^\*\*Q\d+\.?\*\*\s*', l.strip()):
                break
            if re.match(r'^\*\*题目\s*\d+', l.strip()):
                break
            # 优先检查同行多选项（修复Bug：旧版"A. xxx  B. yyy"只识别A）
            inline_opts = cls.RE_OPTION_INLINE.findall(l.strip())
            if inline_opts and len(inline_opts) >= 2:
                has_options = True
                for lbl, txt in inline_opts:
                    q.options.append((lbl.upper(), txt.strip()))
            else:
                opt_match = cls.RE_OPTION.match(l.strip())
                if opt_match:
                    has_options = True
                    label, text = opt_match.group(1).upper(), opt_match.group(2).strip()
                    q.options.append((label, text))
            body_lines.append(l)

        body_text = '\n'.join(body_lines)

        # 如果没通过标记判断，则根据内容自动判断
        if not type_hint or q.qtype == QuestionType.SHORT:
            if has_options and len(q.options) >= 2:
                q.qtype = QuestionType.CHOICE
            elif cls.RE_BLANK.search(body_text) or cls.RE_BLANK.search(q.title):
                q.qtype = QuestionType.BLANK
            elif any('```' in l for l in body_lines):
                q.qtype = QuestionType.CODE

        # 提取代码块
        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', body_text, re.DOTALL)
        if code_match:
            q.code = code_match.group(1).strip()

        # 在题目块后续行中找答案
        q.answer, q.answer_letter = cls._extract_answer(lines, start_idx)
        q.explanation = q.answer
        return q

    @classmethod
    def _parse_legacy_question(cls, lines, start_idx, header_match):
        qnum = header_match.group(1)
        title_part = cls._clean_title(header_match.group(2) if header_match.lastindex >= 2 else "")

        q = Question()
        q.qid = qnum
        q.title = title_part

        body_lines = [title_part]
        has_options = False
        for j in range(start_idx + 1, min(start_idx + 40, len(lines))):
            l = lines[j]
            if cls.RE_QHEADER.match(l.strip()):
                break
            if re.match(r'^\*\*Q\d+\.?\*\*\s*', l.strip()):
                break
            if l.strip() == '---':
                break
            if cls.RE_DETAILS_START.search(l) or l.strip().startswith('<details'):
                break
            # 优先检查同行多选项（修复Bug：旧版"A. xxx  B. yyy"只识别A）
            inline_opts = cls.RE_OPTION_INLINE.findall(l.strip())
            if inline_opts and len(inline_opts) >= 2:
                has_options = True
                for lbl, txt in inline_opts:
                    q.options.append((lbl.upper(), txt.strip()))
            else:
                opt_match = cls.RE_OPTION.match(l.strip())
                if opt_match:
                    has_options = True
                    label, text = opt_match.group(1).upper(), opt_match.group(2).strip()
                    q.options.append((label, text))
            body_lines.append(l)

        body_text = '\n'.join(body_lines)

        if has_options and len(q.options) >= 2:
            q.qtype = QuestionType.CHOICE
        elif cls.RE_BLANK.search(body_text):
            q.qtype = QuestionType.BLANK
        elif any('```' in l for l in body_lines):
            q.qtype = QuestionType.CODE
        else:
            q.qtype = QuestionType.SHORT

        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', body_text, re.DOTALL)
        if code_match:
            q.code = code_match.group(1).strip()

        q.answer, q.answer_letter = cls._extract_answer(lines, start_idx, legacy=True)
        q.explanation = q.answer
        return q

    @classmethod
    def _extract_answer(cls, lines, from_idx, legacy=False):
        """从 <details> 块或答案行中提取答案文本和正确选项字母"""
        answer_text = ""
        answer_letter = ""

        # 先找 <details> 块
        search_range = range(from_idx, min(from_idx + 100, len(lines)))
        in_details = False
        details_lines = []
        for j in search_range:
            l = lines[j]
            if cls.RE_DETAILS_START.search(l):
                in_details = True
                details_lines = []
                continue
            if in_details:
                if cls.RE_DETAILS_END.search(l):
                    in_details = False
                    break
                details_lines.append(l)

        if details_lines:
            full_details = '\n'.join(details_lines)
            # 清理 HTML 标签和 ** 标记
            clean = re.sub(r'<[^>]+>', '', full_details)
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
            answer_text = clean.strip()

            # 智能提取正确选项字母（修复Bug：旧版 charAt(0) 从"A4. B"取'A'而非'B'）
            m_letter = cls.RE_ANSWER_LETTER.search(clean)
            if m_letter:
                answer_letter = m_letter.group(1).upper()
            if not answer_letter:
                m_letter = cls.RE_ANSWER_STANDALONE.search(clean)
                if m_letter:
                    answer_letter = m_letter.group(1).upper()
            # 宽松匹配："答案： C" 格式
            if not answer_letter:
                m_letter = re.search(r'(?:答案[：:]\s*)?\b([A-Da-d])\b', clean)
                if m_letter:
                    answer_letter = m_letter.group(1).upper()

        # 如果没找到details，则尝试匹配 **答案N** 行
        if not answer_text:
            for j in range(from_idx, min(from_idx + 80, len(lines))):
                l = lines[j]
                am = cls.RE_ANSWER_PREFIX.match(l.strip())
                if am:
                    answer_text = am.group(2).strip()
                    # 后续行也是答案解释
                    expl_lines = [answer_text]
                    for k in range(j + 1, min(j + 20, len(lines))):
                        nl = lines[k].strip()
                        if nl == '---' or cls.RE_QHEADER.match(nl):
                            break
                        if nl:
                            expl_lines.append(nl)
                    answer_text = '\n'.join(expl_lines)
                    m_l = cls.RE_ANSWER_LETTER.search(answer_text)
                    if m_l:
                        answer_letter = m_l.group(1).upper()
                    break

        return answer_text, answer_letter

    @classmethod
    def _find_question_end(cls, lines, start_idx):
        """找到题目块的结束位置"""
        for j in range(start_idx + 1, len(lines)):
            l = lines[j].strip()
            if cls.RE_QHEADER.match(l):
                return j
            if re.match(r'^\*\*Q\d+\.?\*\*\s*', l):
                return j
            if re.match(r'^\*\*题目\s*\d+', l):
                return j
            if re.match(r'^#{1,3}\s*题目\s*\d+', l):
                return j
        return len(lines)

    @classmethod
    def _find_legacy_end(cls, lines, start_idx):
        return cls._find_question_end(lines, start_idx)


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

JS_CODE = r"""
// ===== 全局状态 =====
var currentMode = 'learn';  // learn | practice | exam
var answers = {};           // qid -> answer data
var scores = { total: 0, answered: 0, correct: 0 };
var examSubmitted = false;

// ===== 选择题：选择选项 =====
function selectChoice(qid, label) {
    if (examSubmitted) return;
    var card = document.getElementById('card-' + qid);
    var items = card.querySelectorAll('.option-item');
    var wasSelected = false;
    items.forEach(function(item) {
        if (item.getAttribute('data-label') === label) {
            if (item.classList.contains('selected')) {
                wasSelected = true;
            }
            if (currentMode !== 'learn') {
                item.classList.toggle('selected');
                if (!item.classList.contains('selected')) {
                    // deselected
                    delete answers[qid];
                    updateScore();
                    return;
                }
            } else {
                item.classList.add('selected');
            }
        } else {
            item.classList.remove('selected');
        }
    });
    if (!wasSelected) {
        answers[qid] = { type: 'choice', value: label };
        updateScore();
    }
}

// ===== 填空题：收集输入 =====
function onBlankChange(qid) {
    if (examSubmitted) return;
    var card = document.getElementById('card-' + qid);
    var inputs = card.querySelectorAll('.blank-input');
    var vals = [];
    inputs.forEach(function(inp, idx) { vals.push(inp.value.trim()); });
    answers[qid] = { type: 'blank', value: vals };
    updateScore();
}

// ===== 提交：检查答案 =====
function submitAnswer(qid) {
    if (examSubmitted) return;
    var ansData = answers[qid];
    if (!ansData) {
        alert('请先作答！');
        return;
    }
    var correctAnswer = window.__correctAnswers__[qid] || '';
    var isCorrect = false;

    if (ansData.type === 'choice') {
        var sel = ansData.value.toUpperCase();
        var correct = correctAnswer.trim().toUpperCase();
        // 支持 "B" 或 "B." 或 "B. xxx"
        isCorrect = (sel === correct.charAt(0).toUpperCase());
    } else if (ansData.type === 'blank') {
        var userVals = ansData.value;
        var correctVals = correctAnswer.split('|').map(function(s){return s.trim();});
        // 全部匹配才算对
        isCorrect = true;
        for (var i = 0; i < Math.max(userVals.length, correctVals.length); i++) {
            var u = (userVals[i] || '').toLowerCase().replace(/\s/g, '');
            var c = (correctVals[i] || '').toLowerCase().replace(/\s/g, '');
            if (u !== c && c && u) { isCorrect = false; break; }
        }
    }

    revealAnswer(qid, isCorrect);
    if (currentMode === 'practice') {
        if (!(qid in (window.__submitted__ || {}))) {
            window.__submitted__ = window.__submitted__ || {};
            window.__submitted__[qid] = true;
            scores.answered++;
            if (isCorrect) scores.correct++;
            updateScore();
        }
    }
}

// ===== 显示答案 =====
function revealAnswer(qid, isCorrect) {
    var card = document.getElementById('card-' + qid);
    var answerBox = card.querySelector('.answer-box');
    var label = answerBox.querySelector('.answer-label');
    var items = card.querySelectorAll('.option-item');
    var inputs = card.querySelectorAll('.blank-input');
    var correctAnswer = window.__correctAnswers__[qid] || '';

    answerBox.classList.add('show');
    items.forEach(function(item) { item.classList.add('disabled'); });
    inputs.forEach(function(inp) { inp.disabled = true; });

    // 对于选择题，高亮正确选项
    var correctChar = correctAnswer.trim().charAt(0).toUpperCase();
    items.forEach(function(item) {
        var dl = item.getAttribute('data-label').toUpperCase();
        if (dl === correctChar) {
            item.classList.add('reveal-correct');
        }
        if (item.classList.contains('selected') && dl !== correctChar) {
            item.classList.add('reveal-wrong');
        }
    });

    // 对于填空题，高亮每个输入
    var correctVals = correctAnswer.split('|').map(function(s){return s.trim();});
    inputs.forEach(function(inp, idx) {
        var u = (inp.value || '').trim().toLowerCase().replace(/\s/g, '');
        var c = (correctVals[idx] || '').trim().toLowerCase().replace(/\s/g, '');
        if (c && u === c) { inp.classList.add('correct'); }
        else if (c) { inp.classList.add('wrong'); }
        inp.value = inp.value + '  [' + (correctVals[idx] || '?') + ']';
    });

    if (isCorrect) {
        card.classList.add('correct');
        answerBox.className = 'answer-box show correct-answer';
        label.className = 'answer-label correct';
        label.textContent = '✓ 回答正确！';
    } else {
        card.classList.add('wrong');
        answerBox.className = 'answer-box show wrong-answer';
        label.className = 'answer-label wrong';
        label.textContent = '✗ 回答错误';
    }
}

// ===== 简单显示答案（学习模式） =====
function toggleAnswer(qid) {
    var answerBox = document.getElementById('ans-' + qid);
    var btn = document.getElementById('btn-' + qid);
    if (answerBox.classList.contains('show')) {
        answerBox.classList.remove('show');
        btn.textContent = '显示答案';
    } else {
        answerBox.classList.add('show');
        btn.textContent = '隐藏答案';
    }
}

// ===== 更新分数显示 =====
function updateScore() {
    var answeredEl = document.getElementById('stat-answered');
    var totalEl = document.getElementById('stat-total');
    var correctEl = document.getElementById('stat-correct');
    if (answeredEl) answeredEl.textContent = Object.keys(answers).length;
    if (totalEl) totalEl.textContent = window.__totalQuestions__ || 0;
    if (correctEl) correctEl.textContent = scores.correct;
}

// ===== 考试提交 =====
function submitExam() {
    if (examSubmitted) return;
    if (!confirm('确定要提交试卷吗？提交后无法修改。')) return;
    examSubmitted = true;

    var total = window.__totalQuestions__ || 0;
    var correct = 0;
    for (var qid in answers) {
        var ansData = answers[qid];
        var correctAnswer = window.__correctAnswers__[qid] || '';
        if (ansData.type === 'choice') {
            if (ansData.value.toUpperCase() === correctAnswer.trim().charAt(0).toUpperCase()) {
                correct++;
            }
        } else if (ansData.type === 'blank') {
            var userVals = ansData.value;
            var correctVals = correctAnswer.split('|').map(function(s){return s.trim();});
            var ok = true;
            for (var i = 0; i < Math.max(userVals.length, correctVals.length); i++) {
                var u = (userVals[i] || '').toLowerCase().replace(/\s/g, '');
                var c = (correctVals[i] || '').toLowerCase().replace(/\s/g, '');
                if (u !== c && c && u) { ok = false; break; }
            }
            if (ok) correct++;
        }
    }

    // 显示所有答案
    Object.keys(window.__correctAnswers__).forEach(function(qid) {
        var isCorrect = false;
        var ansData = answers[qid];
        if (ansData && ansData.type === 'choice') {
            isCorrect = (ansData.value.toUpperCase() === window.__correctAnswers__[qid].trim().charAt(0).toUpperCase());
        }
        revealAnswer(qid, isCorrect);
    });

    var score = total > 0 ? Math.round(correct / total * 100) : 0;
    var gradeBar = document.getElementById('grade-bar');
    if (gradeBar) {
        gradeBar.innerHTML = '<div style="text-align:center;padding:20px;">' +
            '<div style="font-size:3em;font-weight:700;color:#1a73e8;">' + score + '</div>' +
            '<div style="font-size:1.2em;color:#666;">分</div>' +
            '<div style="margin-top:8px;color:#888;">答对 ' + correct + ' / 共 ' + total + ' 题</div>' +
            '<div style="margin-top:16px;color:#666;">' +
            (score >= 90 ? '🎉 非常优秀！' : score >= 70 ? '👍 表现不错！' : score >= 60 ? '📚 继续加油！' : '💪 需要多加练习！') +
            '</div></div>';
    }
    window.scrollTo(0, 0);
}

// ===== 随机打乱选项（考试模式） =====
function shuffleOptions() {
    document.querySelectorAll('.option-group').forEach(function(group) {
        var items = Array.prototype.slice.call(group.children);
        for (var i = items.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            group.appendChild(items[j]);
        }
    });
}
"""


# ==================== HTML 生成器 ====================
class HTMLGenerator:
    """根据解析结果生成不同模式的 HTML"""

    @staticmethod
    def generate(blocks, questions, mode="learn", file_name="", theme="亮色") -> str:
        """主入口"""
        if mode == "learn":
            return HTMLGenerator._learn_mode(blocks, questions, file_name, theme)
        elif mode == "practice":
            return HTMLGenerator._practice_mode(questions, file_name, theme)
        elif mode == "exam":
            return HTMLGenerator._exam_mode(questions, file_name, theme)
        return ""

    @staticmethod
    def _base_html(body: str, extra_js: str = "", css: str = "") -> str:
        """生成骨架 HTML"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>深度学习考试题库</title>
    <style>{css or CSS_STYLE}</style>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
    <script>
        window.MathJax = {{
            tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] }},
            svg: {{ fontCache: 'global' }}
        }};
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs2015.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>document.addEventListener('DOMContentLoaded',function(){{hljs.highlightAll();}});</script>
</head>
<body>
{body}
<script>{JS_CODE}</script>
<script>{extra_js}</script>
</body>
</html>"""

    @staticmethod
    def _learn_mode(blocks, questions, file_name, theme="亮色"):
        """学习模式：完整渲染markdown + 可折叠答案"""
        parts = []
        for blk in blocks:
            if blk["type"] == "md":
                html_md = markdown.markdown(blk["content"], extensions=['extra', 'toc', 'nl2br', 'fenced_code', 'codehilite'])
                parts.append(html_md)
            elif blk["type"] == "question":
                q = blk["question"]
                parts.append(HTMLGenerator._question_card(q, mode="learn"))

        body = "\n".join(parts)
        if not body:
            body = "<p style='text-align:center;color:#999;padding:40px;'>该文件中未检测到题目，请切换到其他模式或查看更多内容。</p>"

        css = build_css(theme)
        return HTMLGenerator._base_html(body, f"""
            window.__totalQuestions__ = {len(questions)};
            window.__correctAnswers__ = {json.dumps({q.qid: (q.answer_letter if q.qtype == QuestionType.CHOICE else q.answer) for q in questions}, ensure_ascii=False)};
        """)

    @staticmethod
    def _practice_mode(questions, file_name, theme="亮色"):
        """练习模式：每道题独立交互"""
        cards = [HTMLGenerator._question_card(q, mode="practice") for q in questions]
        body = '<div class="score-bar" id="score-panel">\n'
        body += '  <div class="score-stat"><div class="score-num" id="stat-answered">0</div><div class="score-label">已答</div></div>\n'
        body += '  <div class="score-stat"><div class="score-num" id="stat-total">{}</div><div class="score-label">总题数</div></div>\n'.format(len(questions))
        body += '  <div class="score-stat"><div class="score-num" id="stat-correct">0</div><div class="score-label">答对</div></div>\n'
        body += '</div>\n'
        body += '\n'.join(cards)

        if not cards:
            body = "<p style='text-align:center;color:#999;padding:40px;'>该文件中未检测到题目。</p>"

        css = build_css(theme)
        return HTMLGenerator._base_html(body, f"""
            window.__totalQuestions__ = {len(questions)};
            window.__correctAnswers__ = {json.dumps({q.qid: (q.answer_letter if q.qtype == QuestionType.CHOICE else q.answer) for q in questions}, ensure_ascii=False)};
            window.__submitted__ = {{}};
        """)

    @staticmethod
    def _exam_mode(questions, file_name, theme="亮色"):
        """考试模式：计时、一次性提交"""
        cards = [HTMLGenerator._question_card(q, mode="exam") for q in questions]
        body = '<div class="score-bar" id="grade-bar">\n'
        body += '  <div class="score-stat"><div class="score-num" id="stat-answered">0</div><div class="score-label">已答</div></div>\n'
        body += '  <div class="score-stat"><div class="score-num" id="stat-total">{}</div><div class="score-label">总题数</div></div>\n'.format(len(questions))
        body += '  <div class="score-stat"><div class="score-num" id="timer-display">--:--</div><div class="score-label">用时</div></div>\n'
        body += '  <button class="btn btn-success" onclick="submitExam()" id="submit-exam-btn">📝 提交试卷</button>\n'
        body += '</div>\n'
        body += '\n'.join(cards)

        if not cards:
            body = "<p style='text-align:center;color:#999;padding:40px;'>该文件中未检测到题目。</p>"

        extra_js = f"""
            window.__totalQuestions__ = {len(questions)};
            window.__correctAnswers__ = {json.dumps({q.qid: (q.answer_letter if q.qtype == QuestionType.CHOICE else q.answer) for q in questions}, ensure_ascii=False)};
            var examStart = Date.now();
            setInterval(function() {{
                if (examSubmitted) return;
                var elapsed = Math.floor((Date.now() - examStart) / 1000);
                var m = Math.floor(elapsed / 60);
                var s = elapsed % 60;
                var el = document.getElementById('timer-display');
                if (el) el.textContent = (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
            }}, 1000);
            shuffleOptions();
        """
        css = build_css(theme)
        return HTMLGenerator._base_html(body, extra_js)

    @staticmethod
    def _question_card(q: Question, mode="learn") -> str:
        """生成单道题目的 HTML 卡片"""
        badge_map = {
            QuestionType.CHOICE: ("choice", "选择题"),
            QuestionType.BLANK:  ("blank",  "填空题"),
            QuestionType.SHORT:  ("short",  "简答题"),
            QuestionType.CODE:   ("code",   "代码分析"),
            QuestionType.CALC:   ("calc",   "计算题"),
        }
        badge_class, badge_text = badge_map.get(q.qtype, ("short", "题目"))

        title_html = html_mod.escape(q.title)
        did = html_mod.escape(q.display_id or q.qid)

        parts = []
        parts.append(f'<div class="question-card" id="card-{q.qid}">')
        parts.append(f'  <div class="q-header">')
        parts.append(f'    <span class="q-badge {badge_class}">{badge_text} {did}</span>')
        parts.append(f'    <span class="q-title">{title_html}</span>')
        parts.append(f'  </div>')

        # 代码块
        if q.code:
            escaped_code = html_mod.escape(q.code)
            parts.append(f'  <div class="q-code-block">{escaped_code}</div>')

        if q.qtype == QuestionType.CHOICE:
            parts.append(HTMLGenerator._choice_widget(q, mode))
        elif q.qtype == QuestionType.BLANK:
            parts.append(HTMLGenerator._blank_widget(q, mode))
        else:
            # 简答/计算/代码 → 直接显示答案按钮
            pass

        # 答案区
        answer_html = HTMLGenerator._answer_section(q, mode)
        parts.append(answer_html)
        parts.append('</div>')
        return '\n'.join(parts)

    @staticmethod
    def _choice_widget(q: Question, mode="learn") -> str:
        """生成选择题选项的HTML"""
        parts = ['<div class="option-group">']
        for label, text in q.options:
            safe_label = html_mod.escape(label)
            safe_text  = html_mod.escape(text)
            onclick = f'selectChoice(\'{q.qid}\', \'{safe_label}\')' if mode != "learn" else ''
            parts.append(
                f'<label class="option-item" data-label="{safe_label}"'
                f' onclick="{onclick}">'
                f'<span class="option-label">{safe_label}</span>'
                f'{safe_text}</label>'
            )
        parts.append('</div>')

        if mode == "practice":
            parts.append(f'<button class="btn btn-primary" onclick="submitAnswer(\'{q.qid}\')">提交答案</button>')
        return '\n'.join(parts)

    @staticmethod
    def _blank_widget(q: Question, mode="learn") -> str:
        """生成填空题的输入框HTML"""
        parts = ['<div class="blank-group">']
        # 计算题目中横线/空的数量
        blank_count = len(re.findall(r'(_{2,}|\\underline\{[^}]*\}|\$\$?\s*\\underline\{[^}]*\})', q.title))
        if blank_count == 0:
            blank_count = 1  # 默认至少一个空
        for i in range(blank_count):
            idx = i + 1
            onchange = f'onBlankChange(\'{q.qid}\')' if mode != "learn" else ''
            disabled = 'disabled' if mode == "learn" else ''
            parts.append(
                f'<span class="blank-label">({idx})</span>'
                f'<input type="text" class="blank-input" placeholder="请输入...{idx}" '
                f'oninput="{onchange}" {disabled}/>'
            )
        parts.append('</div>')

        if mode == "practice":
            parts.append(f'<button class="btn btn-primary" onclick="submitAnswer(\'{q.qid}\')">提交答案</button>')
        return '\n'.join(parts)

    @staticmethod
    def _answer_section(q: Question, mode="learn") -> str:
        """生成答案区域的 HTML"""
        escaped_answer = html_mod.escape(q.explanation or q.answer or "暂无答案")
        # 处理换行
        answer_html = escaped_answer.replace('\n', '<br>')

        if mode == "learn":
            ans_id = f"ans-{q.qid}"
            btn_id = f"btn-{q.qid}"
            return (
                f'<button class="btn btn-outline btn-small" id="{btn_id}" onclick="toggleAnswer(\'{q.qid}\')">显示答案</button>\n'
                f'<div class="answer-box normal-answer" id="{ans_id}">\n'
                f'  <div class="answer-label">📖 参考答案：</div>\n'
                f'  <div class="answer-text">{answer_html}</div>\n'
                f'</div>'
            )
        else:
            return (
                f'<div class="answer-box normal-answer">\n'
                f'  <div class="answer-label">📖 参考答案：</div>\n'
                f'  <div class="answer-text">{answer_html}</div>\n'
                f'</div>'
            )


# ==================== 主窗口 ====================
class ExamBank(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dir = os.getcwd()
        self.current_file = None
        self.current_blocks = []
        self.current_questions = []
        self.current_mode = "learn"
        self.file_cache = {}  # 缓存已解析的文件
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

        # 模式切换按钮
        mode_layout = QHBoxLayout()
        mode_label = QLabel("模式：")
        mode_label.setStyleSheet("font-weight:600; font-size:13px;")
        mode_layout.addWidget(mode_label)

        self.btn_learn = QPushButton("📖 学习")
        self.btn_practice = QPushButton("✍️ 练习")
        self.btn_exam = QPushButton("📝 考试")
        for btn in [self.btn_learn, self.btn_practice, self.btn_exam]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setStyleSheet(self._btn_style())
            mode_layout.addWidget(btn)

        self.btn_learn.setChecked(True)
        self.btn_learn.clicked.connect(lambda: self.switch_mode("learn"))
        self.btn_practice.clicked.connect(lambda: self.switch_mode("practice"))
        self.btn_exam.clicked.connect(lambda: self.switch_mode("exam"))

        left_layout.addLayout(mode_layout)

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

        # 使用缓存
        cache_key = md_path
        if cache_key in self.file_cache:
            self.current_blocks, self.current_questions = self.file_cache[cache_key]
        else:
            self.current_blocks, self.current_questions = MDParser.parse(md_text, os.path.basename(md_path))
            self.file_cache[cache_key] = (self.current_blocks, self.current_questions)

        if not self.current_questions:
            # 无题目，直接用 markdown 渲染
            html_body = markdown.markdown(md_text, extensions=['extra', 'toc', 'nl2br', 'fenced_code', 'codehilite'])
            full = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><style>{css or CSS_STYLE}</style></head><body>{html_body}</body></html>"""
            self.web_view.setHtml(full, QUrl.fromLocalFile(os.path.dirname(md_path) + '/'))
            self.statusBar().showMessage(f"[无题目] {os.path.basename(md_path)}")
        else:
            self.render_current()

        self.statusBar().showMessage(
            f"当前文件：{os.path.basename(md_path)}  |  "
            f"题目数：{len(self.current_questions)}  |  "
            f"模式：{'学习' if self.current_mode == 'learn' else '练习' if self.current_mode == 'practice' else '考试'}"
        )

    def render_current(self):
        """根据当前模式渲染"""
        if not self.current_file:
            return
        try:
            html = HTMLGenerator.generate(
                self.current_blocks, self.current_questions,
                mode=self.current_mode,
                file_name=os.path.basename(self.current_file),
                theme=self.current_theme
            )
            self.web_view.setHtml(html, QUrl.fromLocalFile(os.path.dirname(self.current_file) + '/'))
        except Exception as e:
            self.web_view.setHtml(f"<p style='color:red;padding:20px;'>渲染出错：{e}</p>")

    # ---------- 模式切换 ----------
    def switch_mode(self, mode):
        self.current_mode = mode
        for btn, m in [(self.btn_learn, "learn"), (self.btn_practice, "practice"), (self.btn_exam, "exam")]:
            btn.setChecked(m == mode)
        # 切换模式时清缓存，强制重新解析
        if self.current_file:
            cache_key = self.current_file
            if cache_key in self.file_cache:
                md_path = self.current_file
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                    self.current_blocks, self.current_questions = MDParser.parse(md_text, os.path.basename(md_path))
                    self.file_cache[cache_key] = (self.current_blocks, self.current_questions)
                except:
                    pass
            self.render_current()

        mode_names = {"learn": "学习", "practice": "练习", "exam": "考试"}
        self.statusBar().showMessage(f"已切换到{mode_names.get(mode, mode)}模式 | {self.current_theme}主题")

    def _on_theme_change(self, name):
        """主题下拉框切换"""
        if name not in THEMES:
            return
        self.current_theme = name
        CUSTOM_THEME["bg"] = ""
        CUSTOM_THEME["text"] = ""
        if self.current_file:
            self._render_file(self.current_file)

    def _custom_bg(self):
        """自定义背景颜色（继承自 MyFirstQt.py 的 QColorDialog 功能）"""
        color = QColorDialog.getColor()
        if color.isValid():
            CUSTOM_THEME["bg"] = color.name()
            if self.current_file:
                self._render_file(self.current_file)
            self.statusBar().showMessage(f"自定义背景: {color.name()}")

    def _custom_text(self):
        """自定义文字颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            CUSTOM_THEME["text"] = color.name()
            if self.current_file:
                self._render_file(self.current_file)
            self.statusBar().showMessage(f"自定义文字: {color.name()}")

    def _reset_theme(self):
        """重置自定义颜色为当前主题默认值"""
        CUSTOM_THEME["bg"] = ""
        CUSTOM_THEME["text"] = ""
        if self.current_file:
            self._render_file(self.current_file)
        self.statusBar().showMessage(f"已重置为 {self.current_theme} 默认色")


# ==================== 启动入口 ====================
def main():
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
