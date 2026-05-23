#!/bin/bash
# =========================================
#   MD互动练习器 - Linux 打包脚本
# =========================================

set -e
echo "========================================"
echo "  MD互动练习器 - Linux 打包脚本"
echo "========================================"
echo ""

if ! python3 -c "import PyInstaller" &>/dev/null; then
    echo "[提示] PyInstaller 未安装，正在安装..."
    pip3 install pyinstaller
fi

echo "[1/3] 清理旧的构建文件..."
rm -rf build dist *.spec

echo "[2/3] 开始打包..."
pyinstaller \
    --onedir \
    --windowed \
    --name="MD练习器" \
    --collect-all PyQt5 \
    --collect-all PyQt5.QtWebEngineWidgets \
    --hidden-import=markdown \
    --hidden-import=PyQt5.QtWebChannel \
    md_practice.py

echo ""
echo "[3/3] 打包完成！"
echo ""
echo "输出文件: dist/MD练习器/MD练习器"
echo ""
echo "★ 运行:"
echo "   把 .md 题库文件复制到 dist/MD练习器/ 目录"
echo "   ./dist/MD练习器/MD练习器"
echo ""
echo "★ 注意:"
echo "   考试模式需要联网（MathJax CDN）"
echo ""
