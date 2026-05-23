@echo off
chcp 65001 >nul
echo ========================================
echo   MD互动练习器 - Windows 打包脚本
echo ========================================
echo.

:: 检查 PyInstaller
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] PyInstaller 未安装，正在安装...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [错误] 安装失败，请手动: pip install pyinstaller
        pause
        exit /b 1
    )
)

echo [1/3] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "*.spec" del /q "*.spec"

echo [2/3] 开始打包...
:: 使用 --onedir 更稳定，输出文件夹
pyinstaller ^
    --onedir ^
    --windowed ^
    --name="MD练习器" ^
    --collect-all PyQt5 ^
    --collect-all PyQt5.QtWebEngineWidgets ^
    --hidden-import=markdown ^
    --hidden-import=PyQt5.QtWebChannel ^
    md_practice.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 打包失败！可能原因：
    echo   1. 缺少依赖：pip install markdown PyQt5 PyQtWebEngine
    echo   2. 杀毒软件拦截，暂时关闭后重试
    pause
    exit /b 1
)

echo [3/3] 打包完成！
echo.
echo 输出文件: dist\MD练习器\MD练习器.exe
echo.
echo ★ 使用说明:
echo   1. 把 .md 题库文件复制到 dist\MD练习器\ 目录下
echo   2. 双击 MD练习器.exe 运行
echo   3. 如初次报错缺少 MSVCP140.dll，需安装 VC++ 运行库
echo.
echo ★ 考试/练习模式需要联网（MathJax数学公式、代码高亮CDN）
echo.
pause
