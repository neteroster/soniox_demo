1. uv sync 如果你使用 uv；或手动安装下列依赖：

```python
dependencies = [
    "numpy>=2.0.0",
    "proc-tap>=0.1.0",
    "psutil>=5.9.0",
    "pyaudiowpatch>=0.2.12",
    "python-dotenv>=1.1.0",
    "rich>=14.0.0",
    "soniox>=2.2.0",
]
```

2. 在 .env 填写 SONIOX API Key

3. 启动：

- 捕获全局音频： `uv run python main_direct.py --mode system`
- 捕获应用音频，示例：`uv run python main_direct.py --mode process --name chrome.exe` 然后选择主进程（没有任何参数的那个）即可
