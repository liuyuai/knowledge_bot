"""
知识库机器人 - 统一启动脚本
支持 dev / test / prod 三种环境。

用法：
    python dev.py              # 默认 dev 环境
    python dev.py dev          # 本地开发（127.0.0.1:8000，自动重载，自动开浏览器）
    python dev.py test         # 测试环境（0.0.0.0:8001，无重载）
    python dev.py prod         # 生产环境（0.0.0.0:8000，无重载，warning 日志）

也可以通过环境变量指定：
    APP_ENV=prod python dev.py

优先级：命令行参数 > 环境变量 > 默认 dev
"""
import sys
import threading
import webbrowser

import uvicorn

from config import APP_ENV, ENV_CONFIG


def get_env():
    """确定运行环境：命令行参数优先，其次环境变量。"""
    valid = ("dev", "test", "prod")
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in valid:
            return arg
        print(f"[警告] 未知环境 '{arg}'，可用: {valid}，使用默认 dev")
    return APP_ENV if APP_ENV in valid else "dev"


def main():
    env = get_env()
    cfg = ENV_CONFIG[env]

    print("=" * 50)
    print(f"  知识库机器人  [{env}] 环境")
    print(f"  地址: http://{cfg['host']}:{cfg['port']}")
    print(f"  自动重载: {'开启' if cfg['reload'] else '关闭'}")
    print(f"  自动开浏览器: {'开启' if cfg['open_browser'] else '关闭'}")
    print(f"  日志级别: {cfg['log_level']}")
    print("=" * 50)
    print()

    # dev 环境自动打开浏览器
    if cfg["open_browser"]:
        url = f"http://127.0.0.1:{cfg['port']}"
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "api:app",
        host=cfg["host"],
        port=cfg["port"],
        reload=cfg["reload"],
        log_level=cfg["log_level"],
    )


if __name__ == "__main__":
    main()
