"""知识库机器人 - 共享配置
API Key 从环境变量或 .env 文件读取，不硬编码在代码中。
优先级：环境变量 > .env 文件 > 默认空值
"""
import os


def _load_dotenv():
    """简单的 .env 文件解析器（无需额外依赖）。"""
    env = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return env
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_DOTENV = _load_dotenv()


def _get(key, default=""):
    """优先读环境变量，其次读 .env 文件，最后用默认值。"""
    return os.environ.get(key, _DOTENV.get(key, default))


# ===== 运行环境 =====
APP_ENV = _get("APP_ENV", "dev")  # dev / test / prod

# 各环境的启动配置
ENV_CONFIG = {
    "dev": {
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "open_browser": True,
        "log_level": "info",
    },
    "test": {
        "host": "0.0.0.0",
        "port": 8001,
        "reload": False,
        "open_browser": False,
        "log_level": "info",
    },
    "prod": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "open_browser": False,
        "log_level": "warning",
    },
}

# ===== 大模型（回答生成）=====
LLM_API_KEY = _get("LLM_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"              # 默认模型：便宜、快、通用
LLM_REASONER_MODEL = "deepseek-chat"     # 推理模型：复杂推理时路由到此，可改为 deepseek-reasoner（贵2倍）

# ===== Embedding（向量化）=====
EMBED_API_KEY = _get("EMBED_API_KEY")
EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

# ===== 向量库 =====
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "knowledge_base"

# ===== 文档处理 =====
DOCS_DIR = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# ===== Agent =====
MAX_TOOL_RESULT_LENGTH = 3000  # 单个工具返回结果的最大字符数，防止撑爆上下文

# ===== Redis（会话存储，可选）=====
# 不配置或连不上时自动回退到内存存储（dev 环境够用）
REDIS_URL = _get("REDIS_URL", "")  # 例如 redis://localhost:6379/0
SESSION_EXPIRE_SECONDS = 86400  # 会话过期时间，默认 24 小时

# ===== 启动时检查密钥 =====
if not LLM_API_KEY:
    print("[警告] 未设置 LLM_API_KEY，请在 .env 文件或环境变量中配置")
if not EMBED_API_KEY:
    print("[警告] 未设置 EMBED_API_KEY，请在 .env 文件或环境变量中配置")
