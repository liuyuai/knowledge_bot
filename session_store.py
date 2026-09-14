"""
会话存储：优先用 Redis，连不上时回退到内存字典。

Redis 模式：多进程共享、重启不丢、自动过期
内存模式：单进程、重启丢失（dev 环境够用）
"""
import json
import logging

from config import REDIS_URL, SESSION_EXPIRE_SECONDS

logger = logging.getLogger(__name__)

_redis_client = None
_use_redis = False


def _init_redis():
    """尝试连接 Redis，失败则回退内存模式。"""
    global _redis_client, _use_redis
    if not REDIS_URL:
        logger.info("[会话存储] 未配置 REDIS_URL，使用内存模式")
        return
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()  # 测试连接
        _use_redis = True
        logger.info(f"[会话存储] Redis 已连接: {REDIS_URL}")
    except Exception as e:
        logger.warning(f"[会话存储] Redis 连接失败，回退内存模式: {e}")
        _redis_client = None
        _use_redis = False


# 模块加载时初始化一次
_init_redis()

# 内存模式的存储
_memory_store = {}


def get_history(session_id):
    """读取会话历史，返回 list 或 None。"""
    if not session_id:
        return None
    if _use_redis:
        data = _redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else None
    return _memory_store.get(session_id)


def save_history(session_id, history):
    """保存会话历史，带过期时间。"""
    if not session_id:
        return
    if _use_redis:
        _redis_client.setex(
            f"session:{session_id}",
            SESSION_EXPIRE_SECONDS,
            json.dumps(history, ensure_ascii=False),
        )
    else:
        _memory_store[session_id] = history


def delete_session(session_id):
    """删除会话。"""
    if not session_id:
        return
    if _use_redis:
        _redis_client.delete(f"session:{session_id}")
    elif session_id in _memory_store:
        del _memory_store[session_id]


def has_session(session_id):
    """判断会话是否存在。"""
    if not session_id:
        return False
    if _use_redis:
        return _redis_client.exists(f"session:{session_id}") > 0
    return session_id in _memory_store


def storage_mode():
    """返回当前存储模式，用于调试。"""
    return "redis" if _use_redis else "memory"
