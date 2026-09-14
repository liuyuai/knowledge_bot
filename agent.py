"""
Agent 核心：大模型自主决策调用工具 —— 全异步版本
====================================
和之前的 RAG 区别：
- RAG：每次都先查知识库再回答（不管需不需要）
- Agent：大模型自己判断要不要查库、要不要算数学、要不要查时间，自己决定调什么工具

工具列表：
- search_knowledge_base(query)  搜索私有知识库
- calculator(expression)        计算数学表达式
- get_current_time()            获取当前时间
- reindex()                     重建知识库（需人工确认）
"""
import json
import os
import re
from datetime import datetime

from config import CHROMA_PATH, COLLECTION_NAME, TOP_K, MAX_TOOL_RESULT_LENGTH, LLM_MODEL, LLM_REASONER_MODEL
from common import embed, chat, chat_with_tools, chat_stream
from retriever import hybrid_search

# ==================== 工具定义（给大模型看的"说明书"）====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索私有知识库，返回相关文档片段。知识库包含AI学习笔记、Python教程、菜谱（红烧肉、小鸡炖蘑菇）、面试资料等。回答任何事实性问题前必须先调用此工具，以知识库内容为准，即使你认为自己知道答案。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。当问题需要数学计算时必须使用，如'123*456'、'(2+3)*4'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当问题涉及现在几点、今天几号、星期几等时间相关问题时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气。当问题涉及天气、气温、下雨、温度等气象相关问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如'北京'、'上海'、'大连'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reindex",
            "description": "重建知识库向量索引。当用户要求更新知识库、重新建库、重新索引、刷新文档时使用。此操作会重新读取 docs/ 目录下所有文件并重建向量数据库，耗时较长，需要用户确认后才能执行。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# 需要人工确认的工具（Human-in-the-Loop）
HUMAN_APPROVAL_TOOLS = {"reindex"}

SYSTEM_PROMPT = """你是一个知识库助手，可以使用工具来帮助回答问题。

规则：
1. 回答任何事实性问题前，先调用 search_knowledge_base 搜索私有知识库。知识库包含AI学习笔记、Python教程、菜谱（红烧肉、小鸡炖蘑菇等）、面试资料等多种内容。即使你觉得自己知道答案，也应该先查库，以知识库内容为准。
2. 任何数学计算必须调用 calculator，禁止自己心算或估算
3. 遇到时间问题，调用 get_current_time
4. 只有纯问候、闲聊（如"你好""谢谢"）可以不调用工具直接回答
5. 每次最多调用一个工具，拿到结果后再决定下一步
6. 优先基于工具返回的结果回答。如果知识库中没有相关内容，可以基于你的常识回答，但要在回答开头注明"（以下内容来自通用知识，非知识库内容）"
7. 只回答用户的问题，回答要简洁，不要主动提出额外建议、追问或延伸服务
8. 调用工具前，先用一句话说明你为什么要调用这个工具、期望得到什么信息（思考过程）
9. 遇到天气问题，调用get_weather
10. 所有输出（包括思考过程和最终回答）必须使用中文，禁止使用英文
"""

# ==================== 工具实现 ====================
async def _search_knowledge_base(query):
    """混合检索（向量+BM25）+ 重排序，返回 JSON 字符串。"""
    results = await hybrid_search(query)
    return json.dumps(
        [{"source": r["source"], "content": r["content"]} for r in results],
        ensure_ascii=False,
    )


def _calculator(expression):
    """安全计算数学表达式。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符，只允许数字和 +-*/()."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def _get_current_time():
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _get_weather(city):
    """获取指定城市的天气。"""
    try:
        import httpx
        r = httpx.get(f"https://wttr.in/{city}?lang=zh&format=3", timeout=10)
        r.raise_for_status()
        return r.text.strip()
    except Exception as e:
        return f"获取天气失败: {e}"


def _reindex():
    """重建知识库：重新读取 docs/ 下所有文件，重建向量数据库。"""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode == 0:
        output = result.stdout
        return f"知识库重建完成。{output.strip()[-200:]}"
    else:
        return f"知识库重建失败：{result.stderr.strip()[-200:]}"


# 工具名 → 实现函数 的映射
TOOL_IMPLS = {
    "search_knowledge_base": _search_knowledge_base,
    "calculator": _calculator,
    "get_current_time": _get_current_time,
    "get_weather": _get_weather,
    "reindex": _reindex,
}


# ==================== 工具执行的安全封装 ====================
def _truncate_result(result, max_len=MAX_TOOL_RESULT_LENGTH):
    """截断过长的工具结果，防止撑爆上下文窗口。"""
    result = str(result)
    if len(result) > max_len:
        return result[:max_len] + f"\n...（结果已截断，原长度{len(result)}字符）"
    return result


async def _safe_execute_tool(name, args):
    """安全执行工具：try/except 包裹 + 结果截断。工具崩了不拖垮整个 Agent。
    自动识别同步/异步工具函数。"""
    if name not in TOOL_IMPLS:
        return f"错误：未知工具 {name}"
    try:
        result = TOOL_IMPLS[name](**args)
        # 如果是协程（异步函数），await 它
        if hasattr(result, "__await__"):
            result = await result
        return _truncate_result(result)
    except Exception as e:
        return f"工具执行失败：{name}，错误：{e}"


def _verify_tool_result(name, result):
    """
    工具调用验证：检查工具返回结果是否有效。
    空结果或异常结果不直接喂给模型，而是返回明确提示，防止模型对着空结果瞎编。
    返回 (验证后的结果, 是否通过验证)
    """
    if name == "search_knowledge_base":
        try:
            data = json.loads(result)
            if not data:
                return "知识库中没有找到相关内容。请基于常识回答，或告知用户知识库中暂无此信息。", False
            if not isinstance(data, list):
                return f"检索结果格式异常：{str(result)[:100]}", False
            return result, True
        except (json.JSONDecodeError, TypeError):
            return "检索结果解析失败，请换个关键词重试。", False

    if name == "calculator":
        if not result or "错误" in result or "失败" in result:
            return f"计算失败：{result}", False
        return result, True

    # 其他工具默认通过
    return result, True


async def _reflect(question, answer, tool_results):
    """
    反省（Reflection）：模型生成回答后，再调一次模型检查自己的回答。
    检查：是否基于工具结果？有没有编造？需不需要修正？
    返回 (修正后的回答, 是否做了修正)
    """
    if not tool_results:
        return answer, False

    tools_text = "\n".join([f"[{i+1}] {r}" for i, r in enumerate(tool_results)])

    prompt = f"""请检查以下回答是否准确基于提供的工具结果。

用户问题：{question}

工具返回的结果：
{tools_text}

模型的回答：
{answer}

请检查：
1. 回答中的事实是否都能在工具结果中找到依据？
2. 有没有编造工具结果中不存在的信息？
3. 如果回答有错误或遗漏，请给出修正后的回答。

如果回答没有问题，只返回"OK"两个字。
如果需要修正，只返回修正后的完整回答，不要解释。"""

    try:
        reflection = await chat([
            {"role": "system", "content": "你是一个严格的回答审查员，只返回OK或修正后的回答。"},
            {"role": "user", "content": prompt},
        ])
        reflection = reflection.strip()

        if reflection.upper() == "OK" or reflection.startswith("OK"):
            return answer, False
        else:
            print(f"\n{'='*50}")
            print(f"🔍 反省修正: 原回答 {len(answer)} 字 -> 修正后 {len(reflection)} 字")
            print(f"{'─'*50}")
            print(f"  修正后: {reflection[:200]}{'...' if len(reflection) > 200 else ''}")
            print(f"{'='*50}\n")
            return reflection, True
    except Exception as e:
        print(f"[Agent] 反省失败，保留原回答: {e}")
        return answer, False


# ==================== 模型路由 ====================
def route_model(question):
    """
    根据问题类型选择最合适的模型。
    - 数学计算 → deepseek-reasoner（推理强、工具调用更听话）
    - 其他 → deepseek-chat（便宜快）
    """
    # 匹配数字 + 运算符的模式（如 "123*456"、"1+1"、"(2+3)*4"）
    if re.search(r'\d+\s*[+\-*/×÷]\s*\d+', question):
        return LLM_REASONER_MODEL
    # 包含"计算""等于多少""算一下"等数学关键词
    if any(k in question for k in ["计算", "等于多少", "算一下", "多少加", "多少乘", "多少减", "多少除"]):
        return LLM_REASONER_MODEL
    return LLM_MODEL


# ==================== Agent 主循环 ====================
async def run_agent(question, max_steps=5):
    """
    Agent 主循环（异步）：大模型自主决策，最多 max_steps 步。
    返回 (最终回答, 完整消息历史, 调用步骤列表)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = []
    last_call = None
    selected_model = route_model(question)

    for step in range(1, max_steps + 1):
        response = await chat_with_tools(messages, TOOLS, model=selected_model)

        if response.get("tool_calls"):
            messages.append(response)
            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                    result = "错误：工具参数不是合法 JSON，请重新调用并提供正确参数。"
                    steps.append(f"第{step}步：调用 {name}(参数解析失败)")
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call["id"], "content": result,
                    })
                    continue

                args_str = json.dumps(args, sort_keys=True)
                if last_call == (name, args_str):
                    result = "你刚刚已经用相同参数调用过这个工具了，请基于已有结果给出回答，不要重复调用。"
                    steps.append(f"第{step}步：重复调用 {name}({args})，已跳过")
                else:
                    result = await _safe_execute_tool(name, args)
                    last_call = (name, args_str)
                    steps.append(f"第{step}步：调用 {name}({args})")

                print(f"[Agent] 第{step}步 → {name}({args})")
                messages.append({
                    "role": "tool", "tool_call_id": tool_call["id"], "content": result,
                })
        else:
            answer = response.get("content", "")
            messages.append(response)
            if not steps:
                steps.append("直接回答（未调用工具）")
            return answer, messages, steps

    return "（达到最大步数，未能完成任务）", messages, steps


def _extract_sources(messages):
    """从 Agent 的消息历史中提取知识库来源文件。"""
    sources = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                results = json.loads(msg["content"])
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and r.get("source") and r["source"] not in sources:
                            sources.append(r["source"])
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[Agent] 工具结果解析失败，无法提取来源: {e}，内容前100字: {msg['content'][:100]}")
    return sources


async def rewrite_query(question, history):
    """
    查询改写（异步）：多轮对话时，把带代词/省略的问题改写成独立完整的查询。
    """
    if not history:
        return question

    recent = history[-6:] if len(history) > 6 else history
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in recent if msg.get("role") in ("user", "assistant")
    ])

    prompt = f"""根据对话历史，把用户的当前问题改写成一个独立完整的查询。
要求：
1. 把代词（它、这个、那个、上面）替换成具体事物
2. 补充省略的上下文
3. 只返回改写后的查询，不要解释，不要加引号

对话历史：
{history_text}

当前问题：{question}

改写后的查询："""

    try:
        rewritten = await chat([
            {"role": "system", "content": "你是一个查询改写助手，只返回改写后的查询文本。"},
            {"role": "user", "content": prompt},
        ])
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"[Agent] 查询改写失败，使用原问题: {e}")

    return question


async def compress_history(history, threshold=14):
    """
    对话历史摘要压缩（异步）：当历史超过 threshold 条时，用大模型把最早的一半
    总结成一段摘要，替换掉详细对话。
    """
    if len(history) <= threshold:
        return history

    split_point = len(history) // 2
    old_messages = history[:split_point]
    recent_messages = history[split_point:]

    dialogue_lines = []
    for msg in old_messages:
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            dialogue_lines.append(f"{role}: {msg.get('content', '')}")
    history_text = "\n".join(dialogue_lines)

    prompt = f"""请把以下对话历史总结成一段简洁的摘要，保留关键信息：
- 用户问了什么核心问题
- 模型给出的关键结论和事实
- 后续对话可能用到的上下文
不要超过200字，只返回摘要文本。

对话历史：
{history_text}

摘要："""

    try:
        summary = await chat([
            {"role": "system", "content": "你是一个对话摘要助手，只返回摘要文本。"},
            {"role": "user", "content": prompt},
        ])
        summary = summary.strip()
        if not summary:
            return history

        compressed = [
            {"role": "system", "content": f"[之前的对话摘要] {summary}"},
        ] + recent_messages

        print(f"[Agent] 历史压缩: {len(history)}条 -> {len(compressed)}条")
        return compressed
    except Exception as e:
        print(f"[Agent] 历史压缩失败，保留原历史: {e}")
        return history


async def run_agent_stream(question, history=None, max_steps=5):
    """
    流式 Agent 主循环（异步生成器）。
    参数:
      question: 当前用户问题
      history: 之前的对话消息列表（不含 system prompt），用于多轮对话
    yield 事件字典：
      {"type": "thought", "content": "..."}         模型的思考过程（CoT）
      {"type": "tool", "name": ..., "args": ...}    调用了工具
      {"type": "token", "content": "..."}            最终回答的一个 token
      {"type": "reflection", "content": "..."}       反省修正后的回答
      {"type": "done", "answer": ..., "sources": ..., "steps": ..., "history": ...}  完成
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)

    # 查询改写：多轮对话时把带代词的问题改写成独立完整的查询
    original_question = question
    if history:
        question = await rewrite_query(question, history)
        if question != original_question:
            print(f"[Agent] 查询改写: '{original_question}' -> '{question}'")

    messages.append({"role": "user", "content": question})
    steps = []
    last_call = None
    gave_final_answer = False
    tool_results_list = []

    # 模型路由：根据问题类型选择模型
    selected_model = route_model(question)
    if selected_model != LLM_MODEL:
        print(f"[路由] 问题 '{question[:30]}' → {selected_model}（推理模型）")

    # ---- 工具决策阶段（非流式）----
    for step in range(1, max_steps + 1):
        response = await chat_with_tools(messages, TOOLS, model=selected_model)

        if response.get("tool_calls"):
            # 显式 CoT：只有调工具时，content 才是思考过程
            thought = response.get("content", "").strip()
            if thought:
                print(f"\n{'='*50}")
                print(f"💭 模型思考（第{step}步）:")
                print(f"{'─'*50}")
                print(f"  {thought}")
                print(f"{'='*50}\n")
                yield {"type": "thought", "content": thought}

            messages.append(response)
            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                    result = "错误：工具参数不是合法 JSON，请重新调用并提供正确参数。"
                    steps.append(f"第{step}步：调用 {name}(参数解析失败)")
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call["id"], "content": result,
                    })
                    continue

                args_str = json.dumps(args, sort_keys=True)
                steps.append(f"第{step}步：调用 {name}({args})")
                yield {"type": "tool", "name": name, "args": args}

                if last_call == (name, args_str):
                    result = "你刚刚已经用相同参数调用过这个工具了，请基于已有结果给出回答，不要重复调用。"
                else:
                    result = await _safe_execute_tool(name, args)
                    last_call = (name, args_str)

                # 工具调用验证
                result, verified = _verify_tool_result(name, result)
                if not verified:
                    print(f"  ⚠️  工具验证未通过: {name} -> {result[:80]}")

                # tool_results_list.append(f"{name}({args}) -> {result[:200]}")  # 反省用

                messages.append({
                    "role": "tool", "tool_call_id": tool_call["id"], "content": result,
                })
            continue

        # 没有 tool_calls = 模型准备给最终回答
        gave_final_answer = True
        break

    # ---- 兜底：达到最大步数还在调工具 ----
    if not gave_final_answer:
        answer = "（经过多步工具调用后仍未能得出结论，请换个方式提问或简化问题。）"
        sources = _extract_sources(messages)
        if not steps:
            steps.append("直接回答（未调用工具）")
        conversation_history = (history or []) + [
            {"role": "user", "content": original_question},
            {"role": "assistant", "content": answer},
        ]
        yield {"type": "done", "answer": answer, "sources": sources, "steps": steps, "history": conversation_history}
        return

    # ---- 最终回答阶段（流式）----
    answer_parts = []
    async for token in chat_stream(messages, model=selected_model):
        answer_parts.append(token)
        yield {"type": "token", "content": token}
    answer = "".join(answer_parts)

    # ---- 反省（已注释）----
    # answer, was_corrected = await _reflect(question, answer, tool_results_list)
    # if was_corrected:
    #     yield {"type": "reflection", "content": answer}

    sources = _extract_sources(messages)
    if not steps:
        steps.append("直接回答（未调用工具）")

    conversation_history = (history or []) + [
        {"role": "user", "content": original_question},
        {"role": "assistant", "content": answer},
    ]

    yield {"type": "done", "answer": answer, "sources": sources, "steps": steps, "history": conversation_history}
