"""
RAG 评估脚本（异步版）
======================
运行：python evaluate.py

评估四个指标：
1. 检索命中率：期望的文档是否出现在检索结果中
2. 回答关键词命中率：回答中是否包含期望的关键词
3. 幻觉率：回答中是否出现了检索结果里没有的信息
4. 平均检索排名：期望文档排在第几位（MRR 简化版）

测试集在下方 TEST_CASES 中，根据你的知识库内容添加更多用例。
"""
import asyncio
import json

from config import TOP_K
from common import chat
from retriever import hybrid_search

# ==================== 测试集（示例）====================
# 每个用例：问题 + 期望检索到的来源文件 + 期望回答中包含的关键词
# 使用时替换为你自己知识库中的真实文档名和问题。
# expected_source 必须和 docs/ 中的文件名完全一致（含扩展名）。
TEST_CASES = [
    {
        "question": "什么是 RAG？",
        "expected_source": "示例文档.md",
        "expected_keywords": ["检索", "知识库"],
    },
    {
        "question": "Python 是什么？",
        "expected_source": "示例文档.md",
        "expected_keywords": ["解释型", "编程语言"],
    },
    {
        "question": "什么是 Function Calling？",
        "expected_source": "示例文档.md",
        "expected_keywords": ["工具", "函数"],
    },
    {
        "question": "向量数据库是干什么的？",
        "expected_source": "示例文档.md",
        "expected_keywords": ["向量", "检索"],
    },
]


def build_prompt(question, docs):
    context = "\n".join(f"[片段{i + 1}] {doc}" for i, doc in enumerate(docs))
    return f"""请根据以下资料回答问题。如果资料中没有答案，请说"根据现有资料无法回答"。

资料：
{context}

问题：{question}
"""


JUDGE_PROMPT = """你是一个严格的事实核查员。请判断以下回答是否完全由参考资料支持。

判断标准：
1. 回答中的每个事实性陈述，都必须能在参考资料中找到依据
2. 回答中可以包含合理的总结和归纳，但不能引入资料中没有的新事实
3. 如果回答说"根据现有资料无法回答"，视为通过
4. 允许使用"首先""其次""综上所述"等连接词，不算幻觉

请只返回 JSON，不要返回其他文字：
{{"pass": true/false, "reason": "简短说明原因"}}

参考资料：
{context}

待核查回答：
{answer}
"""


async def check_hallucination(answer, context_docs):
    """
    LLM-as-judge 幻觉检测：再调一次大模型当裁判。
    比启发式关键词比对准确得多，但成本翻倍（每个用例多一次 API 调用）。
    返回 (是否通过, 原因)
    """
    context = "\n".join(f"[片段{i + 1}] {doc}" for i, doc in enumerate(context_docs))
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)

    try:
        judge_result = await chat([
            {"role": "system", "content": "你是一个严格的事实核查员，只返回 JSON。"},
            {"role": "user", "content": prompt},
        ])
        # 解析 JSON（模型可能返回 markdown 代码块，做容错）
        import json as _json
        cleaned = judge_result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        data = _json.loads(cleaned.strip())
        return data.get("pass", False), data.get("reason", "")
    except Exception as e:
        # 裁判调用失败，默认标记为"未检测"，不误报
        return None, f"裁判调用失败: {e}"


async def evaluate():
    retrieval_hits = 0
    answer_hits = 0
    hallucination_cases = 0
    total = len(TEST_CASES)
    ranks = []

    print(f"开始评估，共 {total} 个测试用例\n")
    print("=" * 60)

    for i, case in enumerate(TEST_CASES):
        print(f"\n[{i + 1}/{total}] 问题: {case['question']}")

        # ---- 1. 评估检索（用完整的混合检索流水线）----
        results = await hybrid_search(case["question"], top_k=TOP_K)
        sources = [r["source"] for r in results]
        docs = [r["content"] for r in results]

        retrieval_ok = case["expected_source"] in sources
        if retrieval_ok:
            retrieval_hits += 1
            rank = sources.index(case["expected_source"]) + 1
            ranks.append(rank)
        else:
            rank = "-"
        print(f"  检索: {'✓' if retrieval_ok else '✗'}  排名: {rank}  实际来源: {sources}")

        # ---- 2. 评估回答 ----
        prompt = build_prompt(case["question"], docs)
        answer = await chat([
            {"role": "system", "content": "你是一个基于资料回答问题的助手，只依据提供的资料回答，不编造。"},
            {"role": "user", "content": prompt},
        ])

        missing = [kw for kw in case["expected_keywords"] if kw not in answer]
        answer_ok = len(missing) == 0
        if answer_ok:
            answer_hits += 1
        print(f"  回答: {'✓' if answer_ok else '✗ 缺少: ' + str(missing)}")

        # ---- 3. 幻觉检测（LLM-as-judge，默认关闭，需要时取消注释）----
        # 注意：启用后每个用例多一次 API 调用，评估成本翻倍
        # hallucination_pass, hallucination_reason = await check_hallucination(answer, docs)
        # if hallucination_pass is False:
        #     hallucination_cases += 1
        #     print(f"  幻觉: ⚠ 未通过 - {hallucination_reason}")
        # elif hallucination_pass is True:
        #     print(f"  幻觉: ✓ 通过")
        # else:
        #     print(f"  幻觉: ? 未检测（{hallucination_reason}）")
        print(f"  幻觉: - （已禁用，启用请取消上方注释）")

        print(f"  回答内容: {answer[:100]}{'...' if len(answer) > 100 else ''}")

    # ---- 总结 ----
    print(f"\n{'=' * 60}")
    print(f"评估结果（共 {total} 题）")
    print(f"  检索命中率:       {retrieval_hits}/{total}  ({retrieval_hits / total * 100:.0f}%)")
    print(f"  回答关键词命中率: {answer_hits}/{total}  ({answer_hits / total * 100:.0f}%)")
    print(f"  幻觉检出率:       {hallucination_cases}/{total}  ({hallucination_cases / total * 100:.0f}%)")
    if ranks:
        avg_rank = sum(ranks) / len(ranks)
        print(f"  平均检索排名:     {avg_rank:.1f}（越小越好，1=第一名）")
    print(f"{'=' * 60}")

    if retrieval_hits < total:
        print("\n[检索优化建议]")
        print("  1. 调整分块大小（CHUNK_SIZE）")
        print("  2. 增加 TOP_K 数量")
        print("  3. 检查知识库文档内容是否覆盖该问题")
        print("  4. 调整重排序阈值（MIN_RERANK_SCORE）")
    if answer_hits < total:
        print("\n[回答优化建议]")
        print("  1. 优化 system prompt")
        print("  2. 检查检索到的片段是否包含足够信息")
        print("  3. 在知识库中补充相关内容")
    if hallucination_cases > 0:
        print("\n[幻觉优化建议]")
        print("  1. 在 system prompt 中强调'只依据资料回答，不编造'")
        print("  2. 增加检索片段数量，让模型有更多依据")
        print("  3. 对回答做后处理校验")


if __name__ == "__main__":
    asyncio.run(evaluate())
