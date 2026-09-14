"""
Prompt A/B 测试脚本
==================
运行：python ab_test.py

用同一批测试用例，跑不同版本的 system prompt，对比哪个效果好。
指标：回答关键词命中率、幻觉率、平均回答长度。

企业级用法：每次改 prompt 都跑一遍，用数据决定留哪个版本，而不是靠感觉。
"""
import asyncio
import re

from config import TOP_K
from common import chat
from retriever import hybrid_search

# ==================== 测试用例（示例，替换为你自己的问题）====================
TEST_CASES = [
    {"question": "什么是 RAG？", "expected_keywords": ["检索", "知识库"]},
    {"question": "Python 是什么？", "expected_keywords": ["解释型", "编程语言"]},
    {"question": "什么是 Function Calling？", "expected_keywords": ["工具", "函数"]},
    {"question": "向量数据库是干什么的？", "expected_keywords": ["向量", "检索"]},
]

# ==================== Prompt 版本 ====================
# 每个版本是一个 system prompt，对比哪个效果好
PROMPT_VERSIONS = {
    "A_极简版": (
        "你是一个基于资料回答问题的助手。"
        "只依据提供的资料回答，不编造。"
        "如果资料中没有答案，说'根据现有资料无法回答'。"
    ),
    "B_标准版": (
        "你是一个基于资料回答问题的助手，只依据提供的资料回答，不编造。\n"
        "规则：\n"
        "1. 回答必须基于资料中的信息\n"
        "2. 如果资料中没有答案，明确说'根据现有资料无法回答'\n"
        "3. 回答要简洁，直接给答案，不要多余的客套话"
    ),
    "C_强制引用版": (
        "你是一个基于资料回答问题的助手，只依据提供的资料回答，不编造。\n"
        "规则：\n"
        "1. 回答中的每个事实必须来自资料，禁止使用你自己的知识\n"
        "2. 关键信息后用（片段N）标注来源\n"
        "3. 如果资料中没有答案，明确说'根据现有资料无法回答'，禁止推测\n"
        "4. 回答要简洁，直接给答案"
    ),
    "D_角色代入版": (
        "你是一位严谨的技术顾问，擅长从技术文档中提取关键信息并给出准确回答。\n"
        "工作原则：\n"
        "1. 只依据提供的资料回答，绝不编造资料中没有的信息\n"
        "2. 回答结构清晰，先给结论，再给依据\n"
        "3. 资料中没有的内容，明确告知无法回答，不做推测\n"
        "4. 用专业但易懂的语言表达"
    ),
}


def check_hallucination(answer, context_docs):
    """简单幻觉检测（和 evaluate.py 一致）。"""
    answer_terms = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', answer))
    context_text = " ".join(context_docs)
    context_terms = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', context_text))
    stopwords = {
        "什么", "怎么", "为什么", "可以", "需要", "应该", "能够", "可能",
        "已经", "还是", "不是", "没有", "就是", "我们", "你们", "他们",
        "这个", "那个", "一个", "一种", "进行", "通过", "使用", "利用",
        "基于", "对于", "关于", "以及", "或者", "如果", "因为", "所以",
        "但是", "然后", "这样", "那样", "根据资料", "根据现有", "资料中",
        "具体来说", "简单来说", "总的来说", "也就是说", "换句话说",
        "首先", "其次", "最后", "此外", "另外", "例如", "比如",
        "包括", "主要", "重要", "关键", "核心", "基本", "通常",
        "大模型", "以下是", "回答如下", "总结一下", "综上所述",
    }
    answer_terms -= stopwords
    hallucinated = {w for w in (answer_terms - context_terms) if len(w) >= 3}
    return len(hallucinated) > 0


async def run_version(version_name, system_prompt):
    """跑一个 prompt 版本，返回指标。"""
    answer_hits = 0
    hallucination_count = 0
    total_length = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        # 检索（所有版本用相同的检索结果，只对比 prompt 的影响）
        results = await hybrid_search(case["question"], top_k=TOP_K)
        docs = [r["content"] for r in results]
        context = "\n".join(f"[片段{i + 1}] {doc}" for i, doc in enumerate(docs))

        prompt = f"""请根据以下资料回答问题。

资料：
{context}

问题：{case['question']}
"""

        answer = await chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ])

        # 关键词命中
        missing = [kw for kw in case["expected_keywords"] if kw not in answer]
        if len(missing) == 0:
            answer_hits += 1

        # 幻觉检测
        if check_hallucination(answer, docs):
            hallucination_count += 1

        total_length += len(answer)

    return {
        "version": version_name,
        "answer_hit_rate": answer_hits / total,
        "hallucination_rate": hallucination_count / total,
        "avg_length": total_length / total,
    }


async def main():
    print(f"Prompt A/B 测试，共 {len(TEST_CASES)} 个测试用例，{len(PROMPT_VERSIONS)} 个 prompt 版本\n")
    print("=" * 70)

    results = []
    for name, prompt in PROMPT_VERSIONS.items():
        print(f"正在测试: {name} ...")
        result = await run_version(name, prompt)
        results.append(result)
        print(f"  完成: 关键词命中 {result['answer_hit_rate']*100:.0f}%, "
              f"幻觉 {result['hallucination_rate']*100:.0f}%, "
              f"平均长度 {result['avg_length']:.0f}字")

    # 对比表格
    print(f"\n{'=' * 70}")
    print(f"{'版本':<16} {'关键词命中率':<14} {'幻觉率':<10} {'平均长度':<10} {'综合评分'}")
    print("-" * 70)

    # 综合评分：命中率权重 0.6，幻觉率权重 0.3（越低越好），简洁度权重 0.1
    best = None
    best_score = -1
    for r in results:
        # 简洁度：长度越短分越高（归一化到 0-1）
        max_len = max(x["avg_length"] for x in results)
        brevity = 1 - r["avg_length"] / max_len if max_len > 0 else 0
        score = r["answer_hit_rate"] * 0.6 + (1 - r["hallucination_rate"]) * 0.3 + brevity * 0.1
        marker = ""
        if score > best_score:
            best_score = score
            best = r["version"]
            marker = " ← 推荐"
        print(f"{r['version']:<16} {r['answer_hit_rate']*100:>10.0f}%   "
              f"{r['hallucination_rate']*100:>6.0f}%   "
              f"{r['avg_length']:>6.0f}字   "
              f"{score:.2f}{marker}")

    print("=" * 70)
    print(f"\n推荐版本: {best}")
    print("\n说明：")
    print("  综合评分 = 关键词命中率×0.6 + (1-幻觉率)×0.3 + 简洁度×0.1")
    print("  你可以根据业务需求调整权重，比如客服场景更看重低幻觉。")


if __name__ == "__main__":
    asyncio.run(main())
