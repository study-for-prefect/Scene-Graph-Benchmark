import openai
import config

client = openai.OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)


def text_to_cypher(user_instruction):
    """利用本地 Qwen 模型将自然语言输入转化为符合严格 Schema 约束的 Cypher 查询语句"""

    # 显式 Schema 约束，防止模型凭空捏造 Label 和关系
    system_prompt = """你是一个 Neo4j Cypher 专家。请严格按照给定的图谱 Schema 生成查询语句。

【图谱 Schema 规范】
1. 所有节点的标签（Label）必须为 `:Entity`。禁止使用其他任何标签。
2. 节点包含两个核心属性：
   - `class`: 字符串，代表物体类别（例如 "building", "table", "counter", "bottle", "cap"）。
   - `id`: 字符串，代表实例编号（例如 "IMG0_OBJ11", "IMG0_OBJ29"）。
3. 关系类型（Relationship）必须全大写，仅支持以下类型：`ABOVE`, `AT`, `BEHIND`, `HAS`, `IN_FRONT_OF`, `UNDER`。

【查询生成规范】
1. 用户若提及组合名称（如 "building_IMG0_OBJ11"），必须在 WHERE 子句中通过字符串拼接进行精确定位：
   `WHERE s.class + "_" + s.id = "building_IMG0_OBJ11"`
2. 统一输出字段。RETURN 子句必须严格命名为：source_node, relation_type, target_node。
3. 只输出 Cypher 语句本身，不要包含 ```cypher 等任何 Markdown 标记或多余解释。

【模板参考】
MATCH (s:Entity)-[r]->(t:Entity)
WHERE s.class + "_" + s.id = "用户提及的物体"
RETURN (s.class + "_" + s.id) AS source_node, type(r) AS relation_type, (t.class + "_" + t.id) AS target_node
LIMIT 20
"""

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户指令：{user_instruction}"}
        ],
        temperature=0.1  # 保持低随机性
    )
    return response.choices[0].message.content.strip()


def graph_reasoning(user_instruction, subgraph_context):
    """结合从图数据库检索出的真实子图拓扑数据，利用本地 Qwen 模型执行空间关系推导"""
    system_prompt = "你是一个严谨的机器人空间拓扑推理专家。请严格基于提供的局部知识图谱数据回答问题。如果没有检索到相关的三元组数据，请直接回答‘依据当前图谱无法推导’，严禁引入外部假设或生成幻觉。"
    prompt = f"【局部图谱上下文】\n{subgraph_context}\n\n【用户指令】\n{user_instruction}\n\n请逐步推导给出最终结论。"

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content