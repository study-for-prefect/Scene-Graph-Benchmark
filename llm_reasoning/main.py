import qwen_service
import neo4j_client


def run_pipeline(instruction):
    print(f"[1/4] 接收指令: {instruction}")

    # 1. 文本转换为 Cypher 语句
    cypher_query = qwen_service.text_to_cypher(instruction)
    print(f"[2/4] 生成 Cypher 语句:\n{cypher_query}\n")

    # 2. 从图数据库检索子图
    subgraph_data = neo4j_client.query_subgraph(cypher_query)
    print(f"[3/4] 检索出的子图拓扑:\n{subgraph_data}\n")

    # 3. 结合上下文进行大模型推理
    result = qwen_service.graph_reasoning(instruction, subgraph_data)
    print(f"[4/4] 推理决策结果:\n{result}")


if __name__ == "__main__":
    # 项目级闭环测试用例
    test_instruction = "分析 building_IMG0_OBJ11 后面有哪些具体的家具和瓶子"
    run_pipeline(test_instruction)