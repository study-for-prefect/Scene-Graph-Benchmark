from neo4j import GraphDatabase
import config

def query_subgraph(cypher_query):
    """根据输入的 Cypher 语句，检索 Neo4j 并返回标准三元组格式文本"""
    with GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH) as driver:
        with driver.session() as session:
            try:
                result = session.run(cypher_query)
                triplets = []
                for record in result:
                    src = record.get("source_node") or "unknown"
                    rel = record.get("relation_type") or "associated"
                    tgt = record.get("target_node") or "unknown"
                    triplets.append(f'("{src}", "{rel}", "{tgt}")')
                return "\n".join(triplets)
            except Exception as e:
                return f"Cypher 执行错误: {str(e)}"