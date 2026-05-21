NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")

# Ollama 本地配置
QWEN_API_KEY = "ollama"  # 任意非空字符串
QWEN_BASE_URL = "http://localhost:11434/v1"  # Ollama 默认的 OpenAI 兼容路径
MODEL_NAME = "qwen2.5vl:3b"  # 替换为 ollama list 中显示的本地模型名称