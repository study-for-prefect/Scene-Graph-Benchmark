import requests
import json


def test_ollama_connection(model_name="qwen2.5vl:7b-q4_K_M"):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个测试节点，只输出JSON格式。"},
            {"role": "user", "content": "测试连接。请输出包含键 'status' 且值为 200 的 JSON。"}
        ],
        "stream": False,
        "format": "json"
    }

    print(f"正在向 {url} 发送请求，模型: {model_name} ...")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        content = result["message"]["content"]

        print("\n连接成功。")
        print(f"原始返回文本:\n{content}")

    except requests.exceptions.ConnectionError:
        print("\n连接失败：Ollama 服务未启动。请在终端执行 'ollama serve' 或启动 Ollama 客户端。")
    except requests.exceptions.Timeout:
        print("\n连接超时：模型加载时间过长或硬件算力不足。")
    except Exception as e:
        print(f"\n发生未知错误：{e}")


if __name__ == "__main__":
    # 若运行的是其他模型，如 qwen2.5:7b，请修改此处参数
    test_ollama_connection(model_name="qwen2.5vl:7b-q4_K_M")