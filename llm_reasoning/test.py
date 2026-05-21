import requests


def simple_chat_test(model_name="qwen2.5vl:7b-q4_K_M"):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": "你好，这是一次普通对话测试。请回复收到。",
        "stream": False
    }

    print(f"正在测试普通对话，使用模型: {model_name}")

    try:
        response = requests.post(url, json=payload)

        # 捕捉 HTTP 错误状态码
        if response.status_code != 200:
            print(f"HTTP 错误码: {response.status_code}")
            print(f"Ollama 返回信息: {response.text}")
            return

        result = response.json()
        print("\n测试成功，模型回复:")
        print(result["response"])

    except requests.exceptions.ConnectionError:
        print("连接失败：Ollama 服务未启动。")
    except Exception as e:
        print(f"发生异常：{e}")


if __name__ == "__main__":
    simple_chat_test(model_name="qwen2.5vl:7b-q4_K_M")