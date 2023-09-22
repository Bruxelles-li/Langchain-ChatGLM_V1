import websocket
import json


# 这是一个回调函数，会在连接成功后收到“欢迎语”后触发
def on_message(ws, message):
    received_messages = message
    print(f"收到消息: {message}")


# 这是一个回调函数，会在连接建立成功后触发
def on_open(ws):
    print("连接成功！发送数据...")
    data = {
        "question": "推荐10部科幻电影",
        "history": [],
        "max_length": 2048,
        "top_p": 0.7,
        "temperature": 0.1
    }
    ws.send(json.dumps(data))


# 替换为你自己的 websocket 服务器地址
websocket_url = "ws://114.115.221.14:7861/local_doc_qa/stream_chat"

# 创建 websocket 连接
ws = websocket.WebSocketApp(websocket_url,
                            on_message=on_message,
                            on_open=on_open)

ws.run_forever()
