import socketserver
import http.server
import urllib.request
import urllib.error
import argparse
import os
import logging
import json


from gigachat import GigaChat
from gigachat.models import Chat, ChatCompletion


from dotenv import find_dotenv, load_dotenv

env_path = find_dotenv(".env")
load_dotenv(env_path)

giga = GigaChat(model="GigaChat-Pro", verify_ssl_certs=False)


def send_to_gigachat(data: dict) -> ChatCompletion:
    try:
        gpt_model = data.pop("model", None)
        if 'functions' not in data:
            data['functions'] = []
            for tool in data.get('tools', []):
                if tool['type'] == 'function':
                    data['functions'].append(tool['function'])
        for message in data['messages']:
            if message['role'] == 'tool':
                message['role'] = 'function'
                message['content'] = json.dumps(message.get('content', ""), ensure_ascii=False)
            message['function_call_id'] = message.get('tool_call_id', None)
            if message['content'] == None:
                message['content'] = ''

        chat = Chat.parse_obj(data)

        giga_resp = giga.chat(chat)
        giga_dict = json.loads(giga_resp.json())
        
        for choise in giga_dict["choices"]:
            if choise["message"]["role"] == "assistant":
                if choise["message"].get('function_call', None):
                    arg_txt = json.dumps(choise["message"]["function_call"]["arguments"], ensure_ascii=True)
                    choise["message"]["tool_calls"] = [
                        {
                            "id": choise["message"]["functions_state_id"],
                            "type": "function",
                            "function": {
                                "name": choise["message"]["function_call"]["name"],
                                "arguments": arg_txt,
                            },
                        }
                    ]
        
        result = {
            "id": "chatcmpl-AIZ9eSj8SVnkS0IhxdtTmot5z5NaX",
            "object": "chat.completion",
            "created": 1728988402,
            "model": gpt_model,
            "choices": giga_dict["choices"],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "system_fingerprint": None,
        }
        return result
    except Exception as e:
        logging.error(f"Error processing the request: {e}", exc_info=True)
        return None

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.handle_proxy_request()

    def do_POST(self):
        self.handle_proxy_request()

    def handle_proxy_request(self):
        try:
            # Чтение запроса от клиента
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length) if content_length else None
            request_body_text = request_body.decode("utf-8", errors="replace")
            json_body = json.loads(request_body_text)

            if self.server.verbose:
                logging.info(f"Request Body: {request_body_text}")

            giga_resp = send_to_gigachat(json_body)

            try:
                self.send_response(200)
                self.end_headers()

                self.wfile.write(
                    json.dumps(giga_resp, ensure_ascii=False).encode("utf-8")
                )

            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(e.read())

        except Exception as e:
            self.send_error(500, f"Error processing the request: {e}")
        finally:
            self.wfile.flush()
            self.connection.close()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Этот класс позволяет использовать потоки для обработки запросов."""

    def __init__(self, server_address, RequestHandlerClass, verbose):
        super().__init__(server_address, RequestHandlerClass)
        self.verbose = verbose


def run_proxy_server(host, port, verbose):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    httpd = ThreadingHTTPServer((host, port), ProxyHandler, verbose)
    print(f"Serving HTTP proxy on {host} port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Асинхронный HTTP-прокси с поддержкой стриминга."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("PROXY_HOST", "localhost"),
        help="Хост для прослушивания",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PROXY_PORT", "8090")),
        help="Порт для прослушивания",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=os.getenv("GPT2GIGA_VERBOSE", True),
        help="Включает вывод запросов и ответов",
    )

    args = parser.parse_args()

    run_proxy_server(args.host, args.port, args.verbose)
