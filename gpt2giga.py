import argparse
import http.server
import json
import logging
import os
import socketserver
import time
import urllib.error
import urllib.request
import uuid

from dotenv import find_dotenv, load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, ChatCompletion

env_path = find_dotenv(".env")
load_dotenv(env_path)

giga = GigaChat(verify_ssl_certs=False, profanity_check=False)


def send_to_gigachat(data: dict) -> ChatCompletion:
    try:
        gpt_model = data.pop("model", None)
        temperature = data.get("temperature", 1e-15)
        if temperature == 0:
            temperature = 1e-15
        data["temperature"] = temperature
        if "functions" not in data and data.get("tools", None):
            data["functions"] = []
            for tool in data.get("tools", []):
                if tool["type"] == "function":
                    data["functions"].append(tool["function"])
        for message in data["messages"]:
            message.pop("name", None)
            if message["role"] == "tool":
                message["role"] = "function"
                message["content"] = json.dumps(
                    message.get("content", ""), ensure_ascii=False
                )
            if message["content"] == None:
                message["content"] = ""

        chat = Chat.parse_obj(data)

        giga_resp = giga.chat(chat)
        giga_dict = json.loads(giga_resp.json())

        # Remove all none from giga_dict in all levels
        def remove_none(d):
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(v) for v in d if v is not None]
            else:
                return d

        giga_dict = remove_none(giga_dict)

        for choise in giga_dict["choices"]:
            choise["index"] = 0
            choise["logprobs"] = None
            choise["message"]["refusal"] = None
            if choise["message"]["role"] == "assistant":

                if choise["message"].get("function_call", None):
                    arg_txt = json.dumps(
                        choise["message"]["function_call"]["arguments"],
                        ensure_ascii=False,
                    )
                    choise["message"]["function_call"] = {
                        "name": choise["message"]["function_call"]["name"],
                        "arguments": arg_txt
                    }
                    if choise["message"].get("content", None) == "":
                        choise["message"]["content"] = None
                    choise["message"].pop("functions_state_id", None)
                    choise["message"]["refusal"] = None

        result = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time() * 1000),
            "model": gpt_model,
            "choices": giga_dict["choices"],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "system_fingerprint": f"fp_{uuid.uuid4()}",
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

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def handle_proxy_request(self):
        try:
            # Чтение запроса от клиента
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length) if content_length else None
            request_body_text = request_body.decode("utf-8", errors="replace")
            accept_encoding = self.headers.get("accept-encoding", "")
            json_body = json.loads(request_body_text)
            stream = json_body.pop("stream", False)
            use_gzip = "gzip" in accept_encoding
            
            if self.server.verbose:
                logging.info(f"Request Headers: {self.headers}")
                logging.info(f"Request Body:")
                logging.info(json_body)

            giga_resp = send_to_gigachat(json_body)

            try:
                resp = json.dumps(giga_resp, ensure_ascii=False, indent=2).encode("utf-8")
                # resp = resp + b"\r\n"

                if self.server.verbose:
                    print("\nResponse:")
                    print(json.dumps(giga_resp, ensure_ascii=False, indent=2))

                self.send_response(200)
                
                # if use_gzip:
                #     out = io.BytesIO()
                #     with gzip.GzipFile(fileobj=out, mode="w") as f:
                #         f.write(resp)
                #     resp = out.getvalue()
                #     self.send_header("Content-Encoding", "gzip")
                
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.send_header("Connection", "keep-alive")
                self.send_header("access-control-expose-headers", "X-Request-ID")
                
                # self.send_header("transfer-encoding", "chunked")
                
                self.send_header("openai-organization", "user-1234567890")
                self.send_header("openai-processing-ms", str(100))
                self.send_header("openai-version", "2020-10-01")
                self.send_header("x-ratelimit-limit-requests", "10000")
                self.send_header("x-ratelimit-limit-tokens", "50000000")
                self.send_header("x-ratelimit-remaining-requests", "9999")
                self.send_header("x-ratelimit-remaining-tokens", "49999945")
                self.send_header("x-ratelimit-reset-requests", "6ms")
                self.send_header("x-ratelimit-reset-token", "0s")
                self.send_header("x-request-id", "req_" + str(uuid.uuid4()))
                self.end_headers()
                self.wfile.write(resp)
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
        description="Gpt2Giga converter proxy. Use GigaChat instead of OpenAI GPT models"
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
        default=os.getenv("GPT2GIGA_VERBOSE", "False") != "False",
        help="Включает вывод запросов и ответов",
    )

    args = parser.parse_args()

    run_proxy_server(args.host, args.port, args.verbose)
