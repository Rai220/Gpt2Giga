import argparse
import http.server
import json
import logging
import os
import socketserver
import time
import uuid
from typing import Any, Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, ChatCompletion

# Load environment variables
env_path = find_dotenv(".env")
load_dotenv(env_path)

# Initialize GigaChat client
def init_gigachat_client() -> GigaChat:
    """
    Initializes the GigaChat client with configurations from environment variables.
    """
    verify_ssl_certs = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "False") != "False"
    profanity_check = os.getenv("GIGACHAT_PROFANITY_CHECK", "False") != "False"
    return GigaChat(verify_ssl_certs=verify_ssl_certs, profanity_check=profanity_check, timeout=600)

giga = init_gigachat_client()

def remove_none(obj: Any) -> Any:
    """
    Recursively removes None values from dictionaries and lists.

    Args:
        obj: The object to clean.

    Returns:
        The cleaned object with None values removed.
    """
    if isinstance(obj, dict):
        return {k: remove_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_none(v) for v in obj if v is not None]
    else:
        return obj

def transform_input_data(data: dict) -> Tuple[Chat, Optional[str]]:
    """
    Transforms the input data from the client to the format expected by GigaChat API.

    Args:
        data: The input data dictionary.

    Returns:
        A tuple containing the Chat object and the GPT model name.
    """
    gpt_model = data.pop("model", None)
    temperature = data.pop("temperature", None)
    if temperature == 0:
        data["top_p"] = 0
    elif temperature and temperature > 0:
        data["temperature"] = temperature

    if "functions" not in data and data.get("tools"):
        data["functions"] = [
            tool["function"] for tool in data.get("tools", []) if tool["type"] == "function"
        ]

    for i, message in enumerate(data["messages"]):
        message.pop("name", None)
        # No non-first system messages available.
        if message["role"] == "system" and i > 0:
            message["role"] = "user"
        if message["role"] == "tool":
            message["role"] = "function"
            message["content"] = json.dumps(message.get("content", ""), ensure_ascii=False)
        if message["content"] is None:
            message["content"] = ""

    chat = Chat.parse_obj(data)
    return chat, gpt_model

def process_gigachat_response(giga_resp: ChatCompletion, gpt_model: str) -> dict:
    """
    Processes the response from GigaChat API and transforms it to the format expected by the client.

    Args:
        giga_resp: The response from GigaChat API.
        gpt_model: The GPT model name.

    Returns:
        A dictionary formatted as the client's expected response.
    """
    giga_dict = json.loads(giga_resp.json())
    giga_dict = remove_none(giga_dict)

    for choice in giga_dict["choices"]:
        choice["index"] = 0
        choice["logprobs"] = None
        choice["message"]["refusal"] = None
        if choice["message"]["role"] == "assistant":
            if choice["message"].get("function_call"):
                arguments = json.dumps(
                    choice["message"]["function_call"]["arguments"],
                    ensure_ascii=False,
                )
                choice["message"]["function_call"] = {
                    "name": choice["message"]["function_call"]["name"],
                    "arguments": arguments,
                }
                if choice["message"].get("content") == "":
                    choice["message"]["content"] = None
                choice["message"].pop("functions_state_id", None)
                choice["message"]["refusal"] = None

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

def send_to_gigachat(data: dict) -> dict:
    """
    Sends the transformed data to GigaChat API and processes the response.

    Args:
        data: The input data dictionary.

    Returns:
        The processed response dictionary.
    """
    chat, gpt_model = transform_input_data(data)
    giga_resp = giga.chat(chat)
    result = process_gigachat_response(giga_resp, gpt_model)
    return result

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """
    Handles HTTP requests and proxies them to the GigaChat API after transforming the data.
    """
    giga = None
    verbose = False

    def __init__(self, *args, **kwargs):
        self.giga = self.__class__.giga
        self.verbose = self.__class__.verbose
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in ("/models", "/v1/models"):
            self.handle_models_request()
        else:
            self.handle_proxy_request()

    def do_POST(self):
        self.handle_proxy_request()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self._send_CORS_headers()
        self.end_headers()

    def _send_CORS_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def handle_proxy_request(self):
        """
        Handles proxy requests by forwarding them to GigaChat after transforming the data.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length) if content_length else b''
            request_body_text = request_body.decode("utf-8", errors="replace")
            json_body = json.loads(request_body_text)
            stream = json_body.pop("stream", False)

            if self.verbose:
                logging.info(f"Request Headers: {self.headers}")
                logging.info("Request Body:")
                logging.info(json_body)

            giga_resp = send_to_gigachat(json_body)
            response_body = json.dumps(giga_resp, ensure_ascii=False, indent=2).encode("utf-8")

            if self.verbose:
                logging.info("Response:")
                logging.info(json.dumps(giga_resp, ensure_ascii=False, indent=2))

            self.send_response(200)
            self._send_CORS_headers()

            if stream:
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                to_send = {
                    "id": giga_resp["id"],
                    "object": "chat.completion.chunk",
                    "created": giga_resp["created"],
                    "model": giga_resp["model"],
                    "system_fingerprint": giga_resp.get("system_fingerprint", ""),
                    "choices": [
                        {
                            "index": 0,
                            "delta": giga_resp["choices"][0]["message"],
                            "logprobs": None,
                            "finish_reason": giga_resp["choices"][0]["finish_reason"],
                        }
                    ],
                }

                self.wfile.write(f"data: {json.dumps(to_send, ensure_ascii=False)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\r\n\r\n")
                self.wfile.write(b"\r\n\r\n")
            else:
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_body)))
                self.send_header('Connection', 'keep-alive')
                self.send_header("Access-Control-Expose-Headers", "X-Request-ID")
                self.send_header("OpenAI-Organization", "user-1234567890")
                self.send_header("OpenAI-Processing-Ms", "100")
                self.send_header("OpenAI-Version", "2020-10-01")
                self.send_header("X-RateLimit-Limit-Requests", "10000")
                self.send_header("X-RateLimit-Limit-Tokens", "50000000")
                self.send_header("X-RateLimit-Remaining-Requests", "9999")
                self.send_header("X-RateLimit-Remaining-Tokens", "49999945")
                self.send_header("X-RateLimit-Reset-Requests", "6ms")
                self.send_header("X-RateLimit-Reset-Tokens", "0s")
                self.send_header("X-Request-ID", "req_" + str(uuid.uuid4()))
                self.end_headers()
                self.wfile.write(response_body)
        except Exception as e:
            logging.error(f"Error processing the request: {e}", exc_info=True)
            self.send_error(500, f"Error processing the request: {e}")

    def handle_models_request(self):
        """
        Handles requests to /models or /v1/models by returning the models.json content.
        """
        try:
            with open("models.json", "r", encoding="utf-8") as f:
                models_data = json.load(f)

            response_data = json.dumps(models_data, ensure_ascii=False).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_data)))
            self._send_CORS_headers()
            self.end_headers()
            self.wfile.write(response_data)
        except FileNotFoundError:
            self.send_error(404, "models.json not found")
        except Exception as e:
            logging.error(f"Error handling /v1/models request: {e}", exc_info=True)
            self.send_error(500, "Internal Server Error")

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """This class allows to handle requests in separate threads."""

def run_proxy_server(host: str, port: int, verbose: bool):
    """
    Runs the proxy server.

    Args:
        host: The host to listen on.
        port: The port to listen on.
        verbose: Enables verbose logging if True.
    """
    server_address = (host, port)
    ProxyHandler.verbose = verbose
    ProxyHandler.giga = giga

    logging_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=logging_level)

    httpd = ThreadingHTTPServer(server_address, ProxyHandler)
    print(f"Serving HTTP proxy on {host} port {port}...")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(
        description="Gpt2Giga converter proxy. Use GigaChat instead of OpenAI GPT models"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("PROXY_HOST", "localhost"),
        help="Host to listen on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PROXY_PORT", "8090")),
        help="Port to listen on",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=os.getenv("GPT2GIGA_VERBOSE", "False") != "False",
        help="enable verbose logging"
    )

    args = parser.parse_args()
    run_proxy_server(args.host, args.port, args.verbose)

if __name__ == "__main__":
    main()
