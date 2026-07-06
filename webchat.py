"""
webchat.py — local browser chat UI for interactively testing a TinyLLM checkpoint.

The CLI demo (chat_demo.py) only proves the pipeline runs -- it doesn't let you poke at
it yourself, which is the actual point of testing "does turn 2's answer track what I
just typed, or is it a memorized script for this opener." This serves the same
generation logic (chat_demo.generate_reply, so there is exactly one place that builds
qa/chatml prompts) over a tiny stdlib-only HTTP server plus a static chat page, so you
can type arbitrary follow-ups in a browser and watch the *live* answer come back.

Stateless by design: the browser resends the full conversation history with every
request and the server rebuilds the prompt from scratch each time (chat_demo.
render_prompt_prefix) -- no session/conversation state lives on the server between
requests, the same mechanic real chat APIs use (see data_utils.encode_conversation's
docstring on why the model is only ever scored on assistant spans).

No extra dependencies -- uses only Python's stdlib http.server, so nothing to pip
install beyond what training already needs.

A reasoning-capable (chatml, joint multitask) checkpoint may answer with a
<THINK>...</THINK> trace before the final answer -- chat_demo._split_reasoning() pulls
that out token-side, and the page renders it as a collapsed "Thoughts" section you can
expand, Claude/ChatGPT-style, instead of showing it inline with the answer.

USAGE:
    python webchat.py --checkpoint checkpoints/multitask_chatml/final.pt \\
        --tokenizer_path checkpoints/tokenizer.json --format chatml
    Then open http://127.0.0.1:8765 in a browser.
"""

import argparse
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from chat_demo import generate_reply, load_model, render_prompt_prefix
from tokenizer import BPETokenizer

STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webchat_ui.html")


def make_handler(model, tokenizer, device, format: str, checkpoint_path: str, tokenizer_path: str):
    n_params = sum(p.numel() for p in model.parameters())

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                with open(STATIC_HTML_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/info":
                self._send_json(200, {
                    "checkpoint": checkpoint_path,
                    "tokenizer_path": tokenizer_path,
                    "format": format,
                    "params": f"{n_params / 1e6:.2f}M params",
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                message = str(body.get("message", "")).strip()
                history = [
                    (str(turn["question"]), str(turn["answer"]))
                    for turn in body.get("history", [])
                ]
                if not message:
                    self._send_json(400, {"error": "empty message"})
                    return

                temperature = float(body.get("temperature", 0.0))
                top_k = int(body.get("top_k", 1))
                max_new_tokens = int(body.get("max_new_tokens", 40))

                prompt_prefix = render_prompt_prefix(format, history)
                result = generate_reply(
                    model, tokenizer, device, prompt_prefix, message, format,
                    max_new_tokens, temperature, top_k,
                )
                # "full" (reasoning included) is what the browser should store back into
                # its history for the *next* request -- render_prompt_prefix rebuilds
                # context from history's "answer" field, and if the client instead sent
                # back the display-only "answer" (reasoning stripped), the model would
                # lose its own reasoning trace from the transcript it conditions on.
                self._send_json(200, {
                    "answer": result["answer"],
                    "reasoning": result["reasoning"],
                    "full": result["full"],
                })
            except Exception as e:  # noqa: BLE001 -- surface any failure to the browser instead of a dropped connection
                self._send_json(500, {"error": str(e)})

        def log_message(self, fmt, *args):
            print(f"[webchat] {self.address_string()} - {fmt % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local browser chat UI for a TinyLLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    parser.add_argument("--format", choices=["qa", "chatml"], default="qa")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no_browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, device)
    tokenizer = BPETokenizer.load(args.tokenizer_path)

    handler = make_handler(model, tokenizer, device, args.format, args.checkpoint, args.tokenizer_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving TinyLLM chat UI at {url} (format={args.format})")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
