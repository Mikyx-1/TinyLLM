"""
webchat.py — local browser chat UI for interactively testing a TinyLLM checkpoint.

The CLI demo (chat_demo.py) only proves the pipeline runs -- it doesn't let you poke at
it yourself, which is the actual point of testing "does turn 2's answer track what I
just typed, or is it a memorized script for this opener." This serves the same
generation logic (chat_demo.generate_reply, so there is exactly one place that builds
qa/chatml prompts) over a tiny stdlib-only HTTP server plus a static chat page, so you
can type arbitrary follow-ups in a browser and watch the answer stream in token by
token, the same way a real LLM actually generates -- not a canned string that appears
all at once.

/chat streams newline-delimited JSON as generation proceeds ({"delta": "..."} per
token, then one final {"done": true, "answer", "reasoning", "full"} once EOS/</CALC>
handling settles the reply) rather than blocking until the whole reply is ready --
model.generate.generate's on_token hook fires once per token actually appended to the
sequence, and the HTTP/1.0 response (no Content-Length, connection closes when the
handler returns) lets the browser's fetch() ReadableStream read each write as it
lands.

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
    python -m demo.webchat --checkpoint checkpoints/multitask_chatml/final.pt \\
        --tokenizer_path checkpoints/tokenizer.json --format chatml
    Then open http://127.0.0.1:8765 in a browser.

    A model this small generates a full reply in well under a second on CPU or GPU --
    much faster than real LLM APIs, whose per-token pace is normally set by network +
    far larger model compute. Pass --stream_delay_ms 60 (say) to add an artificial
    per-token pause so a recording of this UI reads at a human pace instead of
    flashing the whole reply in one frame.
"""

import argparse
import json
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from demo.chat_demo import generate_reply, load_model, render_prompt_prefix
from tokenizer import BPETokenizer

STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webchat_ui.html")


def make_handler(
    model, tokenizer, device, format: str, checkpoint_path: str, tokenizer_path: str,
    stream_delay_ms: float = 0,
):
    n_params = sum(p.numel() for p in model.parameters())

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _start_stream(self):
            # No Content-Length: this is HTTP/1.0 (the class default), so an absent
            # Content-Length means "body runs until the connection closes" -- exactly
            # what a growing, not-yet-fully-known-length stream needs. self.wfile is
            # unbuffered (StreamRequestHandler's wbufsize=0 default), so each write
            # below reaches the socket immediately, no manual flush() required.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _send_line(self, payload: dict):
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))

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
            except Exception as e:  # noqa: BLE001 -- bad request, headers not sent yet
                self._send_json(500, {"error": str(e)})
                return

            # From here on the 200 + streaming headers are already on the wire, so a
            # failure mid-generation can't change the status code -- it's reported as a
            # final {"error": ...} line instead (see webchat_ui.html's stream reader).
            self._start_stream()
            try:
                decoded_so_far = ""
                gen_ids: list[int] = []

                def on_token(token_id: int) -> None:
                    nonlocal decoded_so_far
                    gen_ids.append(token_id)
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    delta = text[len(decoded_so_far):]
                    decoded_so_far = text
                    if delta:
                        self._send_line({"delta": delta})
                        if stream_delay_ms:
                            time.sleep(stream_delay_ms / 1000)

                result = generate_reply(
                    model, tokenizer, device, prompt_prefix, message, format,
                    max_new_tokens, temperature, top_k, on_token=on_token,
                )
                # "full" (reasoning included) is what the browser should store back into
                # its history for the *next* request -- render_prompt_prefix rebuilds
                # context from history's "answer" field, and if the client instead sent
                # back the display-only "answer" (reasoning stripped), the model would
                # lose its own reasoning trace from the transcript it conditions on.
                self._send_line({
                    "done": True,
                    "answer": result["answer"],
                    "reasoning": result["reasoning"],
                    "full": result["full"],
                })
            except Exception as e:  # noqa: BLE001 -- surface any failure to the browser instead of a dropped connection
                self._send_line({"error": str(e)})

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
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (default), 'cpu', or 'cuda' -- this model is tiny enough to "
             "generate fast on either device, so this mostly matters for matching "
             "whatever device you trained/evaluated the checkpoint on",
    )
    parser.add_argument(
        "--stream_delay_ms", type=float, default=0,
        help="artificial pause after each streamed token, in ms (e.g. 60) -- purely "
             "cosmetic pacing for demos/recordings; 0 (default) streams as fast as the "
             "model actually generates",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, device)
    tokenizer = BPETokenizer.load(args.tokenizer_path)

    handler = make_handler(
        model, tokenizer, device, args.format, args.checkpoint, args.tokenizer_path,
        stream_delay_ms=args.stream_delay_ms,
    )
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
