"""OpenAI-compatible wrapper for local ToriiGate captioning.

This is intentionally small: it exposes just enough of the OpenAI
`/v1/models` and `/v1/chat/completions` surface for Studio's LLM tagger.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "llm" / "ToriiGate-0.5"
DEFAULT_SYSTEM_PROMPT = (
    "You are image captioning expert. Describe the user's picture according "
    "to requested format and instructions."
)


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def _decode_image_url(value: str) -> Image.Image:
    if value.startswith("data:"):
        try:
            _, encoded = value.split(",", 1)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid data image URL") from exc
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    if value.startswith(("http://", "https://")):
        req = UrlRequest(value, headers={"User-Agent": "AnimaLoraStudio-ToriiGate/1.0"})
        with urlopen(req, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    if value.startswith("file://"):
        value = value[7:]

    path = Path(value)
    if path.exists():
        return Image.open(path).convert("RGB")

    raise HTTPException(status_code=400, detail="Unsupported or missing image URL")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_text_and_image(messages: list[dict[str, Any]]) -> tuple[list[str], list[str], Image.Image | None]:
    system_texts: list[str] = []
    user_texts: list[str] = []
    image: Image.Image | None = None

    for message in messages:
        role = _as_text(message.get("role") or "user")
        content = message.get("content")
        target = system_texts if role == "system" else user_texts

        if isinstance(content, str):
            if content.strip():
                target.append(content)
            continue

        if not isinstance(content, list):
            text = _as_text(content).strip()
            if text:
                target.append(text)
            continue

        for item in content:
            if not isinstance(item, dict):
                text = _as_text(item).strip()
                if text:
                    target.append(text)
                continue

            item_type = _as_text(item.get("type"))
            if item_type in {"text", "input_text"}:
                text = _as_text(item.get("text")).strip()
                if text:
                    target.append(text)
                continue

            if item_type in {"image_url", "input_image"} and image is None:
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = _as_text(image_url.get("url"))
                else:
                    url = _as_text(image_url)
                if url:
                    image = _decode_image_url(url)

    return system_texts, user_texts, image


class ToriiGateEngine:
    def __init__(
        self,
        model_dir: Path,
        *,
        served_model_name: str,
        device: str,
        dtype: str,
        attn_implementation: str,
        min_pixels: int,
        default_max_tokens: int,
    ) -> None:
        self.model_dir = model_dir
        self.served_model_name = served_model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.min_pixels = min_pixels
        self.default_max_tokens = default_max_tokens
        self.lock = threading.Lock()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.model: Qwen3_5ForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"ToriiGate model directory not found: {self.model_dir}")

        scripts_dir = self.model_dir / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
            try:
                from prompts import system_prompt  # type: ignore

                if isinstance(system_prompt, str) and system_prompt.strip():
                    self.system_prompt = system_prompt.strip()
            except Exception:
                pass

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch_dtype = _torch_dtype(self.dtype)
        if self.device == "cpu" and torch_dtype is torch.bfloat16:
            torch_dtype = torch.float32

        print(f"[toriigate] loading model from {self.model_dir}", flush=True)
        print(f"[toriigate] device={self.device} dtype={torch_dtype}", flush=True)

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            min_pixels=self.min_pixels,
            padding_side="right",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch_dtype,
            attn_implementation=self.attn_implementation,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        print("[toriigate] ready", flush=True)

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float,
        top_p: float | None,
    ) -> tuple[str, dict[str, int]]:
        if self.model is None or self.processor is None:
            raise RuntimeError("ToriiGate model is not loaded")

        system_texts, user_texts, image = _extract_text_and_image(messages)
        if image is None:
            return "ToriiGate local wrapper is ready.", {
                "prompt_tokens": 0,
                "completion_tokens": 7,
                "total_tokens": 7,
            }

        system = "\n\n".join([self.system_prompt, *system_texts]).strip()
        user_text = "\n\n".join(user_texts).strip()
        if not user_text:
            user_text = "Describe this image."

        chat_messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        token_budget = int(max_tokens or self.default_max_tokens)
        token_budget = max(1, min(token_budget, 4096))

        generation_kwargs: dict[str, Any] = {"max_new_tokens": token_budget}
        if temperature and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                generation_kwargs["top_p"] = float(top_p)
        else:
            generation_kwargs["do_sample"] = False

        with self.lock, torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        generated = output[:, prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        content = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return content, usage


def create_app(engine: ToriiGateEngine) -> FastAPI:
    app = FastAPI(title="ToriiGate OpenAI-compatible server")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": engine.served_model_name}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> JSONResponse:
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported")

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        try:
            content, usage = engine.generate(
                messages=messages,
                max_tokens=payload.get("max_tokens"),
                temperature=float(payload.get("temperature") or 0),
                top_p=payload.get("top_p"),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return JSONResponse(body)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("TORIIGATE_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Local ToriiGate model directory",
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("TORIIGATE_MODEL_NAME", "toriigate-0.5"),
        help="Model id exposed by /v1/models",
    )
    parser.add_argument("--host", default=os.environ.get("TORIIGATE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TORIIGATE_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("TORIIGATE_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.environ.get("TORIIGATE_DTYPE", "bfloat16"))
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("TORIIGATE_ATTN", "sdpa"),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=int(os.environ.get("TORIIGATE_MIN_PIXELS", str(256 * 32 * 32))),
    )
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=int(os.environ.get("TORIIGATE_MAX_TOKENS", "512")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = ToriiGateEngine(
        Path(args.model_dir).expanduser().resolve(),
        served_model_name=args.served_model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        default_max_tokens=args.default_max_tokens,
    )
    engine.load()
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
