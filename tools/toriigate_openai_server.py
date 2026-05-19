"""OpenAI-compatible wrapper for local ToriiGate captioning.

This is intentionally small: it exposes just enough of the OpenAI
`/v1/models` and `/v1/chat/completions` surface for Studio's LLM tagger.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
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
ANIMA_JSON_SYSTEM_PROMPT = """You are an anime image captioning assistant for LoRA training.

Return exactly one valid JSON object and no prose outside JSON.
The field "character" is only for a known character name, never for a description.
If a character, series, or artist name is not visually explicit, use an empty string.
Put descriptive details into appearance, tags, environment, and nl."""

ANIMA_JSON_USER_PROMPT = """Look at the image and output this JSON object only:
{
  "quality": "",
  "count": "",
  "character": "",
  "series": "",
  "artist": "",
  "appearance": [],
  "tags": [],
  "environment": [],
  "nl": ""
}

Rules:
- Use concise English Danbooru-style tags with spaces, not underscores.
- Fill count with visible count tags such as "1girl, solo", "1boy, solo", or "multiple girls".
- Put hair, eyes, body features, accessories, and visible clothing in appearance.
- Put pose, expression, framing, composition, medium, and art style in tags.
- Put background, location, lighting, weather, time of day, and palette in environment.
- Fill appearance, tags, environment, and nl with useful visible details.
- Never put a sentence or caption in character, series, or artist.
- Leave quality empty.
- Do not include watermark, signature, username, date, score, source, resolution, or quality tags.
- Keep nl to one short factual English sentence.
- Do not include explanations, Markdown, comments, or extra keys."""


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


def _looks_like_anima_json_request(system_texts: list[str], user_texts: list[str]) -> bool:
    text = "\n".join([*system_texts, *user_texts]).lower()
    return (
        "json" in text
        and "appearance" in text
        and ("environment" in text or "background" in text)
        and (
            '"nl"' in text
            or "natural-language" in text
            or "natural language" in text
            or "lora" in text
            or "danbooru" in text
        )
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _dedupe(items: list[str], *, limit: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        tag = re.sub(r"\s+", " ", item.replace("_", " ").strip().lower())
        tag = tag.strip(" .,;:()[]{}\"'")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= limit:
            break
    return out


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.I) is not None


def _tags_from_description(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    appearance: list[str] = []
    tags: list[str] = []
    environment: list[str] = []

    for color in (
        "black",
        "white",
        "blonde",
        "blond",
        "brown",
        "red",
        "pink",
        "blue",
        "green",
        "purple",
        "silver",
        "grey",
        "gray",
        "yellow",
        "golden",
    ):
        if re.search(rf"\b{color}[- ]+hair\b", lowered):
            appearance.append("blonde hair" if color == "blond" else f"{color} hair")
        if re.search(rf"\b{color}[- ]+eyes\b", lowered):
            appearance.append("gold eyes" if color == "golden" else f"{color} eyes")

    for phrase in (
        "long hair",
        "short hair",
        "medium hair",
        "twintails",
        "ponytail",
        "braid",
        "braids",
        "bangs",
        "ahoge",
        "animal ears",
        "fox ears",
        "cat ears",
        "tail",
        "horns",
        "wings",
        "hat",
        "cap",
        "hair ornament",
        "glasses",
        "earrings",
        "ribbon",
        "bow",
        "jacket",
        "coat",
        "dress",
        "skirt",
        "shirt",
        "blouse",
        "hoodie",
        "kimono",
        "uniform",
        "sailor uniform",
        "necktie",
        "gloves",
        "boots",
        "stockings",
        "thigh-high stockings",
        "thighhighs",
    ):
        if _contains_phrase(lowered, phrase):
            appearance.append(phrase.replace("thigh-high stockings", "thighhighs"))

    tag_phrases = {
        "looking at viewer": ("looking at the viewer", "staring directly", "gazing at the viewer"),
        "smile": ("smile", "smiling"),
        "serious expression": ("serious expression", "serious look"),
        "intense expression": ("intense expression", "intense"),
        "angry": ("angry", "annoyed"),
        "crying": ("crying", "tears"),
        "blush": ("blush", "flushed"),
        "open mouth": ("open mouth", "mouth open"),
        "closed eyes": ("closed eyes", "eyes closed"),
        "hands near face": ("hands near", "hands are positioned in front of her face"),
        "peace sign": ("peace sign",),
        "sitting": ("sitting", "seated"),
        "standing": ("standing",),
        "lying": ("lying", "laying"),
        "upper body": ("upper body", "bust shot"),
        "close-up": ("close-up", "close up"),
        "portrait": ("portrait",),
        "dynamic pose": ("dynamic pose", "dynamic"),
        "anime style": ("anime style", "anime artwork", "anime"),
        "illustration": ("illustration", "artwork"),
        "chibi": ("chibi",),
    }
    for tag, phrases in tag_phrases.items():
        if any(phrase in lowered for phrase in phrases):
            tags.append(tag)

    env_phrases = {
        "simple background": ("simple background", "plain background"),
        "dark background": ("dark background",),
        "indoors": ("indoors", "indoor"),
        "outdoors": ("outdoors", "outdoor"),
        "sky": ("sky",),
        "cloudy sky": ("cloudy sky", "clouds"),
        "night": ("night", "nighttime"),
        "sunset": ("sunset",),
        "city": ("city", "urban"),
        "forest": ("forest",),
        "room": ("room", "bedroom", "living room"),
        "dramatic lighting": ("dramatic lighting",),
        "soft lighting": ("soft lighting", "soft light"),
        "warm colors": ("warm color", "warm palette"),
        "cool colors": ("cool color", "cool palette"),
    }
    for tag, phrases in env_phrases.items():
        if any(phrase in lowered for phrase in phrases):
            environment.append(tag)

    return {
        "appearance": _dedupe(appearance),
        "tags": _dedupe(tags),
        "environment": _dedupe(environment),
    }


def _infer_count(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("multiple girls", "several girls", "three girls", "two girls")):
        return "multiple girls"
    if any(phrase in lowered for phrase in ("multiple boys", "several boys", "three boys", "two boys")):
        return "multiple boys"
    if any(phrase in lowered for phrase in ("multiple characters", "several characters", "two characters")):
        return "multiple"
    if re.search(r"\b(girl|woman|female)\b", lowered):
        return "1girl, solo"
    if re.search(r"\b(boy|man|male)\b", lowered):
        return "1boy, solo"
    if "solo" in lowered:
        return "solo"
    return ""


def _shorten_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    match = re.match(r"(.{1,220}?[.!?])(?:\s|$)", text)
    if match:
        return match.group(1).strip()
    return text[:220].rstrip(" ,;:") + ("." if len(text) > 220 else "")


def _repair_anima_json_text(text: str) -> str:
    parsed = _extract_json_object(text)
    if parsed is None:
        parsed = {"nl": text}
    if isinstance(parsed.get("ai_output"), dict):
        parsed = {**parsed, **parsed["ai_output"]}

    expected = {
        "quality",
        "count",
        "character",
        "series",
        "artist",
        "appearance",
        "tags",
        "environment",
        "nl",
    }
    native_text_keys = [
        key
        for key, value in parsed.items()
        if isinstance(value, str)
        and (
            key.lower().startswith("character")
            or key.lower()
            in {"general", "background", "image_effects", "image effects", "atmosphere", "texts"}
        )
    ]
    if not expected.intersection(parsed) and not native_text_keys:
        parsed = {"nl": text}

    def as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return _dedupe([str(v) for v in value if str(v).strip()])
        if isinstance(value, str) and value.strip():
            return _dedupe([v for v in re.split(r"[,;\n]+", value) if v.strip()])
        return []

    native_text = "\n".join(str(parsed[key]) for key in native_text_keys).strip()
    all_text = "\n".join(
        str(value)
        for value in [native_text, parsed.get("nl"), text]
        if isinstance(value, str) and value.strip()
    )
    inferred = _tags_from_description(all_text)

    out: dict[str, Any] = {
        "quality": str(parsed.get("quality") or ""),
        "count": str(parsed.get("count") or ""),
        "character": str(parsed.get("character") or ""),
        "series": str(parsed.get("series") or ""),
        "artist": str(parsed.get("artist") or ""),
        "appearance": as_list(parsed.get("appearance")),
        "tags": as_list(parsed.get("tags")),
        "environment": as_list(parsed.get("environment")),
        "nl": str(parsed.get("nl") or ""),
    }
    if not out["nl"]:
        for key in ("general", "General", "character_1", "Character_1", "background", "Background"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                out["nl"] = value
                break

    character = out["character"]
    if isinstance(character, str) and (
        len(character) > 80
        or any(mark in character for mark in (".", " with ", " wearing ", " features "))
    ):
        if not out["nl"]:
            out["nl"] = character
        out["character"] = ""

    if not out["count"]:
        out["count"] = _infer_count(all_text)
    for key in ("appearance", "tags", "environment"):
        if not out[key]:
            out[key] = inferred[key]
    if out["tags"] and "anime style" not in out["tags"]:
        out["tags"] = _dedupe([*out["tags"], "anime style"])
    out["nl"] = _shorten_sentence(str(out["nl"]))

    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


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

        wants_anima_json = _looks_like_anima_json_request(system_texts, user_texts)
        if wants_anima_json:
            system = "\n\n".join([ANIMA_JSON_SYSTEM_PROMPT, *system_texts]).strip()
            user_texts = [ANIMA_JSON_USER_PROMPT, *user_texts]
        else:
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
        if wants_anima_json:
            content = _repair_anima_json_text(content)
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
