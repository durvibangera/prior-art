"""Captcha solving for the InPASS portal.

The portal shows a 200x40 PNG containing six mixed-case alphanumeric
characters with light dot noise and mild per-character rotation.  It is
requested from /Captcha/GetCaptchaImage and is bound to the ASP.NET session
cookie, so it must be fetched with the same session that submits the search.

Solvers are pluggable.  `ManualSolver` needs no infrastructure and lets the
tool run today; `VisionSolver` calls an internal vision endpoint.

Captcha images carry no proprietary information, so any endpoint is
acceptable here -- unlike the relevance judge in core/scorer.py.
"""
from __future__ import annotations

import base64
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# InPASS captchas are exactly six mixed-case alphanumeric characters.
CAPTCHA_LEN = 6
CAPTCHA_RE = re.compile(rf"^[A-Za-z0-9]{{{CAPTCHA_LEN}}}$")

PROMPT = (
    "Read the characters in this captcha image. "
    "It contains exactly 6 characters which may be uppercase letters, "
    "lowercase letters or digits. Case matters. "
    "Ignore the background dots and grid lines. "
    "Reply with ONLY the 6 characters, nothing else."
)


class CaptchaSolver(ABC):
    """Solve a captcha PNG. Return the text, or None if it cannot be read."""

    @abstractmethod
    def solve(self, png: bytes) -> str | None: ...

    def report_failure(self, text: str) -> None:
        """Called when the portal rejected `text`, so solvers can adapt."""


class ManualSolver(CaptchaSolver):
    """Write the PNG to disk and prompt the operator to type it.

    Deliberately the default fallback: it makes the tool usable before any
    endpoint exists, and it is the escape hatch when the vision model's
    accuracy drops after a portal captcha redesign.
    """

    def __init__(self, workdir: Path):
        self.path = Path(workdir) / "captcha_current.png"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def solve(self, png: bytes) -> str | None:
        self.path.write_bytes(png)
        print(f"\n  Captcha saved to: {self.path}")
        print("  Open it, then type the 6 characters (or blank to refresh):")
        try:
            text = input("  captcha> ").strip()
        except EOFError:
            return None
        return text or None


class VisionSolver(CaptchaSolver):
    """Call an OpenAI-compatible vision endpoint (vLLM, Ollama, LiteLLM, TGI)."""

    def __init__(self, cfg: dict):
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = cfg.get("timeout_s", 60)
        self.headers = {"Content-Type": "application/json"}
        auth_value = (cfg.get("auth_value") or "").strip()
        if auth_value and auth_value.lower() != "none":
            self.headers[cfg.get("auth_header", "Authorization")] = auth_value

    def solve(self, png: bytes) -> str | None:
        b64 = base64.b64encode(png).decode()
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        }
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              json=payload, headers=self.headers,
                              timeout=self.timeout)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        except Exception as exc:                       # noqa: BLE001
            log.warning("vision endpoint failed: %s", exc)
            return None
        return self._clean(raw)

    @staticmethod
    def _captcha_like(text: str) -> bool:
        """Reject prose that happens to be the right shape.

        Without this, a refusal such as "sorry i cannot" yields the token
        "cannot", which would be submitted to the portal as an answer and
        burn a retry.  Real InPASS captchas are exactly six characters and
        effectively always mix character classes, so requiring a digit or
        mixed case filters plain English words out.
        """
        if not CAPTCHA_RE.match(text):
            return False
        has_digit = any(c.isdigit() for c in text)
        mixed_case = any(c.islower() for c in text) and any(c.isupper() for c in text)
        return has_digit or mixed_case

    @classmethod
    def _clean(cls, raw: str) -> str | None:
        """Extract the answer from whatever wrapping the model added."""
        candidates: list[str] = []

        # 1. The well-behaved case: the model returned just the answer.
        candidates.append(raw.strip().strip("`'\".,:;").replace(" ", ""))

        # 2. Prose wrapping, e.g. "The captcha reads: nG7yhQ".  Tokenise on
        #    real word boundaries so neighbouring words are not fused.
        tokens = re.findall(r"[A-Za-z0-9]+", raw)
        candidates.extend(reversed(tokens))

        # 3. Characters spelled out one by one, e.g. "n G 7 y h Q".
        singles = [t for t in tokens if len(t) == 1]
        if len(singles) >= CAPTCHA_LEN:
            candidates.append("".join(singles[-CAPTCHA_LEN:]))

        for cand in candidates:
            if cls._captcha_like(cand):
                return cand
        return None


class ChainSolver(CaptchaSolver):
    """Try each solver in order; fall through on failure.

    Used for `mode: auto` -- vision first, operator as backstop -- so an
    endpoint outage degrades the run to slow rather than broken.
    """

    def __init__(self, *solvers: CaptchaSolver):
        self.solvers = [s for s in solvers if s is not None]

    def solve(self, png: bytes) -> str | None:
        for solver in self.solvers:
            text = solver.solve(png)
            if text:
                return text
        return None


def build_solver(config: dict, workdir: Path) -> CaptchaSolver:
    mode = config.get("captcha", {}).get("mode", "auto")
    vision_cfg = config.get("endpoints", {}).get("vision", {})
    vision = VisionSolver(vision_cfg) if vision_cfg.get("enabled") else None

    if mode == "manual" or vision is None:
        if mode == "vision":
            raise SystemExit(
                "captcha.mode=vision but endpoints.vision.enabled is false. "
                "Fill in priorart/config.yaml or set captcha.mode=manual."
            )
        return ManualSolver(workdir)
    if mode == "vision":
        return vision
    return ChainSolver(vision, ManualSolver(workdir))
