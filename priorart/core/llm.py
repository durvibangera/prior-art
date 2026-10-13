"""Client for the internal text-reasoning LLM, and the KV-lines protocol.

Confidentiality: this client sends invention text to `endpoints.llm`, which is
therefore required to be a self-hosted internal model (see README). It must
never be pointed at an external API.

WHY KV-LINES AND NOT JSON
-------------------------
The only available model is ~4B. JSON gives it five ways to destroy a whole
response (unbalanced brace, missing open quote, unescaped quote inside a value,
trailing comma, stray fence). The failure actually observed on this task was a
missing OPENING quote on a long free-text value:

    "reasoning": The abstract describes a hybrid system, whereas...

which is unrecoverable by construction, because a JSON parser cannot know where
the value ends. Worse, one such value destroys every other field in the object.

KV-lines has no quoting, no nesting, and no escaping, so malformation is always
LOCAL: a mangled NEWNESS line costs the novelty sentence, not the pillar
assignment. The parser below is a total function -- it never raises.

    COMPONENT: C1
    NAME: Acoustic feature front-end
    DOES: takes raw microphone audio and produces a time-frequency representation
    END
"""
from __future__ import annotations

import ipaddress
import json
import logging
import random
import re
import socket
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

# A field line: an ALL-CAPS name, then either a colon or plain whitespace,
# then the value. The colon is the instructed format, but this model has been
# observed to drop every colon in an otherwise perfectly correct response --
# e.g. "OURS Wavelet Scattering Transform" instead of "OURS: Wavelet...".
# Content was fine; only the separator was missing. `_STRICT` is tried first;
# `_LOOSE` is only trusted for a name the caller told us to expect (checked in
# parse_kv), so a real wrapped sentence that happens to start with a capital
# word can never be mistaken for a new field.
_FIELD_LINE_STRICT = re.compile(r"^([A-Z][A-Z0-9_]*):[ 	]?(.*)$")
_FIELD_LINE_LOOSE = re.compile(r"^([A-Z][A-Z0-9_]*)[ 	]+(.*)$")
_TERMINATOR = "END"


class LLMUnavailable(Exception):
    """The endpoint is disabled, unreachable, or returned something unusable."""


# --------------------------------------------------------------------- specs


@dataclass
class FieldSpec:
    """One expected field in a KV-lines response.

    `repeat` decides what a second occurrence of the same key means:
      overwrite -- last one wins (the default; a model that restates a field
                   is usually correcting itself)
      append    -- join with a space
      multi     -- keep every occurrence as a list (e.g. repeated PRIOR: lines)
    """
    name: str
    required: bool = True
    repeat: str = "overwrite"
    validate: Callable[[str], str | None] | None = None


@dataclass
class RecordSpec:
    """The shape of one expected response.

    `record_key` turns a flat response into a list of records: every time that
    key appears, a new record starts. Used for A1's repeated COMPONENT blocks.
    """
    fields: list[FieldSpec]
    record_key: str | None = None
    min_records: int = 1
    max_records: int = 99

    def by_name(self, name: str) -> FieldSpec | None:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def names(self) -> set[str]:
        return {f.name for f in self.fields}


@dataclass
class KVResult:
    """Parsed response plus everything needed to decide whether to retry.

    `ok` means every required field is present and every validator passed.
    `records` is always populated as far as parsing got -- a caller that wants
    partial data on failure can still read it.
    """
    records: list[dict]
    errors: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)
    raw: str = ""
    attempts: int = 0
    temperature: float = 0.0
    seed: int | None = None
    prompt_version: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def one(self) -> dict:
        """The single record, or {} -- for specs with no record_key."""
        return self.records[0] if self.records else {}

    def failed_fields(self) -> list[str]:
        """Field names mentioned in errors, for building a reduced retry."""
        out = []
        for e in self.errors:
            m = re.match(r"^([A-Z][A-Z0-9_]*)\b", e)
            if m and m.group(1) not in out:
                out.append(m.group(1))
        return out


# -------------------------------------------------------------------- parser


def parse_kv(raw: str, spec: RecordSpec) -> KVResult:
    """Parse a KV-lines response. Never raises.

    Rules:
      - a line matching NAME: value opens a field
      - any other non-blank line is appended to the previous field's value
        (so a wrapped long line survives)
      - END terminates; anything after it is discarded
      - anything before the first field line is discarded (preamble chatter)
      - unknown field names are dropped and reported, not fatal
      - EXCEPTION: for a single-field, non-record spec, if no field name is
        ever recognized at all, the whole response (up to END) is taken as
        that field's value. Observed live on gemma-e4: for a one-field spec
        it sometimes skips the field name entirely and just writes the bare
        answer ("5" instead of "SENTIMENT: 5", or the sentence itself instead
        of "QUESTION: ..."). This is a step beyond the dropped-colon case
        above -- here the whole wrapper is gone, not just the punctuation.
        Scoped narrowly to single-field specs so it can never misattribute
        text between two ambiguous slots; the caller's own validator (e.g.
        one_of_menu, max_words) is still the safety net against a reply that
        merely happens to have no colon in it, such as a refusal sentence.
    """
    records: list[dict] = []
    current: dict = {}
    unknown: list[str] = []
    last_key: str | None = None
    any_field_matched = False

    def flush():
        nonlocal current
        if current:
            records.append(current)
        current = {}

    body_lines: list[str] = []
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == _TERMINATOR or stripped == f"{_TERMINATOR}:":
            break
        body_lines.append(stripped)

        m = _FIELD_LINE_STRICT.match(stripped)
        if not m:
            loose = _FIELD_LINE_LOOSE.match(stripped)
            if loose and loose.group(1) in spec.names:
                m = loose
        if not m:
            # Continuation of the previous value. Ignore leading chatter that
            # arrives before any field has opened.
            if last_key is not None and last_key in current:
                if isinstance(current[last_key], list):
                    current[last_key][-1] = f"{current[last_key][-1]} {stripped}".strip()
                else:
                    current[last_key] = f"{current[last_key]} {stripped}".strip()
            continue

        key, value = m.group(1), m.group(2).strip()

        if spec.record_key and key == spec.record_key and current:
            flush()

        if key not in spec.names:
            unknown.append(key)
            last_key = None
            continue

        any_field_matched = True
        fs = spec.by_name(key)
        if fs.repeat == "multi":
            current.setdefault(key, []).append(value)
        elif fs.repeat == "append" and key in current:
            current[key] = f"{current[key]} {value}".strip()
        else:
            current[key] = value
        last_key = key

    flush()

    if (not records and not any_field_matched and not spec.record_key
            and len(spec.fields) == 1 and body_lines):
        whole = " ".join(body_lines).strip()
        if whole:
            records = [{spec.fields[0].name: whole}]

    errors: list[str] = []
    if len(records) < spec.min_records:
        errors.append(
            f"RECORDS expected at least {spec.min_records}, parsed {len(records)}")
    if len(records) > spec.max_records:
        records = records[:spec.max_records]
        log.warning("truncated to %d records", spec.max_records)

    for i, rec in enumerate(records):
        tag = f"[{i}]" if spec.record_key else ""
        for fs in spec.fields:
            if fs.name not in rec:
                if fs.required:
                    errors.append(f"{fs.name}{tag} missing")
                continue
            if fs.validate:
                values = rec[fs.name] if isinstance(rec[fs.name], list) else [rec[fs.name]]
                for v in values:
                    problem = fs.validate(v)
                    if problem:
                        errors.append(f"{fs.name}{tag} {problem}")

    if unknown:
        log.debug("dropped unknown fields: %s", sorted(set(unknown)))

    return KVResult(records=records, errors=errors,
                    unknown_fields=sorted(set(unknown)), raw=raw or "")


# ------------------------------------------------------------------ validators


def contains_all(*words: str) -> Callable[[str], str | None]:
    """Template-anchor check: `DOES: takes <x> and produces <y>`.

    Turns "did it answer the right question" into a regex, and stops the model
    writing a paragraph where a clause was asked for.
    """
    def check(value: str) -> str | None:
        low = value.lower()
        missing = [w for w in words if w.lower() not in low]
        return f"must contain {missing}" if missing else None
    return check


def _resolve_menu_key(menu: dict[str, str], value: str) -> str | None:
    """Shared by one_of_menu (pass/fail) and menu_key (which key) so they can
    never disagree with each other.

    Accepts the leading token, an embedded label, or (as a last resort) a
    bare digit as the LAST token of an otherwise-noncompliant response.
    That last case exists because a more capable, more talkative model
    (observed live: qwen3-max) sometimes ignores "copy one line only" and
    writes a full explanation instead, but often still concludes with the
    literal digit ("...so this is a 1."). Only the LAST token is checked,
    never a digit found anywhere in the middle of the text -- a number
    mentioned mid-explanation for an unrelated reason (e.g. "the 3rd
    approach") must not be mistaken for the answer, so this stays
    conservative rather than trying to parse free-form reasoning.
    """
    v = (value or "").strip()
    if not v:
        return None
    first = v.split()[0].rstrip(".):")
    if first in menu:
        return first
    low = v.lower()
    for key, label in menu.items():
        if label.lower() in low:
            return key
    last = v.rstrip(". ").split()[-1].rstrip(".):") if v.split() else ""
    if last in menu:
        return last
    return None


def one_of_menu(menu: dict[str, str]) -> Callable[[str], str | None]:
    """Copy-a-line field. Accepts the leading key, the label, or the whole line.

    Menu selection is used wherever a small model would otherwise have to
    generate a category, because picking from a list is reliable where
    free-form classification is not.
    """
    def check(value: str) -> str | None:
        v = (value or "").strip()
        if not v:
            return "is empty"
        if _resolve_menu_key(menu, v) is not None:
            return None
        return f"must copy one menu line, got {v[:40]!r}"
    return check


def menu_key(menu: dict[str, str], value: str) -> str | None:
    """Resolve a validated menu answer back to its key. None if unresolvable."""
    return _resolve_menu_key(menu, value)


def max_words(n: int) -> Callable[[str], str | None]:
    def check(value: str) -> str | None:
        count = len(value.split())
        return f"must be at most {n} words, got {count}" if count > n else None
    return check


def non_empty(value: str) -> str | None:
    return "is empty" if not value.strip() else None


# --------------------------------------------------------------------- client


class ConfidentialityError(Exception):
    """endpoints.llm points somewhere that is not a private/loopback host."""


def assert_private_endpoint(base_url: str) -> None:
    """Refuse to send invention text anywhere but a self-hosted model.

    The README's confidentiality rule used to be a comment; the examiner
    sends the WHOLE disclosure plus the inventor's candid answers, so a
    config typo pointing at a public API would itself create prior art.
    Only loopback and RFC1918/private addresses are accepted. A hostname is
    resolved once; one that cannot be resolved is refused rather than
    trusted.
    """
    host = urlparse(base_url).hostname
    if not host:
        raise ConfidentialityError(f"endpoints.llm.base_url has no host: {base_url!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError) as exc:
            raise ConfidentialityError(
                f"cannot resolve LLM host {host!r} to check it is internal") from exc
    if not (ip.is_private or ip.is_loopback):
        raise ConfidentialityError(
            f"endpoints.llm.base_url {base_url!r} resolves to public address {ip} -- "
            f"invention text may only go to a self-hosted internal model")


@dataclass
class LLMReply:
    text: str
    reasoning: str = ""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def split_reasoning(content: str) -> tuple[str, str]:
    """(answer, reasoning) from a Qwen3 completion.

    With vLLM's reasoning parser on, reasoning arrives separately and
    `content` is already clean. Without it, it arrives inline -- either as a
    full <think>...</think> block, or (Qwen3 chat templates pre-open the tag
    themselves) as bare reasoning followed by a lone </think>.
    """
    content = content or ""
    if "</think>" in content:
        head, _, tail = content.rpartition("</think>")
        return tail.strip(), head.replace("<think>", "").strip()
    return _THINK_BLOCK.sub("", content).strip(), ""


class LLMClient:
    def __init__(self, cfg: dict):
        llm_cfg = cfg.get("endpoints", {}).get("llm", {})
        self.enabled = bool(llm_cfg.get("enabled"))
        self.base_url = llm_cfg.get("base_url", "").rstrip("/")
        self.model = llm_cfg.get("model", "")
        self.timeout = llm_cfg.get("timeout_s", 120)
        # Thinking-mode turns on a 27B model routinely run past the
        # single-shot timeout above.
        self.converse_timeout = llm_cfg.get("converse_timeout_s", 600)
        self.headers = {"Content-Type": "application/json"}
        auth_value = (llm_cfg.get("auth_value") or "").strip()
        if auth_value and auth_value.lower() != "none":
            self.headers[llm_cfg.get("auth_header", "Authorization")] = auth_value
        if self.enabled:
            assert_private_endpoint(self.base_url)

    def ping(self) -> bool:
        """True if the endpoint answers /models -- cheap network-side check."""
        if not self.enabled:
            return False
        try:
            r = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    def converse(self, messages: list[dict], *, thinking: bool = True,
                 json_schema: dict | None = None, temperature: float | None = None,
                 max_tokens: int = 4000,
                 on_token: Callable[[str], None] | None = None) -> LLMReply:
        """One multi-turn completion over the full message list.

        `thinking` toggles Qwen3's reasoning mode via the chat template.
        `json_schema` turns on vLLM guided decoding, so the reply is valid
        JSON by construction -- new code uses this instead of KV-lines.
        `on_token` streams answer text (never reasoning) as it arrives.
        Raises LLMUnavailable.
        """
        if not self.enabled:
            raise LLMUnavailable("endpoints.llm.enabled is false")
        if temperature is None:
            temperature = 0.6 if thinking else 0.2
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95 if thinking else 1.0,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        try:
            if on_token is None:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers=self.headers, timeout=self.converse_timeout)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                text, inline = split_reasoning(msg.get("content") or "")
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or inline
                return LLMReply(text, reasoning)
            return self._stream(payload, thinking, on_token)
        except LLMUnavailable:
            raise
        except Exception as exc:                          # noqa: BLE001
            reason = str(exc).split(":")[0] if ":" in str(exc) else str(exc)
            raise LLMUnavailable(f"{type(exc).__name__}: {reason[:80]}") from exc

    def _stream(self, payload: dict, thinking: bool,
                on_token: Callable[[str], None]) -> LLMReply:
        payload = {**payload, "stream": True}
        content, reasoning = [], []
        # If the server has no reasoning parser, thinking arrives inline in
        # `content` and is only identifiable once </think> shows up -- so
        # hold content back until then, or until the stream ends.
        separate_reasoning = False
        released = not thinking
        with requests.post(f"{self.base_url}/chat/completions", json=payload,
                           headers=self.headers, timeout=self.converse_timeout,
                           stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta") or {}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                if r_piece:
                    separate_reasoning = True
                    reasoning.append(r_piece)
                piece = delta.get("content")
                if not piece:
                    continue
                content.append(piece)
                if released or separate_reasoning:
                    on_token(piece)
                    released = True
                elif "</think>" in "".join(content):
                    released = True
                    on_token("".join(content).rpartition("</think>")[2].lstrip())
        full = "".join(content)
        if not released:
            on_token(split_reasoning(full)[0])
        text, inline = split_reasoning(full)
        return LLMReply(text, "".join(reasoning) or inline)

    def converse_json(self, messages: list[dict], schema: dict, *,
                      thinking: bool = False, max_tokens: int = 4000) -> dict:
        """converse() with guided JSON, parsed. One retry, then the lenient
        extractor, then LLMUnavailable -- never a silent default."""
        last = ""
        for attempt in range(2):
            reply = self.converse(messages, thinking=thinking, json_schema=schema,
                                  max_tokens=max_tokens,
                                  temperature=None if attempt == 0 else 0.5)
            last = reply.text
            try:
                return json.loads(last)
            except json.JSONDecodeError:
                try:
                    out = extract_json(last)
                    if isinstance(out, dict):
                        return out
                except LLMUnavailable:
                    pass
        raise LLMUnavailable(f"no valid JSON after retry: {last[:200]!r}")

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 2000, seed: int | None = None) -> str:
        """Return the raw text of one completion. Raises LLMUnavailable."""
        if not self.enabled:
            raise LLMUnavailable("endpoints.llm.enabled is false")
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": user}],
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:                          # noqa: BLE001
            # Connection errors carry a very long chained message; keep enough
            # to say WHY, since this is surfaced per-call in the CLI and UI.
            reason = str(exc).split(":")[0] if ":" in str(exc) else str(exc)
            raise LLMUnavailable(f"{type(exc).__name__}: {reason[:80]}") from exc

    def chat_json(self, system: str, user: str, **kw) -> list | dict:
        """DEPRECATED -- kept only until claims.py / profile_builder.py /
        compare.py are removed. New code uses `ask()` with a RecordSpec."""
        return extract_json(self.chat(system, user, **kw))

    # The ladder deliberately varies temperature AND seed AND prompt shape.
    # Measured on this model: retrying an identical prompt at temperature 0
    # reproduces the identical malformed output, so a plain retry is worthless.
    # Attempt 3 drops the worked example, because some failures are CAUSED by
    # the example when the real input does not fit its shape.
    LADDER = ((0.0, False), (0.4, True), (0.8, True))

    def ask(self, system: str, user: str, spec: RecordSpec, *,
            max_tokens: int = 1200,
            reduce_prompt: Callable[[list[str]], tuple[str, str]] | None = None,
            prompt_version: str = "v1") -> KVResult:
        """Ask for a KV-lines record, validating and retrying up the ladder.

        Returns the best result obtained. On total failure returns a KVResult
        with `ok == False` and the errors recorded -- never a silent default,
        and never an exception the caller has to remember to catch for the
        ordinary "model wrote nonsense" case.
        """
        best: KVResult | None = None
        total_attempts = 0

        for attempt, (temp, vary_seed) in enumerate(self.LADDER, start=1):
            total_attempts = attempt
            seed = random.randint(1, 2**31 - 1) if vary_seed else 42
            sys_p, usr_p = system, user

            # Final attempt: reduced prompt asking only for what failed.
            if attempt == len(self.LADDER) and reduce_prompt and best:
                failed = best.failed_fields()
                if failed:
                    sys_p, usr_p = reduce_prompt(failed)

            try:
                raw = self.chat(sys_p, usr_p, temperature=temp,
                                max_tokens=max_tokens, seed=seed)
            except LLMUnavailable as exc:
                # Endpoint down is not a prompt problem -- do not burn the
                # ladder on it.
                result = KVResult(records=[], errors=[f"ENDPOINT {exc}"],
                                  attempts=attempt, temperature=temp, seed=seed,
                                  prompt_version=prompt_version)
                return best or result

            result = parse_kv(raw, spec)
            result.attempts = attempt
            result.temperature = temp
            result.seed = seed
            result.prompt_version = prompt_version

            if result.ok:
                result.attempts = total_attempts
                return result
            # Keep whichever attempt got furthest, measured by error count.
            if best is None or len(result.errors) < len(best.errors):
                best = result
            log.info("attempt %d failed validation: %s", attempt, result.errors[:3])

        # `attempts` on the returned result must say how many times the model
        # was actually asked, not just which single attempt's output survived
        # -- otherwise a genuinely-retried-3-times failure and a give-up-after-1
        # failure both silently report "attempts: 1", which is exactly the
        # ambiguity that made C1/C3's empty cards unreadable as a bug report.
        result = best or KVResult(records=[], errors=["no attempts made"])
        result.attempts = total_attempts
        return result


# ----------------------------------------------------- deprecated JSON path
# Retained only until the modules that still call it (claims.py,
# profile_builder.py, compare.py) are removed. New code must use `ask()`.


def extract_json(raw: str) -> list | dict:
    """DEPRECATED -- see the module docstring for why JSON is being retired."""
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    for text in candidates:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for open_c, close_c in (("[", "]"), ("{", "}")):
            start = text.find(open_c)
            if start < 0:
                continue
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
    raise LLMUnavailable(f"could not extract JSON from response: {raw[:200]!r}")
