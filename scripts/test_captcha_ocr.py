#!/usr/bin/env python3
"""Manual end-to-end test of the captcha OCR endpoint against a LIVE InPASS captcha.

Run this on a machine with BOTH internet (to reach InPASS) and intranet access
(to reach the OCR endpoint) -- your workstation, not the sandboxed agent.

    python3 scripts/test_captcha_ocr.py

It fetches a real captcha image, saves it next to this script, sends it to the
configured vision/OCR endpoint, and prints the answer. Open the saved PNG and
compare by eye -- this is the only way to know the endpoint is actually
reading captchas correctly, as opposed to just being reachable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.inpass.client import InpassClient
from priorart.inpass.captcha import VisionSolver, ManualSolver


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "priorart" / "config.yaml").read_text())
    vision_cfg = cfg["endpoints"]["vision"]

    if not vision_cfg.get("enabled"):
        print("endpoints.vision.enabled is false in config.yaml - nothing to test.")
        return

    print(f"Endpoint : {vision_cfg['base_url']}")
    print(f"Model    : {vision_cfg['model']}")
    print()

    # Fetch a real captcha. connect=False + manual solver here because we only
    # want the image, not to actually run a search.
    client = InpassClient(cfg, ManualSolver(root / "out"), )
    png = client._fetch_captcha()

    out_path = Path(__file__).parent / "test_captcha.png"
    out_path.write_bytes(png)
    print(f"Saved captcha image to: {out_path}")
    print("Open it now and read it yourself before looking at the model's answer.\n")

    solver = VisionSolver(vision_cfg)
    answer = solver.solve(png)

    print(f"Model's answer: {answer!r}")
    if answer is None:
        print("\nFAILED - endpoint returned nothing usable. Check the raw response")
        print("by adding a print(raw) in captcha.py VisionSolver.solve() if needed.")
    else:
        print("\nCompare the answer above against the saved PNG.")
        print("If they match: this endpoint is ready for unattended use.")
        print("If they don't: keep captcha.mode: manual in config.yaml for now.")


if __name__ == "__main__":
    main()
