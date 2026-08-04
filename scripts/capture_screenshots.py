#!/usr/bin/env python
"""Regenerate the README screenshots (docs/screenshot.png, docs/screenshot-result.png).

These are the images referenced at the top of README.md. Run this whenever the UI
changes so the screenshots stay current.

Prerequisites (dev-only — NOT needed to run or deploy the app):
    pip install playwright
    python -m playwright install chromium

Usage:
    # 1. Start the app in another terminal:
    python -m streamlit run app.py
    # 2. Then capture both screenshots:
    python scripts/capture_screenshots.py
    # (optional) point at a different URL:
    python scripts/capture_screenshots.py http://localhost:8502

The captures use a 2x device scale for crisp images on high-DPI displays.
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
APP_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
RESULT_QUERY = "Difference between theft and robbery"


def _new_page(pw, theme="dark", width=1280, height=1000):
    browser = pw.chromium.launch()
    page = browser.new_page(
        viewport={"width": width, "height": height}, device_scale_factor=2
    )
    # The app's theme is client-side, driven by localStorage['tl_theme'] which the
    # JS bridge reads on load. Seed it before any page script runs.
    page.add_init_script(f"try{{localStorage.setItem('tl_theme','{theme}')}}catch(e){{}}")
    return browser, page


def _wait_ready(page, timeout=300):
    """Load the app and wait until it's interactive, clicking the Streamlit
    Community Cloud "get this app back up" sleep screen if/when it appears."""
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
    ready = "button:has-text('Analyze Legal Position')"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if page.locator(ready).first.is_visible():
                break
        except Exception:
            pass
        try:
            wake = page.get_by_role(
                "button", name=re.compile("get this app back up", re.I)
            )
            if wake.count() > 0:
                wake.first.click()
        except Exception:
            pass
        time.sleep(3)
    else:
        raise TimeoutError("app never became interactive")
    time.sleep(2)  # let fonts/animations settle


def capture_home(pw, theme="dark", filename="screenshot.png"):
    """Landing page (header, toggle, query input, example queries) in one theme."""
    browser, page = _new_page(pw, theme=theme)
    try:
        _wait_ready(page)
        out = DOCS / filename
        page.screenshot(path=str(out))
        print("saved", out.relative_to(REPO_ROOT))
    finally:
        browser.close()


def capture_result(pw, theme="dark", filename="screenshot-result.png"):
    """A full query result, top to bottom: header, Law Change banner, analysis,
    and the statutory-provision / web-source columns."""
    browser, page = _new_page(pw, theme=theme)
    try:
        _wait_ready(page)
        page.locator("input[type='text'], textarea").first.fill(RESULT_QUERY)
        page.get_by_role("button", name="Analyze Legal Position").click()
        page.wait_for_selector(".statute-card", timeout=120_000)
        time.sleep(3)
        # Streamlit scrolls an inner container, so `full_page` only sees the
        # viewport. Grow the viewport very tall so nothing is clipped, measure the
        # true bottom of the content (the footer), then size the viewport exactly
        # to it for one clean top-to-bottom shot.
        page.set_viewport_size({"width": 1280, "height": 5000})
        time.sleep(2)
        height = page.evaluate(
            """() => {
                const f = document.querySelector('.footer-text');
                const r = (f || document.body).getBoundingClientRect();
                return Math.ceil(r.bottom + window.scrollY);
            }"""
        )
        page.set_viewport_size({"width": 1280, "height": min(int(height) + 24, 9000)})
        time.sleep(1)
        page.evaluate("() => window.scrollTo(0, 0)")
        time.sleep(1)
        out = DOCS / filename
        page.screenshot(path=str(out))
        print("saved", out.relative_to(REPO_ROOT))
    finally:
        browser.close()


def main():
    DOCS.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        # Home screen in both themes (shown side by side in the README).
        capture_home(pw, theme="dark", filename="screenshot.png")
        capture_home(pw, theme="light", filename="screenshot-light.png")
        # Query result (dark) demonstrating the Law Change banner + provisions.
        capture_result(pw, theme="dark", filename="screenshot-result.png")


if __name__ == "__main__":
    main()
