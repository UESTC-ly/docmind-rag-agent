---
name: "playwright"
description: "Use DocMind's bounded Playwright adapter to navigate, inspect, interact with, and capture an allowed website."
---

# DocMind Playwright

Use only the `use_browser_tool` action exposed by DocMind. The host owns the
Chromium runtime, permitted host list, redirect/subresource filtering, timeout,
and per-execution browser session. Do not request CLI, `eval`, arbitrary
JavaScript, local files, browser extensions, or hosts outside the configured
allowlist.

Available actions:

- `open` or `navigate`: arguments `{ "url": "https://allowed.example/path" }`
- `click`: arguments `{ "selector": "accessible CSS selector" }`
- `fill`: arguments `{ "selector": "...", "text": "..." }`
- `text`: arguments `{ "selector": "body" }`
- `screenshot`: arguments `{ "full_page": true }`; the host writes the image
  under the isolated Skill workspace and returns it as an artifact
- `close`: close the current execution's browser session

Workflow:

1. Open an allowed URL.
2. Read targeted text before interacting.
3. Use stable selectors and re-read text after state changes.
4. Capture a screenshot only when it materially proves the result.
5. Close the session when the task is complete.

Report adapter failures honestly. Never claim a click, navigation, extracted
value, or screenshot unless the corresponding observation returned success.
