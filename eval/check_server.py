"""Health check and API conformance test for participant servers."""

import argparse
import json
import sys
import urllib.request
import urllib.error


def check_health(base_url: str) -> bool:
    """Check GET /health returns 200."""
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("[PASS] GET /health returned 200")
                return True
            print(f"[FAIL] GET /health returned {resp.status}")
            return False
    except urllib.error.URLError as e:
        print(f"[FAIL] GET /health failed: {e}")
        return False


def check_chat_completions(base_url: str) -> bool:
    """Check POST /v1/chat/completions returns a valid response."""
    payload = json.dumps({
        "model": "Qwen/Qwen3.5-35B-A3B",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        print(f"[FAIL] POST /v1/chat/completions failed: {e}")
        return False

    errors = []

    # Check top-level fields
    for field in ("id", "choices", "usage"):
        if field not in body:
            errors.append(f"missing top-level field '{field}'")

    # Check choices structure
    choices = body.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        errors.append("'choices' must be a non-empty list")
    else:
        choice = choices[0]
        if "message" not in choice:
            errors.append("choices[0] missing 'message'")
        else:
            msg = choice["message"]
            if "role" not in msg or "content" not in msg:
                errors.append("choices[0].message missing 'role' or 'content'")
            elif not isinstance(msg["content"], str) or len(msg["content"]) == 0:
                errors.append("choices[0].message.content is empty or not a string")
        if "finish_reason" not in choice:
            errors.append("choices[0] missing 'finish_reason'")

    # Check usage structure
    usage = body.get("usage", {})
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if field not in usage:
            errors.append(f"usage missing '{field}'")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"\nResponse body:\n{json.dumps(body, indent=2)}")
        return False

    content_preview = choices[0]["message"]["content"][:80]
    print(f"[PASS] POST /v1/chat/completions returned valid response: \"{content_preview}...\"")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check server health and API conformance")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results = [
        check_health(base_url),
        check_chat_completions(base_url),
    ]

    if all(results):
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nSome checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
