"""CLI entry point for the Qwen Building Pipeline WebUI server."""

import os
import sys


def main():
    import uvicorn

    # Add project root to sys.path so `webui.app` can be found
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    port = int(os.environ.get("QWEN_PORT", "8765"))
    uvicorn.run("webui.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
