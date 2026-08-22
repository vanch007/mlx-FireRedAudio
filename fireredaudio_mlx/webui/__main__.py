"""CLI launcher for FireRedAudio WebUI."""

import argparse
import uvicorn
from .app import create_app
from .config import DEFAULT_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="FireRedAudio Studio WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port number (default: 7860)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Path to FireRedAudio model directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    app = create_app(model_path=args.model)
    print(f"\n{'='*70}")
    print(f"🚀 FireRedAudio Studio WebUI running at: http://{args.host}:{args.port}")
    print(f"📚 Interactive API docs available at: http://{args.host}:{args.port}/docs")
    print(f"{'='*70}\n")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
