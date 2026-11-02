from __future__ import annotations

import argparse

from qaraqu_face.config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qaraqu_face")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health", help="краткий статус")
    sp = sub.add_parser("serve", help="запуск API")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)

    args = p.parse_args(argv)
    cfg = load_config()

    if args.cmd == "health":
        print({"detect": cfg["detect"]["weights"], "recognize": cfg["recognize"]["backend"]})
        return 0
    if args.cmd == "serve":
        import uvicorn

        host = args.host or cfg.get("api", {}).get("host", "0.0.0.0")
        port = args.port or int(cfg.get("api", {}).get("port", 5020))
        uvicorn.run("qaraqu_face.api:app", host=host, port=port, reload=False)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
