"""Backend launcher.

On Windows, uvicorn defaults to the ProactorEventLoop, which psycopg's async
mode (the durable Postgres checkpointer) cannot use. This launcher runs the
server on a SelectorEventLoop instead.

Usage:  python run.py [--port 8000]
"""
import argparse
import asyncio
import selectors
import sys

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    config = uvicorn.Config("main:app", host=args.host, port=args.port)
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        runner = asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        )
        runner.run(server.serve())
    else:
        asyncio.run(server.serve())
