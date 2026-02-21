#!/usr/bin/env python3
"""
HTTP server with Cross-Origin headers for SharedArrayBuffer support.

SharedArrayBuffer (required by Emscripten pthreads / WASM threads) needs:
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp

Usage:
  python serve.py [port]       (default: 8080)
  python serve.py              (serves from current directory on port 8080)
"""

import http.server
import sys

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Serving on http://localhost:{port} with COOP/COEP headers")
    print("(SharedArrayBuffer enabled for WASM threading)")
    server = http.server.HTTPServer(("", port), COOPCOEPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
