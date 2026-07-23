#!/usr/bin/env python3
"""
Standalone bulk import into a Pinecone BYOC (Bring Your Own Cloud) index.

Extracted / distilled from pinecone_toolkit.py so it can run on its own.

What it does
------------
1. Connects to Pinecone with your API key.
2. Connects to a BYOC serverless index by its host.
3. Kicks off a server-side bulk import from object storage
   (S3 / GCS / Azure Blob, depending on which cloud your BYOC runs in).
4. Polls the import until it Completes / Fails / is Cancelled.

Key BYOC detail
---------------
Unlike Standard (SaaS) Pinecone imports, a BYOC import does NOT need a
`storage_integration_id`. The Pinecone data plane runs inside *your* cloud
account and already has IAM access to the bucket, so you only pass the `uri`.

Source layout in the bucket must be:
    <uri-prefix>/<namespace>/<file>.parquet
with each parquet file matching the Pinecone import schema
(id, values, [sparse_values], [metadata]).

Usage
-----
    export PINECONE_API_KEY=pcsk_...
    python byoc_bulk_import.py \
        --host https://my-index.svc.byoc-aws.byoc.pinecone.io \
        --uri s3://my-bucket/import-data \
        --error-mode continue
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import time

from pinecone import Pinecone, ImportErrorMode


PRIVATE_HOST_PROBE_TIMEOUT_SEC = 2


def maybe_privatize_host(host: str) -> str:
    """Rewrite a BYOC public host to its private-endpoint equivalent when reachable.

    Pinecone returns BYOC hosts like:
        <idx>.svc.<env>.byoc.pinecone.io
    Inside the customer VPC the data plane is reached via the VPC endpoint whose
    TLS cert covers:
        *.svc.private.<env>.byoc.pinecone.io
    We only rewrite when the private host actually TCP-connects from this machine,
    since the private endpoint resolves to an RFC1918 address only routable inside
    the VPC.
    """
    m = re.match(r"(https?://)?([^.]+)\.(svc\.)([^.]+\.byoc\.pinecone\.io)(.*)", host)
    if not m:
        return host
    scheme, idx, svc, tail, rest = m.groups()
    private_host = f"{idx}.{svc}private.{tail}"
    try:
        sock = socket.create_connection(
            (private_host, 443), timeout=PRIVATE_HOST_PROBE_TIMEOUT_SEC
        )
        sock.close()
    except (socket.gaierror, socket.timeout, OSError):
        return host
    rewritten = f"{scheme or ''}{private_host}{rest}"
    print(f"  (Using private endpoint: {private_host})")
    return rewritten


def connect(pc: Pinecone, host: str):
    if not host.startswith("https://"):
        host = f"https://{host}"
    host = maybe_privatize_host(host)
    print(f"  Connecting to index host: {host}")
    index = pc.Index(host=host)
    stats = index.describe_index_stats()
    dim = getattr(stats, "dimension", "?")
    total = getattr(stats, "total_vector_count", 0)
    print(f"  Connected (dimension={dim}, vectors={total:,})")
    return index


def start_import(index, uri: str, error_mode_str: str):
    error_mode = (
        ImportErrorMode.ABORT
        if error_mode_str.lower().startswith("a")
        else ImportErrorMode.CONTINUE
    )
    print(f"\n  Starting BYOC import")
    print(f"    URI:        {uri}")
    print(f"    Error mode: {'ABORT' if error_mode == ImportErrorMode.ABORT else 'CONTINUE'}")

    # NOTE: no integration_id for BYOC — the data plane already has bucket access.
    resp = index.start_import(uri=uri, error_mode=error_mode)
    print(f"  Import started. ID: {resp.id}")
    return resp.id


def monitor(index, import_id: str, poll_interval: int = 10):
    print(f"\n  Monitoring import '{import_id}' (every {poll_interval}s). Ctrl+C to stop.")
    print("  " + "-" * 56)
    start = time.time()
    try:
        while True:
            status = index.describe_import(id=import_id)
            elapsed = int(time.time() - start)
            state = status.status
            pct = getattr(status, "percent_complete", None)
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            records = getattr(status, "records_imported", None) or 0
            print(f"  [{elapsed:5d}s]  status: {state:<12}  progress: {pct_str:<8}  records: {records:,}")

            if state == "Completed":
                print("  " + "-" * 56)
                print(f"  Import completed in {elapsed}s. Records: {records:,}")
                print("  Note: vectors can take ~10 min to become queryable.")
                return status
            if state in ("Failed", "Cancelled"):
                print("  " + "-" * 56)
                print(f"  Import {state.lower()}.")
                if getattr(status, "error", None):
                    print(f"    Error: {status.error}")
                return status

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(f"\n  Stopped watching. Import '{import_id}' is still running server-side.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Bulk import into a Pinecone BYOC index.")
    parser.add_argument("--host", required=True, help="BYOC index host (e.g. https://idx.svc.byoc-aws.byoc.pinecone.io)")
    parser.add_argument("--uri", required=True, help="Storage prefix: s3://bucket/prefix, gs://bucket/prefix, or Azure Blob URL")
    parser.add_argument("--error-mode", default="continue", choices=["continue", "abort"], help="How to handle record errors")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between status polls")
    parser.add_argument("--no-monitor", action="store_true", help="Start the import and exit without polling")
    parser.add_argument("--api-key", default=os.environ.get("PINECONE_API_KEY"), help="Pinecone API key (defaults to PINECONE_API_KEY)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set PINECONE_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    pc = Pinecone(api_key=args.api_key)
    index = connect(pc, args.host)
    import_id = start_import(index, args.uri, args.error_mode)

    if not args.no_monitor:
        monitor(index, import_id, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
