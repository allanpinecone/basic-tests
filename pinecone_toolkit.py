#!/usr/bin/env python3
"""
Pinecone Toolkit — one script to rule them all.

Combines index management, bulk import, load testing, and backup
operations into a single interactive CLI.

API key:
  Set the PINECONE_API_KEY environment variable to skip the prompt:
      export PINECONE_API_KEY=pcsk_...

Usage:
  python pinecone_toolkit.py
"""

from __future__ import annotations

import getpass
import os
import random
import re
import socket
import ssl
import statistics
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime

from pinecone import Pinecone, ImportErrorMode

try:
    from pinecone import ServerlessSpec

    HAS_SERVERLESS = True
except ImportError:
    HAS_SERVERLESS = False

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import boto3

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs_storage

    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from azure.storage.blob import BlobServiceClient

    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

ENV_BYOC_AWS = "byoc-aws"
ENV_BYOC_GCP = "byoc-gcp"
ENV_BYOC_AZURE = "byoc-azure"
ENV_SAAS = "saas"

ENV_LABELS = {
    ENV_BYOC_AWS: "BYOC — AWS",
    ENV_BYOC_GCP: "BYOC — GCP",
    ENV_BYOC_AZURE: "BYOC — Azure",
    ENV_SAAS: "Standard Pinecone (SaaS)",
}

VECTOR_DIMENSION = 1024
BATCH_SIZE = 100
DEFAULT_WRITE_THREADS = 10
DEFAULT_READ_THREADS = 20
DEFAULT_THREADS_PER_NAMESPACE = 4

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║              PINECONE TOOLKIT                            ║
║              One script to rule them all.                ║
╚══════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def _masked_input(prompt: str = "", mask: str = "*") -> str:
    """Read a line of input, printing a mask character for each keystroke."""
    try:
        import tty, termios

        sys.stdout.write(prompt)
        sys.stdout.flush()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        chars: list[str] = []
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    break
                elif ch in ("\x7f", "\x08"):
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                elif ch == "\x03":
                    raise KeyboardInterrupt
                elif ch == "\x04":
                    raise EOFError
                else:
                    chars.append(ch)
                    sys.stdout.write(mask)
                sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return "".join(chars)
    except (ImportError, OSError):
        return getpass.getpass(prompt)


def prompt_int(message: str, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{message}{suffix}: ").strip()
        if not value and default is not None:
            return default
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid number.")


def prompt_yes_no(message: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        value = input(f"{message} [{hint}]: ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False


def section_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Index listing & connection
# ═══════════════════════════════════════════════════════════════════════════

def list_indexes_detailed(pc: Pinecone) -> list:
    """List indexes with full details. Returns the raw index list."""
    print("\nFetching indexes...")
    try:
        indexes = list(pc.list_indexes())
    except Exception as e:
        print(f"  Could not list indexes: {e}")
        return []

    if not indexes:
        print("  No indexes found in this project.")
        return indexes

    print(f"\nAvailable indexes ({len(indexes)}):")
    print("-" * 70)
    for i, idx in enumerate(indexes, 1):
        name = getattr(idx, "name", str(idx))
        host = getattr(idx, "host", "N/A")
        dim = getattr(idx, "dimension", "?")
        metric = getattr(idx, "metric", "?")
        status_obj = getattr(idx, "status", None)
        state = getattr(status_obj, "state", "?") if status_obj else "?"
        print(f"  {i}. {name}  (dim={dim}, metric={metric}, state={state})")
        print(f"     host: {host}")
    print("-" * 70)
    return indexes


def pick_index(pc: Pinecone) -> tuple[str | None, str | None]:
    """List indexes and let the user choose one. Returns (name, host)."""
    indexes = list_indexes_detailed(pc)
    if not indexes:
        return None, None

    selection = input(f"Pick an index [1-{len(indexes)}]: ").strip()
    if not selection:
        return None, None
    try:
        sel_idx = int(selection) - 1
        if 0 <= sel_idx < len(indexes):
            chosen = indexes[sel_idx]
            name = getattr(chosen, "name", str(chosen))
            host = getattr(chosen, "host", None)
            if host and not host.startswith("https://"):
                host = f"https://{host}"
            return name, host
    except ValueError:
        pass
    print("  Invalid selection.")
    return None, None


CONNECT_TIMEOUT_SEC = 5


def test_connection(index, timeout: float = CONNECT_TIMEOUT_SEC) -> bool:
    """Quick connectivity check via describe_index_stats, with a hard timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(index.describe_index_stats)
        try:
            stats = future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(
                f"  Connection timed out after {timeout:.0f}s — "
                f"describe_index_stats() did not respond."
            )
            print(
                "  This usually means the index host is unreachable from this "
                "machine (DNS, firewall, VPN, or private-endpoint routing)."
            )
            future.cancel()
            return False
        except Exception as e:
            print(f"  Connection failed: {e}")
            return False

        dim = getattr(stats, "dimension", "?")
        total = getattr(stats, "total_vector_count", 0)
        print(f"  Connected! (dimension={dim}, vectors={total:,})")
        return True
    finally:
        executor.shutdown(wait=False)


def describe_index_stats(index):
    """Display detailed stats for the connected index."""
    try:
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        dim = getattr(stats, "dimension", None)
        fullness = getattr(stats, "index_fullness", None)
        namespaces = getattr(stats, "namespaces", {}) or {}

        print(f"\n  Index Stats:")
        if dim is not None:
            print(f"    Dimension:      {dim}")
        if total is not None:
            print(f"    Total vectors:  {total:,}")
        if fullness is not None:
            print(f"    Index fullness: {fullness}")

        if namespaces:
            print(f"\n    Namespaces ({len(namespaces)}):")
            for ns in sorted(namespaces.keys()):
                ns_info = namespaces[ns]
                count = (
                    ns_info.get("vector_count", 0)
                    if isinstance(ns_info, dict)
                    else getattr(ns_info, "vector_count", 0)
                )
                label = ns if ns != "" else "(default)"
                print(f"      - {label}: {count:,} vectors")
        else:
            print(f"\n    No namespaces found.")
    except Exception as e:
        print(f"  Error: {e}")


PRIVATE_HOST_PROBE_TIMEOUT_SEC = 2


def _maybe_privatize_host(host: str) -> str:
    """Rewrite a BYOC public host to its private-endpoint equivalent.

    The Pinecone API returns hosts like:
        <idx>.svc.<env>.byoc.pinecone.io
    Inside the customer VPC the data-plane is reached via the VPC endpoint
    whose TLS cert covers:
        *.svc.private.<env>.byoc.pinecone.io
    We rewrite to the private host only when it is *actually* reachable from
    this machine — verified with a real TCP connect, not just DNS, since
    BYOC publishes the private endpoint as a public DNS A record pointing
    at an RFC1918 address that is only routable from inside the VPC.
    """
    m = re.match(
        r"(https?://)?([^.]+)\.(svc\.)([^.]+\.byoc\.pinecone\.io)(.*)",
        host,
    )
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


def connect_to_index(pc, name=None, host=None):
    """Connect to an index by host or name, test it, return (index, name, host)."""
    if host:
        if not host.startswith("https://"):
            host = f"https://{host}"
        host = _maybe_privatize_host(host)
        print(f"\n  Connecting to '{name or host}'...")
        index = pc.Index(host=host)
    elif name:
        print(f"\n  Connecting to '{name}'...")
        index = pc.Index(name=name)
    else:
        return None, None, None

    if test_connection(index):
        return index, name, host
    else:
        proceed = input("  Connection test failed. Continue anyway? [y/N]: ").strip().lower()
        if proceed.startswith("y"):
            return index, name, host
        return None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Index Management
# ═══════════════════════════════════════════════════════════════════════════

def test_connectivity(hostname: str, pinecone_host: str) -> bool:
    """Run DNS, TCP, and HTTPS connectivity tests against a host."""
    section_header("Connectivity Tests")

    print(f"\n  1. DNS Resolution for {hostname}...")
    try:
        ips = socket.gethostbyname_ex(hostname)
        print(f"     Resolved to: {ips[2]}")
    except socket.gaierror as e:
        print(f"     FAILED: {e}")
        return False

    print(f"\n  2. TCP connection to {hostname}:443...")
    try:
        sock = socket.create_connection((hostname, 443), timeout=10)
        sock.close()
        print("     SUCCESS: Port 443 is reachable")
    except (socket.timeout, socket.error) as e:
        print(f"     FAILED: {e}")
        return False

    print(f"\n  3. HTTPS connection to {pinecone_host}...")
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(pinecone_host, method="HEAD")
        urllib.request.urlopen(req, timeout=10, context=ctx)
        print("     SUCCESS: HTTPS connection works")
    except urllib.error.HTTPError as e:
        print(f"     SUCCESS: Got HTTP {e.code} (connection works, auth expected)")
    except Exception as e:
        print(f"     FAILED: {e}")
        return False

    print("\n  All connectivity tests passed.")
    return True


def action_test_connectivity_and_upsert(state: dict):
    """Test connectivity to the connected index and optionally upsert a test vector."""
    index = state["index"]
    host = state["index_host"] or ""

    if host.startswith("https://"):
        hostname = host.removeprefix("https://")
    elif host.startswith("http://"):
        hostname = host.removeprefix("http://")
    else:
        hostname = host
    hostname = hostname.rstrip("/")

    if hostname:
        if not test_connectivity(hostname, host):
            print("\n  Connectivity tests failed.")
            return
    else:
        print("  No host URL available for connectivity tests.")
        print("  Checking connection via API...")
        if not test_connection(index):
            return

    if prompt_yes_no("\n  Upsert a single test vector (1024 dims)?", default=True):
        test_vector = [0.1] * 1024
        print("\n  Upserting test vector 'toolkit-test-1'...")
        try:
            start = time.time()
            result = index.upsert(
                vectors=[
                    {
                        "id": "toolkit-test-1",
                        "values": test_vector,
                        "metadata": {"description": "Test vector from pinecone_toolkit"},
                    }
                ]
            )
            elapsed = time.time() - start
            print(f"  SUCCESS ({elapsed:.2f}s): {result}")

            print("  Waiting 2s for propagation...")
            time.sleep(2)
            describe_index_stats(index)
        except Exception as e:
            print(f"  FAILED: {e}")


def action_create_index(state: dict):
    """Create a new serverless Pinecone index."""
    if not HAS_SERVERLESS:
        print("  ServerlessSpec not available in this pinecone version.")
        return

    pc = state["pc"]

    name = input("  Index name: ").strip()
    if not name:
        print("  Name is required.")
        return

    dimension = prompt_int("  Vector dimension", 1024)

    print("  Metric options: cosine, euclidean, dotproduct")
    metric = input("  Metric [cosine]: ").strip().lower() or "cosine"
    if metric not in ("cosine", "euclidean", "dotproduct"):
        print(f"  Invalid metric '{metric}'.")
        return

    print("\n  Cloud options: aws, gcp, azure")
    cloud = input("  Cloud [aws]: ").strip().lower() or "aws"

    default_region = {"aws": "us-east-1", "gcp": "us-central1", "azure": "eastus"}.get(cloud, "us-east-1")
    region = input(f"  Region [{default_region}]: ").strip() or default_region

    print(f"\n  Creating index:")
    print(f"    Name:      {name}")
    print(f"    Dimension: {dimension}")
    print(f"    Metric:    {metric}")
    print(f"    Cloud:     {cloud}")
    print(f"    Region:    {region}")

    if not prompt_yes_no("\n  Proceed?", default=True):
        print("  Cancelled.")
        return

    try:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"\n  Index '{name}' created successfully!")
        print("  It may take a moment to become ready.")

        if prompt_yes_no("  Connect to it now?", default=True):
            idx_name, idx_host = name, None
            try:
                indexes = list(pc.list_indexes())
                for idx in indexes:
                    if getattr(idx, "name", None) == name:
                        idx_host = getattr(idx, "host", None)
                        break
            except Exception:
                pass
            index, _, host = connect_to_index(pc, name=idx_name, host=idx_host)
            if index:
                state["index"] = index
                state["index_name"] = idx_name
                state["index_host"] = host or ""
    except Exception as e:
        print(f"\n  Failed to create index: {e}")


def action_delete_index(state: dict):
    """List indexes and delete a selected one."""
    pc = state["pc"]
    indexes = list_indexes_detailed(pc)
    if not indexes:
        return

    choice = input("\n  Enter the number of the index to delete (or 'q' to cancel): ").strip()
    if choice.lower() == "q" or not choice:
        return

    try:
        selected = indexes[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return

    name = getattr(selected, "name", str(selected))
    confirm = input(f"\n  Are you sure you want to delete '{name}'? This cannot be undone. Type 'yes': ").strip()
    if confirm.lower() != "yes":
        print("  Aborted.")
        return

    print(f"  Deleting index '{name}'...")
    try:
        pc.delete_index(name)
        print("  Done — index deleted.")
        if state.get("index_name") == name:
            state["index"] = None
            state["index_name"] = None
            state["index_host"] = None
            print("  (You were connected to this index — now disconnected.)")
    except Exception as e:
        print(f"  Failed: {e}")


def menu_index_management(state: dict):
    """Sub-menu for index management."""
    while True:
        section_header("INDEX MANAGEMENT")
        print("  1. List all indexes")
        print("  2. Describe index stats")
        print("  3. Test connectivity & upsert test vector")
        print("  4. Create a new index")
        print("  5. Delete an index")
        print("  b. Back to main menu")

        choice = input("\n  Select option: ").strip().lower()

        if choice == "b" or not choice:
            break
        elif choice == "1":
            list_indexes_detailed(state["pc"])
        elif choice == "2":
            if state["index"]:
                describe_index_stats(state["index"])
            else:
                print("  No index connected. Use 'Switch Index' from the main menu.")
        elif choice == "3":
            if state["index"]:
                action_test_connectivity_and_upsert(state)
            else:
                print("  No index connected. Use 'Switch Index' from the main menu.")
        elif choice == "4":
            action_create_index(state)
        elif choice == "5":
            action_delete_index(state)
        else:
            print("  Invalid option.")


# ═══════════════════════════════════════════════════════════════════════════
# Bulk Import — URI parsing & storage validation
# ═══════════════════════════════════════════════════════════════════════════

def _parse_s3_uri(uri: str):
    uri = uri.rstrip("/")
    arn_match = re.match(
        r"^s3://(arn:aws:s3:[^:]*:[^:]*:accesspoint/[^/]+)(?:/(.*))?$", uri
    )
    if arn_match:
        return arn_match.group(1), arn_match.group(2) or ""
    std_match = re.match(r"^s3://([^/]+)(?:/(.*))?$", uri)
    if std_match:
        return std_match.group(1), std_match.group(2) or ""
    return None, None


def _parse_gcs_uri(uri: str):
    uri = uri.rstrip("/")
    match = re.match(r"^gs://([^/]+)(?:/(.*))?$", uri)
    if match:
        return match.group(1), match.group(2) or ""
    return None, None


def _parse_azure_uri(uri: str):
    uri = uri.rstrip("/")
    az_match = re.match(r"^azure://([^/]+)(?:/(.*))?$", uri)
    if az_match:
        return az_match.group(1), az_match.group(2) or ""
    blob_match = re.match(
        r"^https://[^/]+\.blob\.core\.windows\.net/([^/]+)(?:/(.*))?$", uri
    )
    if blob_match:
        return blob_match.group(1), blob_match.group(2) or ""
    return None, None


def _uri_scheme_for_env(env: str) -> str:
    return {
        ENV_BYOC_AWS: "s3://",
        ENV_BYOC_GCP: "gs://",
        ENV_BYOC_AZURE: "https://<account>.blob.core.windows.net/... or gs://",
        ENV_SAAS: "s3://, gs://, or Azure Blob URL",
    }[env]


def _uri_example_for_env(env: str) -> str:
    return {
        ENV_BYOC_AWS: "s3://my-bucket/import-data",
        ENV_BYOC_GCP: "gs://my-bucket/import-data",
        ENV_BYOC_AZURE: "https://myaccount.blob.core.windows.net/mycontainer/import-data",
        ENV_SAAS: "s3://my-bucket/import-data",
    }[env]


def _validate_uri_scheme(uri: str, env: str) -> bool:
    is_s3 = uri.startswith("s3://")
    is_gcs = uri.startswith("gs://")
    is_azure_blob = ".blob.core.windows.net/" in uri

    if env == ENV_BYOC_AWS and not is_s3:
        print(f"  [WARN] AWS indexes only support s3:// URIs, got: {uri}")
        return False
    if env == ENV_BYOC_GCP and not (is_gcs or is_azure_blob):
        print(f"  [WARN] GCP indexes support gs:// or Azure Blob URIs, got: {uri}")
        return False
    if env == ENV_BYOC_AZURE and not (is_gcs or is_azure_blob):
        print(f"  [WARN] Azure indexes support gs:// or Azure Blob URIs, got: {uri}")
        return False
    if env == ENV_SAAS and not (is_s3 or is_gcs or is_azure_blob):
        print(f"  [WARN] Expected s3://, gs://, or Azure Blob URI, got: {uri}")
        return False
    return True


# --- S3 validation ---

def _get_s3_client():
    import botocore.exceptions

    try:
        s3 = boto3.client("s3")
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return s3
    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.ClientError, botocore.exceptions.PartialCredentialsError):
        pass

    print("    No AWS credentials found. Enter them manually:")
    aws_key = input("    AWS Access Key ID: ").strip()
    aws_secret = getpass.getpass("    AWS Secret Access Key: ")
    aws_region = input("    AWS Region [us-east-1]: ").strip() or "us-east-1"
    if not aws_key or not aws_secret:
        return None
    return boto3.client("s3", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, region_name=aws_region)


def validate_s3(s3_uri: str) -> bool:
    if not HAS_BOTO3:
        print("\n  [skip] boto3 not installed — skipping S3 pre-flight check.")
        print("         Install with: pip install boto3")
        return True

    bucket, prefix = _parse_s3_uri(s3_uri)
    if not bucket:
        print(f"\n  [FAIL] Could not parse S3 URI: {s3_uri}")
        return False

    print(f"\n  Validating S3 access...")
    print(f"    Bucket/ARN: {bucket}")
    print(f"    Prefix:     {prefix or '(root)'}")

    s3 = _get_s3_client()
    if not s3:
        print(f"\n  [FAIL] No valid AWS credentials provided.")
        return False

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"    Bucket access: OK")
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "403":
            print(f"    [FAIL] Access denied to bucket '{bucket}'.")
        elif error_code == "404":
            print(f"    [FAIL] Bucket '{bucket}' does not exist.")
        else:
            print(f"    [FAIL] Cannot access bucket: {e}")
        return False

    return _validate_parquet_structure_s3(s3, bucket, prefix, s3_uri)


def _validate_parquet_structure_s3(s3, bucket, prefix, display_uri):
    list_prefix = f"{prefix}/" if prefix else ""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        all_objects = []
        for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
            all_objects.extend(page.get("Contents", []))
    except Exception as e:
        print(f"    [FAIL] Cannot list objects: {e}")
        return False

    if not all_objects:
        print(f"    [FAIL] No objects found under '{display_uri}'.")
        return False

    return _analyze_parquet_layout(all_objects, list_prefix, "S3")


# --- GCS validation ---

def _get_gcs_client():
    from google.auth.exceptions import DefaultCredentialsError

    try:
        client = gcs_storage.Client()
        list(client.list_buckets(max_results=1))
        return client
    except (DefaultCredentialsError, Exception):
        pass

    print("    No GCP credentials found via Application Default Credentials.")
    print("      1. Run 'gcloud auth application-default login' in another terminal")
    print("      2. Set GOOGLE_APPLICATION_CREDENTIALS env var")
    print("      3. Enter a service account key file path now")
    key_path = input("    Service account key file path (or Enter to skip): ").strip()
    if not key_path:
        return None
    try:
        return gcs_storage.Client.from_service_account_json(key_path)
    except Exception as e:
        print(f"    Failed to authenticate with key file: {e}")
        return None


def validate_gcs(gcs_uri: str) -> bool:
    if not HAS_GCS:
        print("\n  [skip] google-cloud-storage not installed — skipping GCS pre-flight check.")
        print("         Install with: pip install google-cloud-storage")
        return True

    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    if not bucket_name:
        print(f"\n  [FAIL] Could not parse GCS URI: {gcs_uri}")
        return False

    print(f"\n  Validating GCS access...")
    print(f"    Bucket: {bucket_name}")
    print(f"    Prefix: {prefix or '(root)'}")

    client = _get_gcs_client()
    if not client:
        print(f"\n  [FAIL] No valid GCP credentials provided.")
        return False

    try:
        client.get_bucket(bucket_name)
        print(f"    Bucket access: OK")
    except Exception as e:
        error_str = str(e)
        if "403" in error_str:
            print(f"    [FAIL] Access denied to bucket '{bucket_name}'.")
        elif "404" in error_str:
            print(f"    [FAIL] Bucket '{bucket_name}' does not exist.")
        else:
            print(f"    [FAIL] Cannot access bucket: {e}")
        return False

    list_prefix = f"{prefix}/" if prefix else ""
    try:
        all_objects = []
        for blob in client.list_blobs(bucket_name, prefix=list_prefix):
            all_objects.append({"Key": blob.name, "Size": blob.size or 0})
    except Exception as e:
        print(f"    [FAIL] Cannot list objects: {e}")
        return False

    if not all_objects:
        print(f"    [FAIL] No objects found under '{gcs_uri}'.")
        return False

    return _analyze_parquet_layout(all_objects, list_prefix, "GCS")


# --- Azure validation ---

def _get_azure_client(container_name: str):
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        try:
            service = BlobServiceClient.from_connection_string(conn_str)
            service.get_container_client(container_name).get_container_properties()
            return service
        except Exception:
            pass

    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
    account_key = os.environ.get("AZURE_STORAGE_KEY")
    if account_name and account_key:
        try:
            service = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=account_key,
            )
            service.get_container_client(container_name).get_container_properties()
            return service
        except Exception:
            pass

    print("    No Azure credentials found in environment.")
    print("      1. Set AZURE_STORAGE_CONNECTION_STRING env var")
    print("      2. Set AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY env vars")
    print("      3. Enter storage account name and key now")

    acct = input("    Storage account name (or Enter to skip): ").strip()
    if not acct:
        return None
    key = getpass.getpass("    Storage account key: ")
    if not key:
        return None
    try:
        service = BlobServiceClient(
            account_url=f"https://{acct}.blob.core.windows.net",
            credential=key,
        )
        service.get_container_client(container_name).get_container_properties()
        return service
    except Exception as e:
        print(f"    Failed to authenticate: {e}")
        return None


def validate_azure(azure_uri: str) -> bool:
    if not HAS_AZURE:
        print("\n  [skip] azure-storage-blob not installed — skipping Azure pre-flight check.")
        print("         Install with: pip install azure-storage-blob")
        return True

    container, prefix = _parse_azure_uri(azure_uri)
    if not container:
        print(f"\n  [FAIL] Could not parse Azure URI: {azure_uri}")
        return False

    print(f"\n  Validating Azure Blob Storage access...")
    print(f"    Container: {container}")
    print(f"    Prefix:    {prefix or '(root)'}")

    service = _get_azure_client(container)
    if not service:
        print(f"\n  [FAIL] No valid Azure credentials provided.")
        return False

    container_client = service.get_container_client(container)
    try:
        container_client.get_container_properties()
        print(f"    Container access: OK")
    except Exception as e:
        print(f"    [FAIL] Cannot access container: {e}")
        return False

    list_prefix = f"{prefix}/" if prefix else ""
    try:
        all_objects = []
        for blob in container_client.list_blobs(name_starts_with=list_prefix or None):
            all_objects.append({"Key": blob.name, "Size": blob.size or 0})
    except Exception as e:
        print(f"    [FAIL] Cannot list blobs: {e}")
        return False

    if not all_objects:
        print(f"    [FAIL] No objects found under '{azure_uri}'.")
        return False

    return _analyze_parquet_layout(all_objects, list_prefix, "Azure Blob")


# --- Shared parquet layout analysis ---

def _analyze_parquet_layout(all_objects: list, list_prefix: str, provider: str) -> bool:
    namespaces: dict[str, list] = {}
    non_parquet: list[str] = []
    bad_structure: list[str] = []

    for obj in all_objects:
        key = obj["Key"]
        rel = key[len(list_prefix):]
        if not rel or rel.endswith("/"):
            continue

        parts = rel.split("/")
        if len(parts) == 2 and parts[1].endswith(".parquet"):
            namespaces.setdefault(parts[0], []).append({
                "file": parts[1],
                "size_mb": obj["Size"] / (1024 * 1024),
            })
        elif len(parts) == 1 and parts[0].endswith(".parquet"):
            bad_structure.append(rel)
        elif not rel.endswith(".parquet"):
            non_parquet.append(rel)
        else:
            bad_structure.append(rel)

    ok = True

    if bad_structure:
        print(f"\n    [WARN] Parquet files without a namespace subdirectory:")
        for f in bad_structure[:5]:
            print(f"           - {f}")
        if len(bad_structure) > 5:
            print(f"           ... and {len(bad_structure) - 5} more")
        print(f"           These will cause 'No namespace detected' errors.")
        ok = False

    if non_parquet:
        print(f"\n    [WARN] {len(non_parquet)} non-parquet file(s) found (will be ignored by Pinecone).")

    if not namespaces:
        print(f"\n    [FAIL] No valid <namespace>/<file>.parquet structure found.")
        return False

    total_files = sum(len(files) for files in namespaces.values())
    total_size = sum(f["size_mb"] for files in namespaces.values() for f in files)
    print(f"\n    Found {len(namespaces)} namespace(s), {total_files} parquet file(s), {total_size:.1f} MB total:")
    for ns, files in sorted(namespaces.items()):
        ns_size = sum(f["size_mb"] for f in files)
        print(f"      {ns}/  ({len(files)} file(s), {ns_size:.1f} MB)")
        for f in files[:3]:
            print(f"        - {f['file']}  ({f['size_mb']:.1f} MB)")
        if len(files) > 3:
            print(f"        ... and {len(files) - 3} more")

    label = f"{provider} validation"
    if ok:
        print(f"\n    {label}: PASSED")
    else:
        print(f"\n    {label}: WARNINGS (see above)")
    return True


def validate_uri(uri: str, env: str) -> bool:
    if uri.startswith("s3://"):
        return validate_s3(uri)
    if uri.startswith("gs://"):
        return validate_gcs(uri)
    if ".blob.core.windows.net/" in uri:
        return validate_azure(uri)
    print(f"  [skip] Unrecognized URI scheme — skipping validation.")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Bulk Import — import operations
# ═══════════════════════════════════════════════════════════════════════════

def _extract_all_fields(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
            if not callable(val):
                result[attr] = val
        except Exception:
            pass
    return result


def _print_import_details(status):
    print(f"\n  Import details:")
    print(f"    ID:               {status.id}")
    print(f"    URI:              {getattr(status, 'uri', 'N/A')}")
    print(f"    Status:           {status.status}")
    pct = getattr(status, "percent_complete", None)
    if pct is not None:
        print(f"    Percent complete: {pct:.1f}%")
    records = getattr(status, "records_imported", None)
    if records is not None:
        print(f"    Records imported: {records:,}")
    created = getattr(status, "created_at", None)
    if created:
        print(f"    Created at:       {created}")
    finished = getattr(status, "finished_at", None)
    if finished:
        print(f"    Finished at:      {finished}")


def monitor_import(index, import_id: str, poll_interval: int = 10, verbose: bool = False):
    print(f"\n  Monitoring import '{import_id}' (polling every {poll_interval}s)...")
    if verbose:
        print("  Verbose mode: dumping full API response each poll.")
    print("  Press Ctrl+C to stop monitoring and return to menu.")
    print("  " + "-" * 56)

    start_time = time.time()
    last_raw = None

    try:
        while True:
            try:
                status = index.describe_import(id=import_id)
                elapsed = int(time.time() - start_time)
                pct = getattr(status, "percent_complete", None)
                state_str = status.status

                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                records = getattr(status, "records_imported", None) or 0

                if records:
                    print(f"  [{elapsed:5d}s]  status: {state_str:<12}  progress: {pct_str:<8}  records: {records:,}")
                else:
                    print(f"  [{elapsed:5d}s]  status: {state_str:<12}  progress: {pct_str:<8}")

                if verbose:
                    raw = _extract_all_fields(status)
                    if raw != last_raw:
                        for k, v in raw.items():
                            print(f"             {k}: {v}")
                        last_raw = raw

                if state_str == "Completed":
                    print("  " + "-" * 56)
                    print(f"\n  Import completed successfully in {elapsed}s!")
                    _print_import_details(status)
                    return status

                if state_str in ("Failed", "Cancelled"):
                    print("  " + "-" * 56)
                    print(f"\n  Import {state_str.lower()}.")
                    _print_import_details(status)
                    if hasattr(status, "error") and status.error:
                        print(f"    Error: {status.error}")
                    return status

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    Warning: error checking status: {e}")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n  Monitoring stopped. Import '{import_id}' is still running.")
        print(f"  Use 'Check import status' to check later.")
        return None


def action_start_import(state: dict):
    index = state["index"]
    env = state["env"]
    storage_integration_id = state.get("storage_integration_id")

    if env == ENV_SAAS:
        uri = input("  Storage URI (e.g. s3://my-bucket/import-data): ").strip()
    else:
        scheme = _uri_scheme_for_env(env)
        example = _uri_example_for_env(env)
        uri = input(f"  Storage URI ({scheme}) e.g. {example}: ").strip()

    if not uri:
        print("  Error: URI is required.")
        return

    if env != ENV_SAAS:
        if not _validate_uri_scheme(uri, env):
            if not prompt_yes_no("  URI scheme doesn't match environment. Continue anyway?", default=False):
                return

        if prompt_yes_no("  Run storage path validation before importing?", default=False):
            if not validate_uri(uri, env):
                if not prompt_yes_no("\n  Validation failed. Start import anyway?", default=False):
                    print("  Import cancelled.")
                    return

    error_mode_input = input("  Error mode — [c]ontinue or [a]bort? [c]: ").strip().lower() or "c"
    error_mode = ImportErrorMode.ABORT if error_mode_input.startswith("a") else ImportErrorMode.CONTINUE

    print(f"\n  Starting import...")
    print(f"    Environment:    {ENV_LABELS[env]}")
    print(f"    URI:            {uri}")
    if storage_integration_id:
        print(f"    Integration ID: {storage_integration_id}")
    print(f"    Error mode:     {'ABORT' if error_mode == ImportErrorMode.ABORT else 'CONTINUE'}")

    import_kwargs = dict(uri=uri, error_mode=error_mode)
    if storage_integration_id:
        import_kwargs["integration_id"] = storage_integration_id

    try:
        resp = index.start_import(**import_kwargs)
        import_id = resp.id
        print(f"\n  Import started!  ID: {import_id}")

        monitor = input("\n  Monitor progress? [Y/n/v(erbose)]: ").strip().lower() or "y"
        if monitor.startswith("v"):
            monitor_import(index, import_id, verbose=True)
        elif monitor.startswith("y"):
            monitor_import(index, import_id)

    except Exception as e:
        print(f"\n  Failed to start import: {e}")


def action_list_imports(state: dict):
    index = state["index"]
    print("\n  Fetching imports...")
    try:
        imports = list(index.list_imports())
        if not imports:
            print("  No imports found.")
            return
        imports.reverse()
        print(f"\n  Found {len(imports)} import(s) (newest first):")
        print("  " + "-" * 56)
        for imp in imports:
            _print_import_details(imp)
            if hasattr(imp, "error") and imp.error:
                print(f"    Error: {imp.error}")
            print("  " + "-" * 56)
    except Exception as e:
        print(f"  Error listing imports: {e}")


def action_cancel_import(state: dict):
    import_id = input("  Enter import ID to cancel: ").strip()
    if not import_id:
        return
    try:
        state["index"].cancel_import(id=import_id)
        print(f"  Cancel request sent for import '{import_id}'.")
    except Exception as e:
        print(f"  Error cancelling import: {e}")


def action_delete_namespace(state: dict):
    index = state["index"]
    try:
        stats = index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", {}) or {}
        if isinstance(stats, dict):
            namespaces = stats.get("namespaces", {})

        if not namespaces:
            print("\n  No namespaces found in this index.")
            return

        ns_list = sorted(namespaces.keys())
        print(f"\n  Namespaces ({len(ns_list)}):")
        for i, ns in enumerate(ns_list, 1):
            ns_info = namespaces[ns]
            count = (
                ns_info.get("vector_count", 0)
                if isinstance(ns_info, dict)
                else getattr(ns_info, "vector_count", 0)
            )
            label = ns if ns != "" else "(default)"
            print(f"    {i}. {label}  ({count:,} vectors)")

        selection = input("\n  Enter number to delete (or 'a' for all, 'c' to cancel): ").strip()

        if selection.lower() in ("c", "") :
            return

        if selection.lower() == "a":
            if input(f"  Delete ALL {len(ns_list)} namespace(s)? Type 'yes': ").strip().lower() != "yes":
                return
            for ns in ns_list:
                index.delete(delete_all=True, namespace=ns)
                print(f"    Deleted: {ns if ns else '(default)'}")
            print("  All namespaces deleted.")
            return

        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(ns_list):
                print("  Invalid selection.")
                return
        except ValueError:
            print("  Invalid selection.")
            return

        ns = ns_list[idx]
        label = ns if ns != "" else "(default)"
        if input(f"  Delete namespace '{label}'? Type 'yes': ").strip().lower() != "yes":
            return
        index.delete(delete_all=True, namespace=ns)
        print(f"  Deleted namespace: {label}")

    except Exception as e:
        print(f"  Error: {e}")


def _ensure_import_env(state: dict) -> bool:
    """Make sure bulk import environment is configured. Returns True if ready."""
    if state.get("env"):
        return True

    print("\n  Select your import environment:")
    print(f"    1. {ENV_LABELS[ENV_BYOC_AWS]}")
    print(f"    2. {ENV_LABELS[ENV_BYOC_GCP]}")
    print(f"    3. {ENV_LABELS[ENV_BYOC_AZURE]}")
    print(f"    4. {ENV_LABELS[ENV_SAAS]}")

    env_choice = input("\n  Environment [1/2/3/4]: ").strip()
    env_map = {
        "1": ENV_BYOC_AWS, "2": ENV_BYOC_GCP,
        "3": ENV_BYOC_AZURE, "4": ENV_SAAS,
        "aws": ENV_BYOC_AWS, "gcp": ENV_BYOC_GCP,
        "azure": ENV_BYOC_AZURE, "saas": ENV_SAAS,
    }
    env = env_map.get(env_choice.lower())
    if not env:
        print(f"  Invalid selection.")
        return False

    state["env"] = env
    print(f"  -> {ENV_LABELS[env]}")

    if env == ENV_SAAS:
        print("\n  Standard SaaS imports require a storage integration ID.")
        state["storage_integration_id"] = input("  Storage Integration ID: ").strip() or None

    return True


def menu_bulk_import(state: dict):
    """Sub-menu for bulk import operations."""
    if not state.get("index"):
        print("  No index connected. Use 'Switch Index' from the main menu.")
        return

    if not _ensure_import_env(state):
        return

    while True:
        env = state["env"]
        env_label = ENV_LABELS[env]
        section_header(f"BULK IMPORT  ({env_label})")
        print("  1. Start new import")
        print("  2. List imports")
        print("  3. Check import status by ID")
        print("  4. Cancel an import")
        print("  5. Validate storage path")
        print("  6. Delete a namespace")
        print("  7. Describe index stats")
        print("  8. Change import environment")
        print("  b. Back to main menu")

        choice = input("\n  Select option: ").strip().lower()

        if choice == "b" or not choice:
            break
        elif choice == "1":
            action_start_import(state)
        elif choice == "2":
            action_list_imports(state)
        elif choice == "3":
            import_id = input("  Enter import ID: ").strip()
            if import_id:
                verbose = input("  Verbose? [y/N]: ").strip().lower().startswith("y")
                monitor_import(state["index"], import_id, verbose=verbose)
        elif choice == "4":
            action_cancel_import(state)
        elif choice == "5":
            if env == ENV_SAAS:
                print("\n  Storage path validation is not available for Standard SaaS.")
            else:
                scheme = _uri_scheme_for_env(env)
                example = _uri_example_for_env(env)
                uri = input(f"  URI to validate ({scheme}) e.g. {example}: ").strip()
                if uri:
                    validate_uri(uri, env)
        elif choice == "6":
            action_delete_namespace(state)
        elif choice == "7":
            describe_index_stats(state["index"])
        elif choice == "8":
            state["env"] = None
            state["storage_integration_id"] = None
            if not _ensure_import_env(state):
                break
        else:
            print("  Invalid option.")


# ═══════════════════════════════════════════════════════════════════════════
# Load Testing
# ═══════════════════════════════════════════════════════════════════════════

class LoadTestMetrics:
    """Thread-safe metrics collector."""

    def __init__(self):
        self.lock = threading.Lock()
        self.operation_count = 0
        self.error_count = 0
        self.latencies: list[float] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

    def record(self, latency_ms: float, success: bool = True):
        with self.lock:
            if success:
                self.operation_count += 1
                self.latencies.append(latency_ms)
            else:
                self.error_count += 1

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def summary(self) -> dict:
        elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
        if not self.latencies:
            return {"operations": 0, "errors": self.error_count, "elapsed_sec": elapsed}

        sorted_lat = sorted(self.latencies)
        return {
            "operations": self.operation_count,
            "errors": self.error_count,
            "elapsed_sec": round(elapsed, 2),
            "ops_per_sec": round(self.operation_count / elapsed, 2) if elapsed > 0 else 0,
            "avg_latency_ms": round(statistics.mean(self.latencies), 2),
            "p50_latency_ms": round(statistics.median(self.latencies), 2),
            "p95_latency_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2) if sorted_lat else 0,
            "p99_latency_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2) if sorted_lat else 0,
            "min_latency_ms": round(min(self.latencies), 2),
            "max_latency_ms": round(max(self.latencies), 2),
        }


class MultiNamespaceMetrics:
    """Thread-safe metrics with per-namespace breakdown."""

    def __init__(self):
        self.lock = threading.Lock()
        self.global_metrics = LoadTestMetrics()
        self.per_namespace: dict[str, LoadTestMetrics] = defaultdict(LoadTestMetrics)

    def record(self, namespace: str, latency_ms: float, success: bool = True):
        self.global_metrics.record(latency_ms, success)
        with self.lock:
            ns_metrics = self.per_namespace[namespace]
        ns_metrics.record(latency_ms, success)

    def start(self):
        self.global_metrics.start()

    def stop(self):
        self.global_metrics.stop()

    def summary(self) -> dict:
        return self.global_metrics.summary()

    def per_namespace_summary(self) -> dict[str, dict]:
        with self.lock:
            namespaces = dict(self.per_namespace)
        result = {}
        for ns, m in namespaces.items():
            if m.latencies:
                result[ns] = {
                    "queries": m.operation_count,
                    "errors": m.error_count,
                    "avg_ms": round(statistics.mean(m.latencies), 2),
                    "p50_ms": round(statistics.median(m.latencies), 2),
                }
        return result


def _generate_random_vector(dimension: int = VECTOR_DIMENSION) -> list:
    return [random.uniform(-1, 1) for _ in range(dimension)]


def _generate_vector_batch(start_id: int, count: int) -> list:
    return [
        {
            "id": f"vec-{start_id + i}",
            "values": _generate_random_vector(),
            "metadata": {"batch_id": start_id // BATCH_SIZE},
        }
        for i in range(count)
    ]


def _upsert_batch(index, vectors: list, metrics: LoadTestMetrics):
    try:
        start = time.time()
        index.upsert(vectors=vectors)
        latency_ms = (time.time() - start) * 1000
        metrics.record(latency_ms, success=True)
        return len(vectors)
    except Exception as e:
        metrics.record(0, success=False)
        print(f"     Upsert error: {e}")
        return 0


def _query_random(index, metrics: LoadTestMetrics, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            start = time.time()
            index.query(vector=_generate_random_vector(), top_k=10)
            metrics.record((time.time() - start) * 1000, success=True)
        except Exception as e:
            metrics.record(0, success=False)
            if not stop_event.is_set():
                print(f"     Query error: {e}")


def _query_namespace(index, namespace: str, metrics: MultiNamespaceMetrics, stop_event: threading.Event, top_k: int = 10):
    while not stop_event.is_set():
        try:
            start = time.time()
            index.query(vector=_generate_random_vector(), top_k=top_k, namespace=namespace)
            metrics.record(namespace, (time.time() - start) * 1000, success=True)
        except Exception as e:
            metrics.record(namespace, 0, success=False)
            if not stop_event.is_set():
                print(f"     Query error [{namespace}]: {e}")


def _print_load_results(label: str, summary: dict, extra_lines: list[str] | None = None):
    print(f"\n  --- {label} ---")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print(f"  Operations:    {summary['operations']:,}")
    print(f"  Errors:        {summary['errors']}")
    print(f"  Total time:    {summary['elapsed_sec']}s")
    if summary.get("ops_per_sec"):
        print(f"  Throughput:    {summary['ops_per_sec']:,} ops/sec")
    if summary.get("avg_latency_ms") is not None:
        print(f"  Avg latency:   {summary['avg_latency_ms']}ms")
        print(f"  P50 latency:   {summary['p50_latency_ms']}ms")
        print(f"  P95 latency:   {summary['p95_latency_ms']}ms")
        print(f"  P99 latency:   {summary['p99_latency_ms']}ms")
        print(f"  Min latency:   {summary['min_latency_ms']}ms")
        print(f"  Max latency:   {summary['max_latency_ms']}ms")


def run_write_load_test(index, num_vectors: int, num_threads: int = DEFAULT_WRITE_THREADS):
    section_header("WRITE LOAD TEST")
    print(f"  Vectors to upsert: {num_vectors:,}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Threads:           {num_threads}")
    print(f"  Vector dimension:  {VECTOR_DIMENSION}")

    metrics = LoadTestMetrics()
    total_batches = (num_vectors + BATCH_SIZE - 1) // BATCH_SIZE
    vectors_upserted = 0

    print(f"\n  Generating and upserting {total_batches} batches...")
    metrics.start()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch_num in range(total_batches):
            start_id = batch_num * BATCH_SIZE
            count = min(BATCH_SIZE, num_vectors - start_id)
            vectors = _generate_vector_batch(start_id, count)
            futures.append(executor.submit(_upsert_batch, index, vectors, metrics))
            if (batch_num + 1) % 10 == 0:
                print(f"     Submitted {batch_num + 1}/{total_batches} batches...")

        for future in as_completed(futures):
            vectors_upserted += future.result()

    metrics.stop()
    summary = metrics.summary()
    vps = round(vectors_upserted / summary["elapsed_sec"], 2) if summary["elapsed_sec"] > 0 else 0
    _print_load_results("Write Results", summary, [f"Vectors upserted: {vectors_upserted:,}", f"Vector throughput: {vps} vectors/sec"])
    return vectors_upserted


def run_read_load_test(index, duration_seconds: int, num_threads: int = DEFAULT_READ_THREADS):
    section_header("READ LOAD TEST")
    print(f"  Duration:    {duration_seconds} seconds")
    print(f"  Threads:     {num_threads}")
    print(f"  Query type:  Random vector, top_k=10")

    metrics = LoadTestMetrics()
    stop_event = threading.Event()

    print(f"\n  Running queries for {duration_seconds}s...")
    metrics.start()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_query_random, index, metrics, stop_event) for _ in range(num_threads)]

        start = time.time()
        while time.time() - start < duration_seconds:
            remaining = duration_seconds - int(time.time() - start)
            print(f"     {remaining}s remaining... ({metrics.operation_count:,} queries so far)", end="\r")
            time.sleep(1)

        stop_event.set()
        print(f"\n     Stopping threads...")

    metrics.stop()
    _print_load_results("Read Results", metrics.summary())


def run_namespace_storm(index, duration_seconds: int, threads_per_namespace: int = DEFAULT_THREADS_PER_NAMESPACE, top_k: int = 10):
    section_header("MULTI-NAMESPACE QUERY STORM")

    print("  Discovering namespaces...")
    stats = index.describe_index_stats()
    namespaces = list(stats.namespaces.keys()) if stats.namespaces else []

    if not namespaces:
        print("  No namespaces found in index. Nothing to query.")
        return

    num_ns = len(namespaces)
    total_threads = num_ns * threads_per_namespace
    print(f"  Namespaces discovered: {num_ns}")
    print(f"  Threads per namespace: {threads_per_namespace}")
    print(f"  Total concurrent threads: {total_threads:,}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  top_k: {top_k}")
    print(f"  Total vectors in index: {stats.total_vector_count:,}")

    metrics = MultiNamespaceMetrics()
    stop_event = threading.Event()

    print(f"\n  Launching {total_threads:,} query threads across {num_ns} namespaces...")
    metrics.start()

    with ThreadPoolExecutor(max_workers=total_threads) as executor:
        futures = []
        for ns in namespaces:
            for _ in range(threads_per_namespace):
                futures.append(executor.submit(_query_namespace, index, ns, metrics, stop_event, top_k))

        start = time.time()
        last_ops = 0
        while time.time() - start < duration_seconds:
            elapsed = int(time.time() - start)
            remaining = duration_seconds - elapsed
            current_ops = metrics.global_metrics.operation_count
            current_errors = metrics.global_metrics.error_count
            delta = current_ops - last_ops
            last_ops = current_ops
            print(
                f"     {remaining:>4}s remaining | "
                f"{current_ops:>10,} queries | "
                f"~{delta:,} qps | "
                f"{current_errors} errors",
                end="\r",
            )
            time.sleep(1)

        stop_event.set()
        print(f"\n     Stopping {total_threads:,} threads...")

    metrics.stop()
    _print_load_results("Aggregate Results", metrics.summary())

    ns_summary = metrics.per_namespace_summary()
    if ns_summary:
        print(f"\n  {'Namespace':<30} {'Queries':>10} {'Errors':>8} {'Avg ms':>10} {'P50 ms':>10}")
        print("  " + "-" * 72)
        sorted_ns = sorted(ns_summary.items(), key=lambda x: x[1]["queries"], reverse=True)
        for ns_name, ns_data in sorted_ns[:20]:
            display = ns_name if ns_name else "(default)"
            print(f"  {display:<30} {ns_data['queries']:>10,} {ns_data['errors']:>8} {ns_data['avg_ms']:>10.2f} {ns_data['p50_ms']:>10.2f}")
        if len(sorted_ns) > 20:
            print(f"     ... and {len(sorted_ns) - 20} more namespaces")

        fastest = min(sorted_ns, key=lambda x: x[1]["avg_ms"])
        slowest = max(sorted_ns, key=lambda x: x[1]["avg_ms"])
        print(f"\n  Fastest: {fastest[0] or '(default)'} — avg {fastest[1]['avg_ms']}ms")
        print(f"  Slowest: {slowest[0] or '(default)'} — avg {slowest[1]['avg_ms']}ms")


def _timed(fn, *args, **kwargs):
    """Call fn and return (result, elapsed_ms)."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, (time.time() - t0) * 1000


def run_demo_loop(index):
    """Continuous loop cycling through upsert, query, list, fetch, update every 500ms."""
    section_header("DEMO MODE — Continuous CRUD")
    print("  Each cycle: upsert → query → list → fetch → update")
    print("  Interval: 500ms between cycles")
    print("  Press Ctrl+C to stop.\n")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    cycle = 0
    upserted_ids = []
    try:
        while True:
            cycle += 1
            vec_id = f"demo-{run_id}-{cycle}"
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            _, write_ms = _timed(
                index.upsert,
                vectors=[{"id": vec_id, "values": _generate_random_vector(), "metadata": {"demo_run": run_id, "cycle": cycle}}],
            )
            upserted_ids.append(vec_id)

            _, query_ms = _timed(index.query, vector=_generate_random_vector(), top_k=1)

            _, list_ms = _timed(index.list, prefix=f"demo-{run_id}-", limit=10)

            pick = upserted_ids[random.randint(0, len(upserted_ids) - 1)]
            _, fetch_ms = _timed(index.fetch, ids=[pick])

            _, update_ms = _timed(
                index.update,
                id=pick,
                set_metadata={"updated_cycle": cycle},
            )

            print(
                f"  [{ts}]  upsert {write_ms:5.0f}ms  "
                f"query {query_ms:5.0f}ms  "
                f"list {list_ms:5.0f}ms  "
                f"fetch {fetch_ms:5.0f}ms  "
                f"update {update_ms:5.0f}ms"
            )

            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\n  Demo stopped after {cycle} cycles. {len(upserted_ids)} vectors remain in index.")


def run_list_and_fetch_test(index):
    """Test list and fetch operations against the connected index."""
    section_header("LIST & FETCH TEST")

    namespace = input("  Namespace (Enter for default): ").strip() or ""
    ns_display = namespace or "(default)"

    prefix = input("  ID prefix filter (Enter for none): ").strip() or None
    limit = prompt_int("  Max IDs to list", 20)

    print(f"\n  Listing vectors in namespace '{ns_display}'...")
    list_kwargs = {"namespace": namespace, "limit": limit}
    if prefix:
        list_kwargs["prefix"] = prefix
        print(f"  Prefix filter: '{prefix}'")

    try:
        t0 = time.time()
        results = index.list(**list_kwargs)
        list_ms = (time.time() - t0) * 1000
    except Exception as e:
        print(f"  List failed: {e}")
        return

    vector_ids = []
    if hasattr(results, "vectors"):
        vector_ids = [v.id if hasattr(v, "id") else str(v) for v in results.vectors]
    elif isinstance(results, dict) and "vectors" in results:
        vector_ids = [v.get("id", str(v)) if isinstance(v, dict) else str(v) for v in results["vectors"]]
    else:
        for item in results:
            if isinstance(item, str):
                vector_ids.append(item)
            elif isinstance(item, list):
                vector_ids.extend(item)
            elif hasattr(item, "id"):
                vector_ids.append(item.id)

    print(f"\n  List returned {len(vector_ids)} ID(s) in {list_ms:.1f}ms")
    if not vector_ids:
        print("  No vectors found. Nothing to fetch.")
        return

    for vid in vector_ids[:20]:
        print(f"    - {vid}")
    if len(vector_ids) > 20:
        print(f"    ... and {len(vector_ids) - 20} more")

    fetch_count = min(len(vector_ids), prompt_int(f"\n  How many to fetch? (max {len(vector_ids)})", min(len(vector_ids), 10)))
    ids_to_fetch = vector_ids[:fetch_count]

    print(f"\n  Fetching {len(ids_to_fetch)} vector(s)...")
    try:
        t0 = time.time()
        fetch_result = index.fetch(ids=ids_to_fetch, namespace=namespace)
        fetch_ms = (time.time() - t0) * 1000
    except Exception as e:
        print(f"  Fetch failed: {e}")
        return

    fetched = {}
    if hasattr(fetch_result, "vectors"):
        fetched = fetch_result.vectors or {}
    elif isinstance(fetch_result, dict):
        fetched = fetch_result.get("vectors", {})

    print(f"  Fetched {len(fetched)} vector(s) in {fetch_ms:.1f}ms\n")

    for vid, vec in list(fetched.items())[:5]:
        values = None
        metadata = None
        if hasattr(vec, "values"):
            values = vec.values
            metadata = getattr(vec, "metadata", None)
        elif isinstance(vec, dict):
            values = vec.get("values")
            metadata = vec.get("metadata")

        dim_str = f"{len(values)}d" if values else "?"
        print(f"    {vid}  ({dim_str})")
        if metadata:
            print(f"      metadata: {dict(metadata) if not isinstance(metadata, dict) else metadata}")
    if len(fetched) > 5:
        print(f"    ... and {len(fetched) - 5} more")

    print(f"\n  Summary:")
    print(f"    List:  {len(vector_ids)} IDs in {list_ms:.1f}ms")
    print(f"    Fetch: {len(fetched)}/{len(ids_to_fetch)} vectors in {fetch_ms:.1f}ms")


def run_update_test(index):
    """Test update operations — upsert vectors then update their metadata and values."""
    section_header("UPDATE TEST")

    namespace = input("  Namespace (Enter for default): ").strip() or ""
    ns_display = namespace or "(default)"
    count = prompt_int("  Number of vectors to create and update", 5)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    ids = [f"update-test-{run_id}-{i}" for i in range(1, count + 1)]

    print(f"\n  Step 1: Upserting {count} vector(s) into namespace '{ns_display}'...")
    vectors = [
        {"id": vid, "values": _generate_random_vector(), "metadata": {"version": 1, "update_test": run_id}}
        for vid in ids
    ]
    try:
        t0 = time.time()
        index.upsert(vectors=vectors, namespace=namespace)
        upsert_ms = (time.time() - t0) * 1000
        print(f"  Upserted {count} vector(s) in {upsert_ms:.1f}ms")
    except Exception as e:
        print(f"  Upsert failed: {e}")
        return

    print(f"\n  Step 2: Fetching to confirm original state...")
    try:
        t0 = time.time()
        result = index.fetch(ids=ids, namespace=namespace)
        fetch_ms = (time.time() - t0) * 1000
        fetched = result.vectors if hasattr(result, "vectors") else result.get("vectors", {})
        print(f"  Fetched {len(fetched)} vector(s) in {fetch_ms:.1f}ms")
        for vid in ids[:3]:
            vec = fetched.get(vid)
            if vec:
                meta = getattr(vec, "metadata", None) or (vec.get("metadata") if isinstance(vec, dict) else None)
                print(f"    {vid}  metadata: {dict(meta) if meta and not isinstance(meta, dict) else meta}")
        if count > 3:
            print(f"    ... and {count - 3} more")
    except Exception as e:
        print(f"  Fetch failed: {e}")

    print(f"\n  Step 3: Updating vectors (new values + metadata version=2)...")
    update_times = []
    for vid in ids:
        try:
            t0 = time.time()
            index.update(
                id=vid,
                values=_generate_random_vector(),
                set_metadata={"version": 2, "updated_at": datetime.now().isoformat()},
                namespace=namespace,
            )
            ms = (time.time() - t0) * 1000
            update_times.append(ms)
            print(f"    Updated {vid} in {ms:.1f}ms")
        except Exception as e:
            print(f"    Update failed for {vid}: {e}")

    print(f"\n  Step 4: Fetching to verify updates...")
    try:
        t0 = time.time()
        result = index.fetch(ids=ids, namespace=namespace)
        fetch_ms = (time.time() - t0) * 1000
        fetched = result.vectors if hasattr(result, "vectors") else result.get("vectors", {})
        print(f"  Fetched {len(fetched)} vector(s) in {fetch_ms:.1f}ms")
        for vid in ids[:3]:
            vec = fetched.get(vid)
            if vec:
                meta = getattr(vec, "metadata", None) or (vec.get("metadata") if isinstance(vec, dict) else None)
                print(f"    {vid}  metadata: {dict(meta) if meta and not isinstance(meta, dict) else meta}")
        if count > 3:
            print(f"    ... and {count - 3} more")
    except Exception as e:
        print(f"  Fetch failed: {e}")

    print(f"\n  Summary:")
    print(f"    Upsert:  {count} vectors in {upsert_ms:.1f}ms")
    if update_times:
        avg = sum(update_times) / len(update_times)
        print(f"    Updates: {len(update_times)} in {sum(update_times):.1f}ms total (avg {avg:.1f}ms)")
    print(f"    Vectors remain in index under namespace '{ns_display}'.")


def action_delete_all_vectors(state: dict):
    index = state["index"]
    try:
        stats = index.describe_index_stats()
        total = stats.total_vector_count
        print(f"\n  Current vector count: {total:,}")
        if total == 0:
            print("  No vectors to delete.")
            return

        if not prompt_yes_no(f"  Delete ALL {total:,} vectors?", default=False):
            return

        print("  Deleting all vectors...")
        start = time.time()
        index.delete(delete_all=True)
        elapsed = time.time() - start
        print(f"  Delete completed in {elapsed:.2f}s")

        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"  Vectors remaining: {stats.total_vector_count:,}")
    except Exception as e:
        print(f"  Delete error: {e}")


def menu_load_testing(state: dict):
    """Sub-menu for load testing."""
    if not state.get("index"):
        print("  No index connected. Use 'Switch Index' from the main menu.")
        return

    index = state["index"]

    while True:
        section_header("LOAD TESTING")
        print("  1. Full test (write + read + optional delete)")
        print("  2. Write only (upsert vectors)")
        print("  3. Read only (query existing vectors)")
        print("  4. Multi-namespace query storm")
        print("  5. List & fetch test")
        print("  6. Update test")
        print("  7. Demo mode (continuous read/write every 500ms)")
        print("  8. Delete all test vectors")
        print("  9. Show index stats")
        print("  b. Back to main menu")

        choice = input("\n  Select option: ").strip().lower()

        if choice == "b" or not choice:
            break
        elif choice == "1":
            num_vectors = prompt_int("  Number of vectors to generate", 10000)
            write_threads = prompt_int("  Write threads", DEFAULT_WRITE_THREADS)
            run_write_load_test(index, num_vectors, write_threads)
            print("\n  Waiting 5s for vectors to be indexed...")
            time.sleep(5)
            describe_index_stats(index)
            read_duration = prompt_int("  Seconds to run read test", 30)
            read_threads = prompt_int("  Read threads", DEFAULT_READ_THREADS)
            run_read_load_test(index, read_duration, read_threads)
            if prompt_yes_no("\n  Delete all vectors?", default=False):
                action_delete_all_vectors(state)
        elif choice == "2":
            num_vectors = prompt_int("  Number of vectors to generate", 10000)
            write_threads = prompt_int("  Write threads", DEFAULT_WRITE_THREADS)
            run_write_load_test(index, num_vectors, write_threads)
            time.sleep(2)
            describe_index_stats(index)
        elif choice == "3":
            describe_index_stats(index)
            read_duration = prompt_int("  Seconds to run read test", 30)
            read_threads = prompt_int("  Read threads", DEFAULT_READ_THREADS)
            run_read_load_test(index, read_duration, read_threads)
        elif choice == "4":
            describe_index_stats(index)
            duration = prompt_int("  Seconds to run storm", 60)
            threads_per_ns = prompt_int("  Threads per namespace", DEFAULT_THREADS_PER_NAMESPACE)
            top_k = prompt_int("  top_k per query", 10)
            run_namespace_storm(index, duration, threads_per_ns, top_k)
        elif choice == "5":
            run_list_and_fetch_test(index)
        elif choice == "6":
            run_update_test(index)
        elif choice == "7":
            run_demo_loop(index)
        elif choice == "8":
            action_delete_all_vectors(state)
        elif choice == "9":
            describe_index_stats(index)
        else:
            print("  Invalid option.")


# ═══════════════════════════════════════════════════════════════════════════
# Backups
# ═══════════════════════════════════════════════════════════════════════════

def _format_size(size_bytes: int | float) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def _print_backup_details(b):
    print(f"    ID:      {b.backup_id}")
    print(f"    Name:    {b.name}")
    print(f"    Status:  {b.status}")
    if hasattr(b, "source_index_name") and b.source_index_name:
        print(f"    Source:  {b.source_index_name}")
    if hasattr(b, "description") and b.description:
        print(f"    Desc:    {b.description}")
    if hasattr(b, "record_count") and b.record_count is not None:
        print(f"    Records: {b.record_count:,}")
    if hasattr(b, "size_bytes") and b.size_bytes is not None:
        print(f"    Size:    {_format_size(b.size_bytes)}")
    if hasattr(b, "cloud") and b.cloud:
        print(f"    Cloud:   {b.cloud}")
    if hasattr(b, "region") and b.region:
        print(f"    Region:  {b.region}")
    if hasattr(b, "created_at") and b.created_at:
        print(f"    Created: {b.created_at}")


def monitor_backup(pc, backup_id: str, poll_interval: int = 5):
    print(f"\n  Monitoring backup '{backup_id}' (polling every {poll_interval}s)...")
    print("  Press Ctrl+C to stop monitoring and return to menu.")
    print("  " + "-" * 56)

    start_time = time.time()
    last_status = None

    try:
        while True:
            try:
                backup = pc.describe_backup(backup_id=backup_id)
                status = backup.status
                elapsed = int(time.time() - start_time)

                if status != last_status or elapsed % 30 == 0:
                    print(f"  [{elapsed:5d}s]  status: {status}")
                    last_status = status

                if status == "Ready":
                    print("  " + "-" * 56)
                    print(f"\n  Backup completed successfully in {elapsed}s!")
                    _print_backup_details(backup)
                    return backup

                elif status in ("Failed", "Cancelled"):
                    print("  " + "-" * 56)
                    print(f"\n  Backup {status.lower()}.")
                    _print_backup_details(backup)
                    return backup

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    Warning: error checking status: {e}")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n  Monitoring stopped. Backup '{backup_id}' may still be in progress.")
        return None


def action_create_backup(state: dict):
    pc = state["pc"]
    index_name = state.get("index_name") or "unknown"

    default_name = f"{index_name}-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    backup_name = input(f"  Backup name [{default_name}]: ").strip() or default_name
    description = input(f"  Description (optional): ").strip() or f"Backup of {index_name}"

    print(f"\n  Creating backup...")
    print(f"    Index:       {index_name}")
    print(f"    Backup name: {backup_name}")
    print(f"    Description: {description}")

    try:
        backup = pc.create_backup(
            index_name=index_name,
            backup_name=backup_name,
            description=description,
        )
        print(f"\n  Backup initiated!")
        print(f"    Backup ID: {backup.backup_id}")
        print(f"    Status:    {backup.status}")

        if prompt_yes_no("\n  Monitor progress?", default=True):
            monitor_backup(pc, backup.backup_id)

    except Exception as e:
        print(f"\n  Backup failed: {e}")


def action_list_backups(state: dict):
    pc = state["pc"]
    print("\n  Fetching backups...")
    try:
        backups = list(pc.list_backups())
    except Exception as e:
        print(f"  Error listing backups: {e}")
        return

    if not backups:
        print("  No backups found.")
        return

    print(f"\n  Found {len(backups)} backup(s):")
    print("  " + "-" * 56)
    for i, b in enumerate(backups, 1):
        icon = {"Ready": "+", "Failed": "!", "InProgress": "~"}.get(b.status, "?")
        print(f"  {i}. [{icon}] {b.name}")
        _print_backup_details(b)
        print("  " + "-" * 56)


def action_delete_backup(state: dict):
    pc = state["pc"]
    print("\n  Fetching backups...")
    try:
        backups = list(pc.list_backups())
    except Exception as e:
        print(f"  Error listing backups: {e}")
        return

    if not backups:
        print("  No backups found.")
        return

    print(f"\n  Backups ({len(backups)}):")
    print("  " + "-" * 56)
    for i, b in enumerate(backups, 1):
        icon = {"Ready": "+", "Failed": "!", "InProgress": "~"}.get(b.status, "?")
        source = getattr(b, "source_index_name", "?")
        print(f"    {i}. [{icon}] {b.name}  (source: {source}, status: {b.status})")
    print("  " + "-" * 56)

    selection = input(f"  Pick a backup to delete [1-{len(backups)}] or Enter to cancel: ").strip()
    if not selection:
        return

    try:
        sel_idx = int(selection) - 1
        if sel_idx < 0 or sel_idx >= len(backups):
            print("  Invalid selection.")
            return
    except ValueError:
        print("  Invalid selection.")
        return

    chosen = backups[sel_idx]
    print(f"\n  About to delete:")
    _print_backup_details(chosen)

    if input(f"\n  Type 'yes' to confirm deletion: ").strip().lower() != "yes":
        print("  Cancelled.")
        return

    try:
        pc.delete_backup(backup_id=chosen.backup_id)
        print(f"  Backup '{chosen.name}' deleted.")
    except Exception as e:
        print(f"  Error deleting backup: {e}")


def menu_backups(state: dict):
    """Sub-menu for backup operations."""
    pc = state["pc"]

    while True:
        index_name = state.get("index_name") or "(not connected)"
        section_header(f"BACKUPS  (Index: {index_name})")
        print("  1. Create new backup")
        print("  2. List all backups")
        print("  3. Check backup status by ID")
        print("  4. Delete a backup")
        print("  5. Describe index stats")
        print("  b. Back to main menu")

        choice = input("\n  Select option: ").strip().lower()

        if choice == "b" or not choice:
            break
        elif choice == "1":
            if not state.get("index_name"):
                print("  No index connected. Use 'Switch Index' from the main menu.")
            else:
                action_create_backup(state)
        elif choice == "2":
            action_list_backups(state)
        elif choice == "3":
            backup_id = input("  Enter backup ID: ").strip()
            if backup_id:
                monitor_backup(pc, backup_id)
        elif choice == "4":
            action_delete_backup(state)
        elif choice == "5":
            if state.get("index"):
                describe_index_stats(state["index"])
            else:
                print("  No index connected.")
        else:
            print("  Invalid option.")


# ═══════════════════════════════════════════════════════════════════════════
# Parquet Inspector
# ═══════════════════════════════════════════════════════════════════════════

def action_inspect_parquet():
    """Inspect a local parquet file for Pinecone compatibility."""
    if not HAS_PYARROW:
        print("  pyarrow not installed. Install with: pip install pyarrow")
        return

    filepath = input("  Path to parquet file: ").strip()
    if not filepath:
        return
    if not os.path.isfile(filepath):
        print(f"  File not found: {filepath}")
        return

    print(f"\n  Reading: {filepath}\n")
    table = pq.read_table(filepath)

    print(f"  Schema:")
    print(f"    {table.schema}\n")
    print(f"  Total rows: {table.num_rows:,}")
    print(f"  Columns:    {table.column_names}\n")

    if "values" in table.column_names:
        first_vec = table.column("values")[0].as_py()
        print(f"  Vector dimensions: {len(first_vec)}")
    else:
        print("  No 'values' column found (dense vectors).")

    if "sparse_values" in table.column_names:
        print("  Sparse values column: present")

    print(f"\n  Sample (first row):")
    for col in table.column_names:
        val = table.column(col)[0].as_py()
        if isinstance(val, list) and len(val) > 5:
            preview = f"[{val[0]}, {val[1]}, {val[2]}, ... ] ({len(val)} elements)"
        elif isinstance(val, str) and len(val) > 200:
            preview = val[:200] + "..."
        else:
            preview = val
        print(f"    {col}: {preview}")


# ═══════════════════════════════════════════════════════════════════════════
# Switch index helper
# ═══════════════════════════════════════════════════════════════════════════

def action_switch_index(state: dict):
    """Prompt the user to pick a new index to connect to."""
    pc = state["pc"]
    print("\n  How would you like to specify the index?")
    print("    1. List my indexes and pick one")
    print("    2. Enter the index host URL manually")
    print("    3. Enter the index name manually")

    hc = input("\n  Choice [1]: ").strip() or "1"

    new_name, new_host = None, None
    if hc == "1":
        new_name, new_host = pick_index(pc)
    elif hc == "2":
        new_host = input("  Index host: ").strip()
    elif hc == "3":
        new_name = input("  Index name: ").strip()
    else:
        print("  Invalid choice.")
        return

    if not new_name and not new_host:
        host_input = input("  Index host URL: ").strip()
        if not host_input:
            print("  Cancelled.")
            return
        new_host = host_input

    index, name, host = connect_to_index(pc, name=new_name, host=new_host)
    if index:
        state["index"] = index
        state["index_name"] = name
        state["index_host"] = host


# ═══════════════════════════════════════════════════════════════════════════
# Setup wizard
# ═══════════════════════════════════════════════════════════════════════════

def setup_wizard() -> dict:
    """Collect API key, pick an index, and return initial state."""
    print(BANNER)

    # 1. API key — env var first, then prompt
    env_key = os.environ.get("PINECONE_API_KEY")
    if env_key:
        use_env = input("  PINECONE_API_KEY found in environment. Use it? [Y/n]: ").strip().lower() or "y"
        if use_env.startswith("y"):
            api_key = env_key
            print("  Using API key from environment.")
        else:
            api_key = _masked_input("  Enter your Pinecone API key: ")
    else:
        print("  Tip: set PINECONE_API_KEY env var to skip this prompt next time.")
        api_key = _masked_input("  Enter your Pinecone API key: ")

    if not api_key:
        print("  Error: API key is required.")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    # 2. Pick an index
    print("\n  How would you like to specify the index?")
    print("    1. List my indexes and pick one")
    print("    2. Enter the index host URL manually")
    print("    3. Skip — I'll connect later")

    host_choice = input("\n  Choice [1]: ").strip() or "1"

    index, index_name, index_host = None, None, None

    if host_choice == "1":
        index_name, index_host = pick_index(pc)
        if index_host:
            index, index_name, index_host = connect_to_index(pc, name=index_name, host=index_host)
    elif host_choice == "2":
        host_input = input("  Index host: ").strip()
        if host_input:
            index, index_name, index_host = connect_to_index(pc, host=host_input)
    elif host_choice == "3":
        print("  Skipping index connection. You can connect later from the main menu.")
    else:
        print("  Invalid choice, skipping index connection.")

    return {
        "pc": pc,
        "index": index,
        "index_name": index_name,
        "index_host": index_host,
        "env": None,
        "storage_integration_id": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main menu
# ═══════════════════════════════════════════════════════════════════════════

def main():
    state = setup_wizard()

    while True:
        idx_display = state.get("index_name") or state.get("index_host") or "(not connected)"
        print(f"\n{'=' * 60}")
        print(f"  PINECONE TOOLKIT")
        print(f"  Index: {idx_display}")
        print(f"{'=' * 60}")
        print("  1. Index Management")
        print("  2. Bulk Import")
        print("  3. Load Testing")
        print("  4. Backups")
        print("  5. Demo Mode (continuous read/write)")
        print("  6. Describe Index Stats")
        print("  7. Inspect Parquet File")
        print("  8. Switch / Connect Index")
        print("  q. Quit")

        choice = input("\n  Select option: ").strip().lower()

        if choice in ("q", "quit"):
            print("\n  Goodbye!")
            break
        elif choice == "1":
            menu_index_management(state)
        elif choice == "2":
            menu_bulk_import(state)
        elif choice == "3":
            menu_load_testing(state)
        elif choice == "4":
            menu_backups(state)
        elif choice == "5":
            if state.get("index"):
                run_demo_loop(state["index"])
            else:
                print("  No index connected. Use option 8 to connect first.")
        elif choice == "6":
            if state.get("index"):
                describe_index_stats(state["index"])
            else:
                print("  No index connected. Use option 8 to connect first.")
        elif choice == "7":
            action_inspect_parquet()
        elif choice == "8":
            action_switch_index(state)
        else:
            print("  Invalid option.")


if __name__ == "__main__":
    main()
