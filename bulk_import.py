#!/usr/bin/env python3
"""
Pinecone Bulk Import — unified script for BYOC (AWS/GCP/Azure) and Standard SaaS.

Supports:
  - BYOC-AWS:   import from S3 (no storage integration needed)
  - BYOC-GCP:   import from GCS (no storage integration needed)
  - BYOC-Azure: import from Azure Blob Storage (no storage integration needed)
  - Standard:   import from S3/GCS with a storage integration ID
"""

from __future__ import annotations

import getpass
import os
import re
import sys
import time
from pinecone import Pinecone, ImportErrorMode


def _masked_input(prompt: str = "", mask: str = "*") -> str:
    """Read a line of input, printing a mask character for each keystroke."""
    try:
        import tty, termios
        sys.stdout.write(prompt)
        sys.stdout.flush()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        chars = []
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    break
                elif ch in ("\x7f", "\x08"):  # backspace / delete
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                elif ch == "\x03":  # Ctrl-C
                    raise KeyboardInterrupt
                elif ch == "\x04":  # Ctrl-D
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

# ---------------------------------------------------------------------------
# Optional cloud SDK imports
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Environment enum
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def _parse_s3_uri(uri: str):
    """Parse s3://bucket/prefix or s3://arn:aws:s3:...:accesspoint/<name>/prefix."""
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
    """Parse gs://bucket/prefix."""
    uri = uri.rstrip("/")
    match = re.match(r"^gs://([^/]+)(?:/(.*))?$", uri)
    if match:
        return match.group(1), match.group(2) or ""
    return None, None


def _parse_azure_uri(uri: str):
    """Parse azure://container/prefix  or  https://<account>.blob.core.windows.net/container/prefix."""
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
        ENV_BYOC_AZURE: "azure://",
        ENV_SAAS: "s3:// or gs://",
    }[env]


def _uri_example_for_env(env: str) -> str:
    return {
        ENV_BYOC_AWS: "s3://my-bucket/import-data",
        ENV_BYOC_GCP: "gs://my-bucket/import-data",
        ENV_BYOC_AZURE: "azure://my-container/import-data",
        ENV_SAAS: "s3://my-bucket/import-data",
    }[env]

# ---------------------------------------------------------------------------
# Storage validation — S3
# ---------------------------------------------------------------------------

def _get_s3_client():
    import botocore.exceptions
    try:
        s3 = boto3.client("s3")
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return s3
    except (botocore.exceptions.NoCredentialsError,
            botocore.exceptions.ClientError,
            botocore.exceptions.PartialCredentialsError):
        pass

    print("    No AWS credentials found. Enter them manually:")
    aws_key = input("    AWS Access Key ID: ").strip()
    aws_secret = getpass.getpass("    AWS Secret Access Key: ")
    aws_region = input("    AWS Region [us-east-1]: ").strip() or "us-east-1"
    if not aws_key or not aws_secret:
        return None
    return boto3.client(
        "s3",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region,
    )


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

# ---------------------------------------------------------------------------
# Storage validation — GCS
# ---------------------------------------------------------------------------

def _get_gcs_client():
    from google.auth.exceptions import DefaultCredentialsError
    try:
        client = gcs_storage.Client()
        list(client.list_buckets(max_results=1))
        return client
    except (DefaultCredentialsError, Exception):
        pass

    print("    No GCP credentials found via Application Default Credentials.")
    print("    Options:")
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

# ---------------------------------------------------------------------------
# Storage validation — Azure Blob Storage
# ---------------------------------------------------------------------------

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
    print("    Options:")
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

# ---------------------------------------------------------------------------
# Shared parquet layout analysis
# ---------------------------------------------------------------------------

def _analyze_parquet_layout(all_objects: list, list_prefix: str, provider: str) -> bool:
    """Analyze <namespace>/<file>.parquet structure and report findings."""
    namespaces = {}
    non_parquet = []
    bad_structure = []

    for obj in all_objects:
        key = obj["Key"]
        rel = key[len(list_prefix):]
        if not rel or rel.endswith("/"):
            continue

        parts = rel.split("/")
        if len(parts) == 2 and parts[1].endswith(".parquet"):
            ns = parts[0]
            namespaces.setdefault(ns, []).append({
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

# ---------------------------------------------------------------------------
# Dispatcher: validate storage URI based on environment
# ---------------------------------------------------------------------------

def validate_uri(uri: str, env: str) -> bool:
    if env == ENV_BYOC_AWS:
        return validate_s3(uri)
    if env == ENV_BYOC_GCP:
        return validate_gcs(uri)
    if env == ENV_BYOC_AZURE:
        return validate_azure(uri)
    if env == ENV_SAAS:
        if uri.startswith("gs://"):
            return validate_gcs(uri)
        return validate_s3(uri)
    return True


def _validate_uri_scheme(uri: str, env: str) -> bool:
    """Quick sanity check that the URI scheme matches the chosen environment."""
    if env == ENV_BYOC_AWS and not uri.startswith("s3://"):
        print(f"  [WARN] Expected s3:// URI for BYOC-AWS, got: {uri}")
        return False
    if env == ENV_BYOC_GCP and not uri.startswith("gs://"):
        print(f"  [WARN] Expected gs:// URI for BYOC-GCP, got: {uri}")
        return False
    if env == ENV_BYOC_AZURE and not (uri.startswith("azure://") or ".blob.core.windows.net" in uri):
        print(f"  [WARN] Expected azure:// URI for BYOC-Azure, got: {uri}")
        return False
    if env == ENV_SAAS and not (uri.startswith("s3://") or uri.startswith("gs://")):
        print(f"  [WARN] Standard SaaS import expects s3:// or gs:// URI, got: {uri}")
        return False
    return True

# ---------------------------------------------------------------------------
# Import monitoring & management (shared across all environments)
# ---------------------------------------------------------------------------

def monitor_import(index, import_id: str, poll_interval: int = 10, verbose: bool = False):
    """Poll import status until it reaches a terminal state."""
    print(f"\nMonitoring import '{import_id}' (polling every {poll_interval}s)...")
    if verbose:
        print("Verbose mode: dumping full API response each poll.")
    print("Press Ctrl+C to stop monitoring and return to menu.")
    print("-" * 60)

    start_time = time.time()
    last_pct = None
    last_print_time = 0
    last_raw = None

    try:
        while True:
            try:
                status = index.describe_import(id=import_id)
                elapsed = int(time.time() - start_time)
                pct = getattr(status, "percent_complete", None)
                state = status.status

                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                records = getattr(status, "records_imported", None) or 0

                if records:
                    print(f"[{elapsed:5d}s]  status: {state:<12}  progress: {pct_str:<8}  records imported: {records:,}")
                else:
                    print(f"[{elapsed:5d}s]  status: {state:<12}  progress: {pct_str:<8}")

                if verbose:
                    raw = _extract_all_fields(status)
                    if raw != last_raw:
                        for k, v in raw.items():
                            print(f"           {k}: {v}")
                        last_raw = raw

                if state == "Completed":
                    print("-" * 60)
                    print(f"\nImport completed successfully in {elapsed}s!")
                    _print_import_details(status)
                    return status

                if state in ("Failed", "Cancelled"):
                    print("-" * 60)
                    print(f"\nImport {state.lower()}.")
                    _print_import_details(status)
                    if hasattr(status, "error") and status.error:
                        print(f"  Error: {status.error}")
                    return status

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  Warning: error checking status: {e}")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Import '{import_id}' is still running.")
        print(f"Use option 3 to check its status later.")
        return None


def _extract_all_fields(obj):
    result = {}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
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
    print(f"\nImport details:")
    print(f"  ID:               {status.id}")
    print(f"  URI:              {getattr(status, 'uri', 'N/A')}")
    print(f"  Status:           {status.status}")
    pct = getattr(status, "percent_complete", None)
    if pct is not None:
        print(f"  Percent complete: {pct:.1f}%")
    records = getattr(status, "records_imported", None)
    if records is not None:
        print(f"  Records imported: {records:,}")
    created = getattr(status, "created_at", None)
    if created:
        print(f"  Created at:       {created}")
    finished = getattr(status, "finished_at", None)
    if finished:
        print(f"  Finished at:      {finished}")


def list_imports(index):
    print("\nFetching imports...")
    try:
        imports = list(index.list_imports())
        if not imports:
            print("No imports found.")
            return
        imports.reverse()
        print(f"\nFound {len(imports)} import(s) (newest first):")
        print("-" * 70)
        for imp in imports:
            _print_import_details(imp)
            if hasattr(imp, "error") and imp.error:
                print(f"  Error: {imp.error}")
            print("-" * 70)
    except Exception as e:
        print(f"Error listing imports: {e}")


def cancel_import(index):
    import_id = input("Enter import ID to cancel: ").strip()
    if not import_id:
        print("No ID provided.")
        return
    try:
        index.cancel_import(id=import_id)
        print(f"Cancel request sent for import '{import_id}'.")
    except Exception as e:
        print(f"Error cancelling import: {e}")


def delete_namespace(index):
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})

        if not namespaces:
            print("\nNo namespaces found in this index.")
            return

        ns_list = sorted(namespaces.keys())
        print(f"\nNamespaces ({len(ns_list)}):")
        for i, ns in enumerate(ns_list, 1):
            ns_info = namespaces[ns]
            count = ns_info.get("vector_count", 0) if isinstance(ns_info, dict) else getattr(ns_info, "vector_count", 0)
            label = ns if ns != "" else "(default)"
            print(f"  {i}. {label}  ({count:,} vectors)")

        selection = input("\nEnter number to delete (or 'a' to delete all, 'c' to cancel): ").strip()

        if selection.lower() == "c" or not selection:
            print("Cancelled.")
            return

        if selection.lower() == "a":
            confirm = input(f"Delete ALL {len(ns_list)} namespace(s)? Type 'yes' to confirm: ").strip()
            if confirm.lower() != "yes":
                print("Cancelled.")
                return
            for ns in ns_list:
                index.delete(delete_all=True, namespace=ns)
                label = ns if ns != "" else "(default)"
                print(f"  Deleted namespace: {label}")
            print("All namespaces deleted.")
            return

        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(ns_list):
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid selection.")
            return

        ns = ns_list[idx]
        label = ns if ns != "" else "(default)"
        confirm = input(f"Delete namespace '{label}'? Type 'yes' to confirm: ").strip()
        if confirm.lower() != "yes":
            print("Cancelled.")
            return

        index.delete(delete_all=True, namespace=ns)
        print(f"Deleted namespace: {label}")

    except Exception as e:
        print(f"Error: {e}")


def describe_index_stats(index):
    try:
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        dim = getattr(stats, "dimension", None)
        fullness = getattr(stats, "index_fullness", None)
        namespaces = getattr(stats, "namespaces", {}) or {}

        print(f"\nIndex Stats:")
        if dim is not None:
            print(f"  Dimension:      {dim}")
        if total is not None:
            print(f"  Total vectors:  {total:,}")
        if fullness is not None:
            print(f"  Index fullness: {fullness}")

        if namespaces:
            print(f"\n  Namespaces ({len(namespaces)}):")
            for ns in sorted(namespaces.keys()):
                ns_info = namespaces[ns]
                count = ns_info.get("vector_count", 0) if isinstance(ns_info, dict) else getattr(ns_info, "vector_count", 0)
                label = ns if ns != "" else "(default)"
                print(f"    - {label}: {count:,} vectors")
        else:
            print(f"\n  No namespaces found.")
    except Exception as e:
        print(f"Error: {e}")

# ---------------------------------------------------------------------------
# Start import — dispatches based on environment
# ---------------------------------------------------------------------------

def start_import(index, env: str, storage_integration_id: str | None):
    if env == ENV_SAAS:
        uri = input("Storage URI (e.g. s3://my-bucket/import-data): ").strip()
        if not uri:
            print("Error: URI is required.")
            return
    else:
        scheme = _uri_scheme_for_env(env)
        example = _uri_example_for_env(env)
        uri = input(f"Storage URI ({scheme}) e.g. {example}: ").strip()
        if not uri:
            print("Error: URI is required.")
            return

        if not _validate_uri_scheme(uri, env):
            proceed = input("URI scheme doesn't match environment. Continue anyway? [y/N]: ").strip().lower()
            if not proceed.startswith("y"):
                print("Cancelled.")
                return

        run_validation = input("Run storage path validation before importing? [y/N]: ").strip().lower()
        if run_validation.startswith("y"):
            if not validate_uri(uri, env):
                proceed = input("\nValidation failed. Start import anyway? [y/N]: ").strip().lower()
                if not proceed.startswith("y"):
                    print("Import cancelled.")
                    return

    error_mode_input = input("Error mode — [c]ontinue or [a]bort? [c]: ").strip().lower() or "c"
    error_mode = ImportErrorMode.ABORT if error_mode_input.startswith("a") else ImportErrorMode.CONTINUE

    print(f"\nStarting import...")
    print(f"  Environment:    {ENV_LABELS[env]}")
    print(f"  URI:            {uri}")
    if storage_integration_id:
        print(f"  Integration ID: {storage_integration_id}")
    print(f"  Error mode:     {'ABORT' if error_mode == ImportErrorMode.ABORT else 'CONTINUE'}")

    import_kwargs = dict(uri=uri, error_mode=error_mode)
    if storage_integration_id:
        import_kwargs["integration_id"] = storage_integration_id

    try:
        resp = index.start_import(**import_kwargs)
        import_id = resp.id
        print(f"\nImport started!  ID: {import_id}")

        monitor = input("\nMonitor progress? [Y/n/v(erbose)]: ").strip().lower() or "y"
        if monitor.startswith("v"):
            monitor_import(index, import_id, verbose=True)
        elif monitor.startswith("y"):
            monitor_import(index, import_id)

    except Exception as e:
        print(f"\nFailed to start import: {e}")

# ---------------------------------------------------------------------------
# Index listing & picker
# ---------------------------------------------------------------------------

def list_and_pick_index(pc: Pinecone) -> str | None:
    """List available indexes and let the user pick one, returning its host."""
    print("\nFetching indexes...")
    try:
        indexes = pc.list_indexes()
        idx_list = list(indexes)
    except Exception as e:
        print(f"  Could not list indexes: {e}")
        return None

    if not idx_list:
        print("  No indexes found in this project.")
        return None

    print(f"\nAvailable indexes ({len(idx_list)}):")
    print("-" * 70)
    for i, idx in enumerate(idx_list, 1):
        name = getattr(idx, "name", str(idx))
        host = getattr(idx, "host", "N/A")
        dim = getattr(idx, "dimension", "?")
        metric = getattr(idx, "metric", "?")
        status_obj = getattr(idx, "status", None)
        state = getattr(status_obj, "state", "?") if status_obj else "?"
        print(f"  {i}. {name}  (dim={dim}, metric={metric}, state={state})")
        print(f"     host: {host}")
    print("-" * 70)

    selection = input(f"Pick an index [1-{len(idx_list)}] or Enter to type host manually: ").strip()
    if not selection:
        return None
    try:
        sel_idx = int(selection) - 1
        if sel_idx < 0 or sel_idx >= len(idx_list):
            print("Invalid selection.")
            return None
        host = getattr(idx_list[sel_idx], "host", None)
        if host:
            return f"https://{host}" if not host.startswith("https://") else host
    except ValueError:
        print("Invalid selection.")
    return None

# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_connection(index) -> bool:
    """Quick connectivity check by calling describe_index_stats."""
    try:
        stats = index.describe_index_stats()
        dim = getattr(stats, "dimension", "?")
        total = getattr(stats, "total_vector_count", 0)
        print(f"  Connected! (dimension={dim}, vectors={total:,})")
        return True
    except Exception as e:
        print(f"  Connection failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

def setup_wizard():
    """Interactive setup: collect API key, environment, index host, and integration ID."""
    print("=" * 60)
    print("  PINECONE BULK IMPORT")
    print("  One script to rule them all.")
    print("=" * 60)

    # 1. API key (check env var first)
    env_key = os.environ.get("PINECONE_API_KEY")
    if env_key:
        use_env = input(f"\nPINECONE_API_KEY found in environment. Use it? [Y/n]: ").strip().lower() or "y"
        if use_env.startswith("y"):
            api_key = env_key
        else:
            api_key = _masked_input("Enter your Pinecone API key: ")
    else:
        api_key = _masked_input("\nEnter your Pinecone API key: ")

    if not api_key:
        print("Error: API key is required.")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    # 2. Environment selection
    print("\nSelect your environment:")
    print(f"  1. {ENV_LABELS[ENV_BYOC_AWS]}")
    print(f"  2. {ENV_LABELS[ENV_BYOC_GCP]}")
    print(f"  3. {ENV_LABELS[ENV_BYOC_AZURE]}")
    print(f"  4. {ENV_LABELS[ENV_SAAS]}")

    env_choice = input("\nEnvironment [1/2/3/4]: ").strip()
    env_map = {
        "1": ENV_BYOC_AWS,
        "2": ENV_BYOC_GCP,
        "3": ENV_BYOC_AZURE,
        "4": ENV_SAAS,
        "aws": ENV_BYOC_AWS,
        "gcp": ENV_BYOC_GCP,
        "azure": ENV_BYOC_AZURE,
        "saas": ENV_SAAS,
    }
    env = env_map.get(env_choice.lower())
    if not env:
        print(f"Invalid selection: '{env_choice}'. Please enter 1, 2, 3, or 4.")
        sys.exit(1)

    print(f"\n  -> {ENV_LABELS[env]}")

    # 3. Storage integration ID (SaaS only)
    storage_integration_id = None
    if env == ENV_SAAS:
        print("\n  Standard SaaS imports require a storage integration ID.")
        print("  (Create one in the Pinecone console under Storage Integrations.)")
        storage_integration_id = input("  Storage Integration ID: ").strip()
        if not storage_integration_id:
            print("  Warning: no integration ID provided. Import may fail.")

    # 4. Index host — offer to list indexes or type manually
    print("\nHow would you like to specify the index?")
    print("  1. List my indexes and pick one")
    print("  2. Enter the index host URL manually")

    host_choice = input("\nChoice [1]: ").strip() or "1"

    index_host = None
    if host_choice == "1":
        index_host = list_and_pick_index(pc)

    if not index_host:
        index_host = input("Index host (e.g. https://my-index-abc123.svc.aped-1234.pinecone.io): ").strip()

    if not index_host:
        print("Error: Index host is required.")
        sys.exit(1)

    # 5. Connect and test
    print(f"\nConnecting to index...")
    index = pc.Index(host=index_host)
    if not test_connection(index):
        proceed = input("Connection test failed. Continue anyway? [y/N]: ").strip().lower()
        if not proceed.startswith("y"):
            sys.exit(1)

    return pc, index, env, storage_integration_id, index_host

# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    pc, index, env, storage_integration_id, index_host = setup_wizard()

    env_label = ENV_LABELS[env]
    integration_display = storage_integration_id or "(none)"

    while True:
        print(f"\n{'=' * 60}")
        print(f"  Env:   {env_label}")
        print(f"  Index: {index_host}")
        if env == ENV_SAAS:
            print(f"  Integration ID: {integration_display}")
        print(f"{'=' * 60}")
        print("  1. Start new import")
        print("  2. List imports")
        print("  3. Check import status by ID")
        print("  4. Cancel an import")
        print("  5. Describe index stats")
        print("  6. Choose a namespace to delete")
        print(f"  7. Validate storage path{'' if env != ENV_SAAS else ' (BYOC only)'}")
        print("  8. Switch index")
        print("  q. Quit")

        choice = input("\nSelect option [1]: ").strip() or "1"

        if choice in ("q", "Q"):
            print("Bye!")
            break

        elif choice == "1":
            start_import(index, env, storage_integration_id)

        elif choice == "2":
            list_imports(index)

        elif choice == "3":
            import_id = input("Enter import ID: ").strip()
            if import_id:
                verb = input("Verbose? [y/N]: ").strip().lower().startswith("y")
                monitor_import(index, import_id, verbose=verb)

        elif choice == "4":
            cancel_import(index)

        elif choice == "5":
            describe_index_stats(index)

        elif choice == "6":
            delete_namespace(index)

        elif choice == "7":
            if env == ENV_SAAS:
                print("\n  Storage path validation is not available for Standard SaaS.")
                print("  Pinecone accesses storage through the storage integration.")
            else:
                scheme = _uri_scheme_for_env(env)
                example = _uri_example_for_env(env)
                uri = input(f"URI to validate ({scheme}) e.g. {example}: ").strip()
                if uri:
                    validate_uri(uri, env)

        elif choice == "8":
            print("\nSwitch environment too?")
            print(f"  Current: {env_label}")
            print(f"  1. {ENV_LABELS[ENV_BYOC_AWS]}")
            print(f"  2. {ENV_LABELS[ENV_BYOC_GCP]}")
            print(f"  3. {ENV_LABELS[ENV_BYOC_AZURE]}")
            print(f"  4. {ENV_LABELS[ENV_SAAS]}")
            print(f"  Enter. Keep current")
            env_input = input("\nEnvironment [Enter to keep]: ").strip().lower()
            new_env_map = {
                "1": ENV_BYOC_AWS, "2": ENV_BYOC_GCP,
                "3": ENV_BYOC_AZURE, "4": ENV_SAAS,
                "aws": ENV_BYOC_AWS, "gcp": ENV_BYOC_GCP,
                "azure": ENV_BYOC_AZURE, "saas": ENV_SAAS,
            }
            if env_input and env_input in new_env_map:
                env = new_env_map[env_input]
                env_label = ENV_LABELS[env]
                if env == ENV_SAAS:
                    storage_integration_id = input("  Storage Integration ID: ").strip() or None
                    integration_display = storage_integration_id or "(none)"
                else:
                    storage_integration_id = None
                    integration_display = "(none)"
            elif env_input:
                print("Invalid selection, keeping current environment.")

            print("\nHow would you like to specify the new index?")
            print("  1. List my indexes and pick one")
            print("  2. Enter the index host URL manually")
            hc = input("\nChoice [1]: ").strip() or "1"
            new_host = None
            if hc == "1":
                new_host = list_and_pick_index(pc)
            if not new_host:
                new_host = input("Index host: ").strip()
            if new_host:
                index_host = new_host
                print(f"\nConnecting to index...")
                index = pc.Index(host=index_host)
                test_connection(index)

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
