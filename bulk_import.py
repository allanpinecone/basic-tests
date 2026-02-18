#!/usr/bin/env python3
"""Pinecone Bulk Import Script — import data from S3 into a BYOC index."""

import getpass
import re
import time
from pinecone import Pinecone, ImportErrorMode

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# For BYOC indexes, no storage integration is needed — the data plane
# runs in your AWS account and can access S3 directly.
# Set this only if importing into a standard (non-BYOC) serverless index.
STORAGE_INTEGRATION_ID = None


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

                if pct != last_pct or (elapsed - last_print_time) >= 30:
                    pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                    records = getattr(status, "records_imported", None) or 0

                    if records:
                        print(f"[{elapsed:5d}s]  status: {state:<12}  progress: {pct_str:<8}  records imported: {records:,}")
                    else:
                        print(f"[{elapsed:5d}s]  status: {state:<12}  progress: {pct_str:<8}")

                    last_pct = pct
                    last_print_time = elapsed

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
                print(f"  ⚠ Error checking status: {e}")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Import '{import_id}' is still running.")
        print(f"Use option 3 to check its status later.")
        return None


def _extract_all_fields(obj):
    """Extract all fields from a Pinecone response object into a dict."""
    result = {}
    # Try dict-like access
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    # Fall back to inspecting attributes
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
    """Pretty-print the fields of an import status object."""
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
    """List all recent and ongoing imports for the index."""
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


def delete_namespace(index):
    """List namespaces and let the user pick one to delete."""
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


def _parse_s3_uri(uri: str):
    """Parse an S3 URI into (bucket_or_arn, prefix).

    Supports:
      s3://bucket-name/prefix
      s3://arn:aws:s3:<region>:<account>:accesspoint/<name>/prefix
    """
    uri = uri.rstrip("/")

    # S3 Access Point ARN: s3://arn:aws:s3:...:accesspoint/<name>[/prefix]
    arn_match = re.match(
        r"^s3://(arn:aws:s3:[^:]*:[^:]*:accesspoint/[^/]+)(?:/(.*))?$", uri
    )
    if arn_match:
        return arn_match.group(1), arn_match.group(2) or ""

    # Standard bucket: s3://bucket-name[/prefix]
    std_match = re.match(r"^s3://([^/]+)(?:/(.*))?$", uri)
    if std_match:
        return std_match.group(1), std_match.group(2) or ""

    return None, None


def _get_s3_client():
    """Create an S3 client, prompting for credentials if needed."""
    import botocore.exceptions

    # First try default credential chain (instance profile, env vars, ~/.aws)
    try:
        s3 = boto3.client("s3")
        sts = boto3.client("sts")
        sts.get_caller_identity()  # verify credentials work
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


def validate_s3(s3_uri: str):
    """Validate S3 access and directory structure before importing.

    Returns True if validation passes, False otherwise.
    """
    if not HAS_BOTO3:
        print("\n  [skip] boto3 not installed — skipping S3 pre-flight check.")
        return True

    bucket, prefix = _parse_s3_uri(s3_uri)
    if not bucket:
        print(f"\n  [FAIL] Could not parse S3 URI: {s3_uri}")
        return False

    print(f"\n  Validating S3 access...")
    print(f"    Bucket: {bucket}")
    print(f"    Prefix: {prefix or '(root)'}")

    s3 = _get_s3_client()
    if not s3:
        print(f"\n  [FAIL] No valid AWS credentials provided.")
        return False

    # 1. Check bucket access
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

    # 2. List objects under prefix and analyze structure
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
        print(f"    [FAIL] No objects found under '{s3_uri}'.")
        return False

    # 3. Analyze namespace structure
    namespaces = {}  # namespace_name -> list of parquet files
    non_parquet = []
    bad_structure = []

    for obj in all_objects:
        key = obj["Key"]
        # Strip the prefix to get the relative path
        rel = key[len(list_prefix):]
        if not rel or rel.endswith("/"):
            continue  # skip directory markers

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

    # 4. Report findings
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

    if ok:
        print(f"\n    S3 validation: PASSED")
    else:
        print(f"\n    S3 validation: WARNINGS (see above)")

    return True


def cancel_import(index):
    """Cancel a running import by ID."""
    import_id = input("Enter import ID to cancel: ").strip()
    if not import_id:
        print("No ID provided.")
        return
    try:
        index.cancel_import(id=import_id)
        print(f"Cancel request sent for import '{import_id}'.")
    except Exception as e:
        print(f"Error cancelling import: {e}")


def start_import(index):
    """Start a new bulk import from S3."""
    s3_uri = input("S3 URI (e.g. s3://bucket/import-dir): ").strip()
    if not s3_uri:
        print("Error: S3 URI is required.")
        return

    # Pre-flight S3 validation
    if not validate_s3(s3_uri):
        proceed = input("\nValidation failed. Start import anyway? [y/N]: ").strip().lower()
        if not proceed.startswith("y"):
            print("Import cancelled.")
            return

    error_mode_input = input("Error mode — [c]ontinue or [a]bort? [c]: ").strip().lower() or "c"
    error_mode = ImportErrorMode.ABORT if error_mode_input.startswith("a") else ImportErrorMode.CONTINUE

    print(f"\nStarting import...")
    print(f"  URI:            {s3_uri}")
    print(f"  Integration ID: {STORAGE_INTEGRATION_ID or '(none — BYOC direct access)'}")
    print(f"  Error mode:     {'ABORT' if error_mode == ImportErrorMode.ABORT else 'CONTINUE'}")

    import_kwargs = dict(
        uri=s3_uri,
        error_mode=error_mode,
    )
    if STORAGE_INTEGRATION_ID:
        import_kwargs["integration_id"] = STORAGE_INTEGRATION_ID

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


def main():
    print("=" * 60)
    print("PINECONE BULK IMPORT")
    print("=" * 60)

    api_key = getpass.getpass("Enter your Pinecone API key: ")
    pc = Pinecone(api_key=api_key)

    index_host = input("Index host (e.g. https://my-index-abc123.svc.aped-1234.pinecone.io): ").strip()
    if not index_host:
        print("Error: Index host is required.")
        return

    print(f"\nConnecting to index at {index_host}...")
    index = pc.Index(host=index_host)

    while True:
        print("\nOptions:")
        print("  1. Start new import")
        print("  2. List imports")
        print("  3. Check import status by ID")
        print("  4. Cancel an import")
        print("  5. Describe index stats")
        print("  6. Delete namespace")
        print("  7. Validate S3 path")
        print("  q. Quit")

        choice = input("\nSelect option [1]: ").strip() or "1"

        if choice in ("q", "Q"):
            print("Bye!")
            break
        elif choice == "1":
            start_import(index)
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
        elif choice == "6":
            delete_namespace(index)
        elif choice == "7":
            uri = input("S3 URI to validate: ").strip()
            if uri:
                validate_s3(uri)
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
