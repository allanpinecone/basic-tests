# Pinecone Toolkit

One script to rule them all. An interactive CLI that covers index management, bulk import, load testing, and backups for Pinecone — including BYOC (AWS, GCP, Azure) and standard SaaS environments.

## Quick Start

```bash
pip install -r requirements.txt
python pinecone_toolkit.py
```

To skip the API key prompt on every run, set it as an environment variable:

```bash
export PINECONE_API_KEY=pcsk_...
python pinecone_toolkit.py
```

## Requirements

- Python 3.10+
- `pinecone` — Pinecone SDK
- `pyarrow` — for local parquet file inspection

Install everything:

```bash
pip install -r requirements.txt
```

### Optional dependencies

These are only needed if you want storage path validation for bulk imports:

| Package | When you need it |
|---------|-----------------|
| `boto3` | S3 validation (BYOC-AWS) |
| `google-cloud-storage` | GCS validation (BYOC-GCP) |
| `azure-storage-blob` | Azure Blob validation (BYOC-Azure) |

## What's Inside

The toolkit is organized into sub-menus accessible from a single main menu:

### 1. Index Management
- List all indexes in your project
- Describe index stats (dimensions, vector count, namespaces)
- Test connectivity with DNS/TCP/HTTPS checks and a test vector upsert
- Create a new serverless index
- Delete an index

### 2. Bulk Import
- Start a new bulk import from S3, GCS, or Azure Blob Storage
- List, monitor, and cancel imports
- Validate storage paths before importing (checks bucket access, parquet layout, namespace structure)
- Delete namespaces
- Supports all environments: BYOC-AWS, BYOC-GCP, BYOC-Azure, and standard SaaS (with storage integration ID)

### 3. Load Testing
- **Full test** — write + read + optional cleanup
- **Write only** — multi-threaded batch upserts with configurable thread count
- **Read only** — multi-threaded random vector queries for a set duration
- **Multi-namespace query storm** — discovers all namespaces and hammers them concurrently with per-namespace latency breakdowns
- Results include throughput, p50/p95/p99 latencies, and error counts

### 4. Backups
- Create backups with auto-generated names
- List all backups with status indicators
- Monitor backup progress with polling
- Delete backups

### 5. Utilities
- **Inspect Parquet File** — validate a local `.parquet` file's schema, dimensions, and structure for Pinecone compatibility
- **Switch Index** — change your connected index at any time without restarting

## Standalone Scripts & Guides

| File | Purpose |
|------|---------|
| `byoc_bulk_import.py` | Self-contained bulk import into a BYOC index, with private-endpoint detection |
| `BYOC_BULK_IMPORT_TROUBLESHOOTING.md` | Troubleshooting guide for failed BYOC bulk imports — permissions, network routing, data layout, and a diagnostic playbook |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PINECONE_API_KEY` | Pinecone API key (skips the interactive prompt) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string (for import validation) |
| `AZURE_STORAGE_ACCOUNT` | Azure storage account name (alternative to connection string) |
| `AZURE_STORAGE_KEY` | Azure storage account key (used with `AZURE_STORAGE_ACCOUNT`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account key file (for GCS validation) |
