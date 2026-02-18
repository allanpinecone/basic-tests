"""Bare minimum BYOC bulk import for a SageMaker notebook."""

import time
from pinecone import Pinecone, ImportErrorMode

# ── Fill these in ──────────────────────────────────────────────
API_KEY = "your-pinecone-api-key"
INDEX_HOST = "https://your-index-abc123.svc.aped-1234.pinecone.io"
S3_URI = "s3://your-bucket/your-import-prefix"
# ───────────────────────────────────────────────────────────────

pc = Pinecone(api_key=API_KEY)
index = pc.Index(host=INDEX_HOST)

# Start the import (no integration_id needed for BYOC)
resp = index.start_import(uri=S3_URI, error_mode=ImportErrorMode.CONTINUE)
print(f"Import started: {resp.id}")

# Poll until done
while True:
    status = index.describe_import(id=resp.id)
    pct = getattr(status, "percent_complete", 0) or 0
    records = getattr(status, "records_imported", 0) or 0
    print(f"  {status.status}  {pct:.1f}%  {records:,} records")
    if status.status in ("Completed", "Failed", "Cancelled"):
        break
    time.sleep(10)

# Dump everything from the final status
print(f"\n{'=' * 60}")
print(f"Final status: {status.status}")
print(f"{'=' * 60}")
if hasattr(status, "to_dict"):
    for k, v in status.to_dict().items():
        print(f"  {k}: {v}")
else:
    for attr in dir(status):
        if not attr.startswith("_"):
            val = getattr(status, attr, None)
            if not callable(val):
                print(f"  {attr}: {val}")
