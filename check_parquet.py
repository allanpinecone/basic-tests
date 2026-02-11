#!/usr/bin/env python3
"""Inspect a Parquet file to verify it's compatible with a Pinecone index."""

import sys
import pyarrow.parquet as pq


def inspect_parquet(filepath: str):
    print(f"Reading: {filepath}\n")

    table = pq.read_table(filepath)

    print(f"Schema:")
    print(f"  {table.schema}\n")

    print(f"Total rows: {table.num_rows:,}")
    print(f"Columns:    {table.column_names}\n")

    # Check vector dimensions
    if "values" in table.column_names:
        first_vec = table.column("values")[0].as_py()
        print(f"Vector dimensions: {len(first_vec)}")
    else:
        print("No 'values' column found (dense vectors).")

    if "sparse_values" in table.column_names:
        print("Sparse values column: present")

    # Show first row as a sample
    print(f"\nSample (first row):")
    for col in table.column_names:
        val = table.column(col)[0].as_py()
        if isinstance(val, list) and len(val) > 5:
            preview = f"[{val[0]}, {val[1]}, {val[2]}, ... ] ({len(val)} elements)"
        elif isinstance(val, str) and len(val) > 200:
            preview = val[:200] + "..."
        else:
            preview = val
        print(f"  {col}: {preview}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_parquet.py <path_to_parquet_file>")
        sys.exit(1)
    inspect_parquet(sys.argv[1])
