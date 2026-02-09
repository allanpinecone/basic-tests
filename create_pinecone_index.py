#!/usr/bin/env python3
"""Create a test Pinecone index with a single 1024-dimension vector (BYOC)."""

import getpass
import time
import socket
import urllib.request
import ssl
from pinecone import Pinecone


# BYOC private host
PINECONE_HOST = "https://byoc-test2-bjxtu6k.svc.private.preprod-aws-us-east-1-25dc.byoc.pinecone.io"
HOSTNAME = "byoc-test2-bjxtu6k.svc.private.preprod-aws-us-east-1-25dc.byoc.pinecone.io"


def test_connectivity():
    """Test basic network connectivity to the host."""
    print(f"\n=== Connectivity Tests ===")
    
    # DNS resolution
    print(f"\n1. DNS Resolution for {HOSTNAME}...")
    try:
        ips = socket.gethostbyname_ex(HOSTNAME)
        print(f"   Resolved to: {ips[2]}")
    except socket.gaierror as e:
        print(f"   FAILED: {e}")
        return False
    
    # TCP connection on port 443
    print(f"\n2. TCP connection to {HOSTNAME}:443...")
    try:
        sock = socket.create_connection((HOSTNAME, 443), timeout=10)
        sock.close()
        print("   SUCCESS: Port 443 is reachable")
    except (socket.timeout, socket.error) as e:
        print(f"   FAILED: {e}")
        return False
    
    # HTTPS connection
    print(f"\n3. HTTPS connection to {PINECONE_HOST}...")
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(PINECONE_HOST, method='HEAD')
        urllib.request.urlopen(req, timeout=10, context=ctx)
        print("   SUCCESS: HTTPS connection works")
    except urllib.error.HTTPError as e:
        # HTTP errors (401, 403, etc.) mean we connected successfully
        print(f"   SUCCESS: Got HTTP {e.code} (connection works, auth expected)")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    print("\n=== All connectivity tests passed ===\n")
    return True


def main():
    # Test connectivity first
    if not test_connectivity():
        print("\nConnectivity tests failed. Please check network/firewall settings.")
        return
    
    # Prompt for API key (hidden input)
    api_key = getpass.getpass("Enter your Pinecone API key: ")
    
    # Initialize Pinecone client
    print("\nInitializing Pinecone client...")
    pc = Pinecone(api_key=api_key)
    
    print(f"Connecting to BYOC host: {PINECONE_HOST}")
    
    # Connect to the index using the private host
    index = pc.Index(host=PINECONE_HOST)
    print("Index object created.")
    
    # Test with describe_index_stats first (lighter operation)
    print("\nTesting connection with describe_index_stats()...")
    try:
        start = time.time()
        stats = index.describe_index_stats()
        elapsed = time.time() - start
        print(f"   SUCCESS ({elapsed:.2f}s): {stats}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return
    
    # Create a single test vector (1024 dimensions)
    test_vector = [0.1] * 1024  # Simple test vector with all values set to 0.1
    
    # Upsert the vector
    print("\nUpserting test vector...")
    print(f"   Vector ID: test-vector-1")
    print(f"   Dimensions: 1024")
    try:
        start = time.time()
        result = index.upsert(
            vectors=[
                {
                    "id": "test-vector-1",
                    "values": test_vector,
                    "metadata": {"description": "Test vector for demonstration"}
                }
            ]
        )
        elapsed = time.time() - start
        print(f"   SUCCESS ({elapsed:.2f}s): {result}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return
    
    # Verify the vector was inserted
    print("\nWaiting 2s for upsert to propagate...")
    time.sleep(2)
    
    print("Fetching final index stats...")
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print("\nSuccessfully upserted 1 vector to BYOC index!")


if __name__ == "__main__":
    main()
