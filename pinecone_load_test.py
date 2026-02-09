#!/usr/bin/env python3
"""Pinecone BYOC Load Testing Script - Multi-threaded writes and reads."""

import getpass
import time
import random
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone


# Configuration
VECTOR_DIMENSION = 1024
BATCH_SIZE = 100  # Pinecone recommended batch size
DEFAULT_WRITE_THREADS = 10
DEFAULT_READ_THREADS = 20


class LoadTestMetrics:
    """Thread-safe metrics collector."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.operation_count = 0
        self.error_count = 0
        self.latencies = []
        self.start_time = None
        self.end_time = None
    
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
        
        return {
            "operations": self.operation_count,
            "errors": self.error_count,
            "elapsed_sec": round(elapsed, 2),
            "ops_per_sec": round(self.operation_count / elapsed, 2) if elapsed > 0 else 0,
            "avg_latency_ms": round(statistics.mean(self.latencies), 2),
            "p50_latency_ms": round(statistics.median(self.latencies), 2),
            "p95_latency_ms": round(sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0, 2),
            "p99_latency_ms": round(sorted(self.latencies)[int(len(self.latencies) * 0.99)] if self.latencies else 0, 2),
            "min_latency_ms": round(min(self.latencies), 2),
            "max_latency_ms": round(max(self.latencies), 2),
        }


def generate_random_vector(dimension: int = VECTOR_DIMENSION) -> list:
    """Generate a random vector with values between -1 and 1."""
    return [random.uniform(-1, 1) for _ in range(dimension)]


def generate_vector_batch(start_id: int, count: int) -> list:
    """Generate a batch of vectors with sequential IDs."""
    return [
        {
            "id": f"vec-{start_id + i}",
            "values": generate_random_vector(),
            "metadata": {"batch_id": start_id // BATCH_SIZE}
        }
        for i in range(count)
    ]


def upsert_batch(index, vectors: list, metrics: LoadTestMetrics):
    """Upsert a batch of vectors and record metrics."""
    try:
        start = time.time()
        index.upsert(vectors=vectors)
        latency_ms = (time.time() - start) * 1000
        metrics.record(latency_ms, success=True)
        return len(vectors)
    except Exception as e:
        metrics.record(0, success=False)
        print(f"   Upsert error: {e}")
        return 0


def query_random(index, metrics: LoadTestMetrics, stop_event: threading.Event):
    """Continuously query with random vectors until stop_event is set."""
    while not stop_event.is_set():
        try:
            query_vector = generate_random_vector()
            start = time.time()
            index.query(vector=query_vector, top_k=10)
            latency_ms = (time.time() - start) * 1000
            metrics.record(latency_ms, success=True)
        except Exception as e:
            metrics.record(0, success=False)
            if not stop_event.is_set():
                print(f"   Query error: {e}")


def run_write_load_test(index, num_vectors: int, num_threads: int = DEFAULT_WRITE_THREADS):
    """Run write load test with multi-threaded batch upserts."""
    print(f"\n{'='*60}")
    print(f"WRITE LOAD TEST")
    print(f"{'='*60}")
    print(f"Vectors to upsert: {num_vectors:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Threads: {num_threads}")
    print(f"Vector dimension: {VECTOR_DIMENSION}")
    
    metrics = LoadTestMetrics()
    total_batches = (num_vectors + BATCH_SIZE - 1) // BATCH_SIZE
    vectors_upserted = 0
    
    print(f"\nGenerating and upserting {total_batches} batches...")
    metrics.start()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for batch_num in range(total_batches):
            start_id = batch_num * BATCH_SIZE
            count = min(BATCH_SIZE, num_vectors - start_id)
            vectors = generate_vector_batch(start_id, count)
            futures.append(executor.submit(upsert_batch, index, vectors, metrics))
            
            # Progress update every 10 batches
            if (batch_num + 1) % 10 == 0:
                print(f"   Submitted {batch_num + 1}/{total_batches} batches...")
        
        # Wait for all to complete
        for future in as_completed(futures):
            vectors_upserted += future.result()
    
    metrics.stop()
    
    print(f"\n--- Write Results ---")
    summary = metrics.summary()
    print(f"Vectors upserted: {vectors_upserted:,}")
    print(f"Batches completed: {summary['operations']}")
    print(f"Errors: {summary['errors']}")
    print(f"Total time: {summary['elapsed_sec']}s")
    print(f"Throughput: {round(vectors_upserted / summary['elapsed_sec'], 2)} vectors/sec")
    print(f"Batch throughput: {summary['ops_per_sec']} batches/sec")
    print(f"Avg batch latency: {summary['avg_latency_ms']}ms")
    print(f"P50 latency: {summary['p50_latency_ms']}ms")
    print(f"P95 latency: {summary['p95_latency_ms']}ms")
    print(f"P99 latency: {summary['p99_latency_ms']}ms")
    
    return vectors_upserted


def run_read_load_test(index, duration_seconds: int, num_threads: int = DEFAULT_READ_THREADS):
    """Run read load test with multi-threaded random queries."""
    print(f"\n{'='*60}")
    print(f"READ LOAD TEST")
    print(f"{'='*60}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Threads: {num_threads}")
    print(f"Query type: Random vector, top_k=10")
    
    metrics = LoadTestMetrics()
    stop_event = threading.Event()
    
    print(f"\nRunning queries for {duration_seconds}s...")
    metrics.start()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Start all query threads
        futures = [
            executor.submit(query_random, index, metrics, stop_event)
            for _ in range(num_threads)
        ]
        
        # Wait for duration, showing progress
        start = time.time()
        while time.time() - start < duration_seconds:
            elapsed = int(time.time() - start)
            remaining = duration_seconds - elapsed
            current_ops = metrics.operation_count
            print(f"   {remaining}s remaining... ({current_ops:,} queries so far)", end='\r')
            time.sleep(1)
        
        # Signal threads to stop
        stop_event.set()
        print(f"\n   Stopping threads...")
    
    metrics.stop()
    
    print(f"\n--- Read Results ---")
    summary = metrics.summary()
    print(f"Total queries: {summary['operations']:,}")
    print(f"Errors: {summary['errors']}")
    print(f"Total time: {summary['elapsed_sec']}s")
    print(f"Throughput: {summary['ops_per_sec']} queries/sec")
    print(f"Avg latency: {summary['avg_latency_ms']}ms")
    print(f"P50 latency: {summary['p50_latency_ms']}ms")
    print(f"P95 latency: {summary['p95_latency_ms']}ms")
    print(f"P99 latency: {summary['p99_latency_ms']}ms")
    print(f"Min latency: {summary['min_latency_ms']}ms")
    print(f"Max latency: {summary['max_latency_ms']}ms")


def delete_all_vectors(index):
    """Delete all vectors from the index."""
    print(f"\n{'='*60}")
    print(f"DELETING ALL VECTORS")
    print(f"{'='*60}")
    
    try:
        # Get current stats
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        print(f"Current vector count: {total_vectors:,}")
        
        if total_vectors == 0:
            print("No vectors to delete.")
            return
        
        print("Deleting all vectors...")
        start = time.time()
        index.delete(delete_all=True)
        elapsed = time.time() - start
        
        print(f"Delete completed in {elapsed:.2f}s")
        
        # Verify deletion
        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"Vectors remaining: {stats.total_vector_count:,}")
        
    except Exception as e:
        print(f"Delete error: {e}")


def show_index_stats(index):
    """Display current index statistics."""
    print(f"\n--- Index Stats ---")
    try:
        stats = index.describe_index_stats()
        print(f"Total vectors: {stats.total_vector_count:,}")
        print(f"Dimension: {stats.dimension}")
        if stats.namespaces:
            print(f"Namespaces: {dict(stats.namespaces)}")
    except Exception as e:
        print(f"Error getting stats: {e}")


def prompt_int(message: str, default: int = None) -> int:
    """Prompt for an integer with optional default."""
    while True:
        try:
            prompt = f"{message}"
            if default is not None:
                prompt += f" [{default}]"
            prompt += ": "
            value = input(prompt).strip()
            if not value and default is not None:
                return default
            return int(value)
        except ValueError:
            print("Please enter a valid number.")


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Prompt for yes/no with default."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{message} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


def main_menu():
    """Display main menu and get selection."""
    print(f"\n{'='*60}")
    print("PINECONE LOAD TESTER - MAIN MENU")
    print(f"{'='*60}")
    print("1. Full test (write + read + optional delete)")
    print("2. Write only (upsert vectors)")
    print("3. Read only (query existing vectors)")
    print("4. Delete all vectors")
    print("5. Show index stats")
    print("6. Exit")
    print()
    
    while True:
        try:
            choice = int(input("Select option [1-6]: "))
            if 1 <= choice <= 6:
                return choice
        except ValueError:
            pass
        print("Please enter a number 1-6.")


def main():
    print(f"\n{'='*60}")
    print("PINECONE BYOC LOAD TESTER")
    print(f"{'='*60}")
    
    # Get Pinecone host URL
    pinecone_host = input("\nEnter your Pinecone host URL: ").strip()
    if not pinecone_host:
        print("Error: Pinecone host URL is required.")
        return
    
    print(f"Host: {pinecone_host}")
    
    # Get API key
    api_key = getpass.getpass("\nEnter your Pinecone API key: ")
    
    # Initialize client
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=pinecone_host)
    
    # Verify connection
    try:
        stats = index.describe_index_stats()
        print(f"Connected! Current vector count: {stats.total_vector_count:,}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    while True:
        choice = main_menu()
        
        if choice == 1:  # Full test
            num_vectors = prompt_int("Number of vectors to generate", 10000)
            write_threads = prompt_int("Number of write threads", DEFAULT_WRITE_THREADS)
            
            run_write_load_test(index, num_vectors, write_threads)
            
            # Wait for vectors to be indexed
            print("\nWaiting 5s for vectors to be indexed...")
            time.sleep(5)
            show_index_stats(index)
            
            read_duration = prompt_int("Seconds to run read test", 30)
            read_threads = prompt_int("Number of read threads", DEFAULT_READ_THREADS)
            
            run_read_load_test(index, read_duration, read_threads)
            
            if prompt_yes_no("\nDelete all vectors?", default=False):
                delete_all_vectors(index)
            else:
                print("Vectors preserved. You can run read tests again later.")
        
        elif choice == 2:  # Write only
            num_vectors = prompt_int("Number of vectors to generate", 10000)
            write_threads = prompt_int("Number of write threads", DEFAULT_WRITE_THREADS)
            run_write_load_test(index, num_vectors, write_threads)
            time.sleep(2)
            show_index_stats(index)
        
        elif choice == 3:  # Read only
            show_index_stats(index)
            read_duration = prompt_int("Seconds to run read test", 30)
            read_threads = prompt_int("Number of read threads", DEFAULT_READ_THREADS)
            run_read_load_test(index, read_duration, read_threads)
        
        elif choice == 4:  # Delete
            if prompt_yes_no("Are you sure you want to delete ALL vectors?", default=False):
                delete_all_vectors(index)
        
        elif choice == 5:  # Stats
            show_index_stats(index)
        
        elif choice == 6:  # Exit
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
