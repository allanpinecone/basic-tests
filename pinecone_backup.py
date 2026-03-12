#!/usr/bin/env python3
"""Pinecone Backup — interactive script for creating and managing index backups."""

from __future__ import annotations

import getpass
import os
import sys
import time
from datetime import datetime
from pinecone import Pinecone


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


# ---------------------------------------------------------------------------
# Index listing & picker
# ---------------------------------------------------------------------------

def list_and_pick_index(pc: Pinecone) -> tuple[str | None, str | None]:
    """List available indexes and let the user pick one. Returns (name, host)."""
    print("\nFetching indexes...")
    try:
        indexes = list(pc.list_indexes())
    except Exception as e:
        print(f"  Could not list indexes: {e}")
        return None, None

    if not indexes:
        print("  No indexes found in this project.")
        return None, None

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

    selection = input(f"Pick an index [1-{len(indexes)}]: ").strip()
    if not selection:
        return None, None
    try:
        sel_idx = int(selection) - 1
        if sel_idx < 0 or sel_idx >= len(indexes):
            print("Invalid selection.")
            return None, None
        chosen = indexes[sel_idx]
        name = getattr(chosen, "name", str(chosen))
        host = getattr(chosen, "host", None)
        return name, host
    except ValueError:
        print("Invalid selection.")
        return None, None


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_connection(index) -> bool:
    """Quick connectivity check via describe_index_stats."""
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
# Backup operations
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int | float) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def _print_backup_details(b):
    """Print formatted details for a single backup object."""
    print(f"  ID:     {b.backup_id}")
    print(f"  Name:   {b.name}")
    print(f"  Status: {b.status}")
    if hasattr(b, "source_index_name") and b.source_index_name:
        print(f"  Source: {b.source_index_name}")
    if hasattr(b, "description") and b.description:
        print(f"  Desc:   {b.description}")
    if hasattr(b, "record_count") and b.record_count is not None:
        print(f"  Records: {b.record_count:,}")
    if hasattr(b, "size_bytes") and b.size_bytes is not None:
        print(f"  Size:   {_format_size(b.size_bytes)}")
    if hasattr(b, "cloud") and b.cloud:
        print(f"  Cloud:  {b.cloud}")
    if hasattr(b, "region") and b.region:
        print(f"  Region: {b.region}")
    if hasattr(b, "created_at") and b.created_at:
        print(f"  Created: {b.created_at}")


def monitor_backup(pc, backup_id: str, poll_interval: int = 5):
    """Monitor backup status until completion. Ctrl+C returns to menu."""
    print(f"\nMonitoring backup '{backup_id}' (polling every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring and return to menu.")
    print("-" * 60)

    start_time = time.time()
    last_status = None

    try:
        while True:
            try:
                backup = pc.describe_backup(backup_id=backup_id)
                status = backup.status
                elapsed = int(time.time() - start_time)

                if status != last_status or elapsed % 30 == 0:
                    print(f"[{elapsed:5d}s]  status: {status}")
                    last_status = status

                if status == "Ready":
                    print("-" * 60)
                    print(f"\nBackup completed successfully in {elapsed}s!")
                    _print_backup_details(backup)
                    return backup

                elif status in ("Failed", "Cancelled"):
                    print("-" * 60)
                    print(f"\nBackup {status.lower()}.")
                    _print_backup_details(backup)
                    return backup

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  Warning: error checking status: {e}")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Backup '{backup_id}' may still be in progress.")
        print("Use option 3 to check its status later.")
        return None


def create_backup(pc, index_name: str):
    """Create a new backup for the connected index."""
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

        monitor = input("\n  Monitor progress? [Y/n]: ").strip().lower() or "y"
        if monitor.startswith("y"):
            monitor_backup(pc, backup.backup_id)

    except Exception as e:
        print(f"\n  Backup failed: {e}")


def list_backups(pc):
    """List all existing backups."""
    print("\nFetching backups...")
    try:
        backups = list(pc.list_backups())
    except Exception as e:
        print(f"  Error listing backups: {e}")
        return

    if not backups:
        print("  No backups found.")
        return

    print(f"\nFound {len(backups)} backup(s):")
    print("-" * 70)
    for i, b in enumerate(backups, 1):
        status_icon = {"Ready": "+", "Failed": "!", "InProgress": "~"}.get(b.status, "?")
        print(f"  {i}. [{status_icon}] {b.name}")
        _print_backup_details(b)
        print("-" * 70)


def describe_backup(pc):
    """Look up a specific backup by ID."""
    backup_id = input("  Enter backup ID: ").strip()
    if not backup_id:
        print("  No ID provided.")
        return

    print(f"\n  Fetching backup '{backup_id}'...")
    try:
        backup = pc.describe_backup(backup_id=backup_id)
        print()
        _print_backup_details(backup)
    except Exception as e:
        print(f"  Error: {e}")


def delete_backup(pc):
    """Delete a backup, with optional listing to pick from."""
    print("\nFetching backups...")
    try:
        backups = list(pc.list_backups())
    except Exception as e:
        print(f"  Error listing backups: {e}")
        return

    if not backups:
        print("  No backups found.")
        return

    print(f"\nBackups ({len(backups)}):")
    print("-" * 70)
    for i, b in enumerate(backups, 1):
        status_icon = {"Ready": "+", "Failed": "!", "InProgress": "~"}.get(b.status, "?")
        source = getattr(b, "source_index_name", "?")
        print(f"  {i}. [{status_icon}] {b.name}  (source: {source}, status: {b.status})")
    print("-" * 70)

    selection = input(f"  Pick a backup to delete [1-{len(backups)}] or Enter to cancel: ").strip()
    if not selection:
        print("  Cancelled.")
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

    confirm = input(f"\n  Type 'yes' to confirm deletion: ").strip()
    if confirm.lower() != "yes":
        print("  Cancelled.")
        return

    try:
        pc.delete_backup(backup_id=chosen.backup_id)
        print(f"  Backup '{chosen.name}' deleted.")
    except Exception as e:
        print(f"  Error deleting backup: {e}")


def describe_index_stats(index):
    """Show stats for the connected index."""
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


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

def setup_wizard():
    """Interactive setup: collect API key, pick an index, test connection."""
    print("=" * 60)
    print("  PINECONE BACKUP")
    print("  Create, monitor, and manage index backups.")
    print("=" * 60)

    # 1. API key
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

    # 2. Pick an index
    print("\nHow would you like to specify the index?")
    print("  1. List my indexes and pick one")
    print("  2. Enter the index name manually")

    host_choice = input("\nChoice [1]: ").strip() or "1"

    index_name = None
    index_host = None

    if host_choice == "1":
        index_name, index_host = list_and_pick_index(pc)

    if not index_name:
        index_name = input("Index name: ").strip()
        if not index_name:
            print("Error: Index name is required.")
            sys.exit(1)

    # 3. Connect and test
    if index_host:
        print(f"\nConnecting to '{index_name}'...")
        index = pc.Index(host=index_host)
    else:
        print(f"\nConnecting to '{index_name}'...")
        index = pc.Index(name=index_name)

    if not test_connection(index):
        proceed = input("Connection test failed. Continue anyway? [y/N]: ").strip().lower()
        if not proceed.startswith("y"):
            sys.exit(1)

    return pc, index, index_name


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    pc, index, index_name = setup_wizard()

    while True:
        print(f"\n{'=' * 60}")
        print(f"  Index: {index_name}")
        print(f"{'=' * 60}")
        print("  1. Create new backup")
        print("  2. List all backups")
        print("  3. Check backup status by ID")
        print("  4. Delete a backup")
        print("  5. Describe index stats")
        print("  6. Switch index")
        print("  q. Quit")

        choice = input("\nSelect option [1]: ").strip() or "1"

        if choice in ("q", "Q"):
            print("Bye!")
            break

        elif choice == "1":
            create_backup(pc, index_name)

        elif choice == "2":
            list_backups(pc)

        elif choice == "3":
            backup_id = input("  Enter backup ID: ").strip()
            if backup_id:
                monitor_backup(pc, backup_id)

        elif choice == "4":
            delete_backup(pc)

        elif choice == "5":
            describe_index_stats(index)

        elif choice == "6":
            print("\nHow would you like to specify the new index?")
            print("  1. List my indexes and pick one")
            print("  2. Enter the index name manually")
            hc = input("\nChoice [1]: ").strip() or "1"

            new_name = None
            new_host = None
            if hc == "1":
                new_name, new_host = list_and_pick_index(pc)
            if not new_name:
                new_name = input("Index name: ").strip()

            if new_name:
                index_name = new_name
                if new_host:
                    index = pc.Index(host=new_host)
                else:
                    index = pc.Index(name=index_name)
                print(f"\nConnecting to '{index_name}'...")
                test_connection(index)

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
