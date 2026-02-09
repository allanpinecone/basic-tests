#!/usr/bin/env python3
"""Pinecone BYOC Backup Script with status monitoring."""

import getpass
import time
from datetime import datetime
from pinecone import Pinecone


def monitor_backup(pc, backup_id: str, poll_interval: int = 5):
    """Monitor backup status until completion."""
    print(f"\nMonitoring backup status (polling every {poll_interval}s)...")
    print("-" * 50)
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            backup = pc.describe_backup(backup_id=backup_id)
            status = backup.status
            elapsed = int(time.time() - start_time)
            
            # Only print if status changed or every 30 seconds
            if status != last_status or elapsed % 30 == 0:
                print(f"[{elapsed:4d}s] Status: {status}")
                last_status = status
            
            # Check for terminal states
            if status == "Ready":
                print("-" * 50)
                print(f"\nBackup completed successfully in {elapsed}s!")
                print(f"\nBackup details:")
                print(f"  ID: {backup.backup_id}")
                print(f"  Name: {backup.name}")
                print(f"  Status: {backup.status}")
                if hasattr(backup, 'record_count'):
                    print(f"  Records: {backup.record_count:,}")
                if hasattr(backup, 'size_bytes'):
                    print(f"  Size: {backup.size_bytes / (1024*1024):.2f} MB")
                if hasattr(backup, 'cloud'):
                    print(f"  Cloud: {backup.cloud}")
                if hasattr(backup, 'region'):
                    print(f"  Region: {backup.region}")
                return backup
            
            elif status in ("Failed", "Cancelled"):
                print("-" * 50)
                print(f"\nBackup {status.lower()}!")
                print(f"\n{backup}")
                return backup
            
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(poll_interval)


def list_backups(pc):
    """List all existing backups."""
    print("\nFetching existing backups...")
    try:
        backups = pc.list_backups()
        if not backups:
            print("No backups found.")
            return
        
        print(f"\nFound {len(backups)} backup(s):")
        print("-" * 80)
        for b in backups:
            print(f"  ID: {b.backup_id}")
            print(f"  Name: {b.name}")
            print(f"  Status: {b.status}")
            if hasattr(b, 'source_index_name'):
                print(f"  Source: {b.source_index_name}")
            print("-" * 80)
    except Exception as e:
        print(f"Error listing backups: {e}")


def main():
    print("=" * 60)
    print("PINECONE BACKUP")
    print("=" * 60)
    
    # Get API key
    api_key = getpass.getpass("Enter your Pinecone API key: ")
    
    # Initialize client
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Menu
    print("\nOptions:")
    print("1. Create new backup")
    print("2. List existing backups")
    print("3. Check backup status by ID")
    
    choice = input("\nSelect option [1]: ").strip() or "1"
    
    if choice == "2":
        list_backups(pc)
        return
    
    if choice == "3":
        backup_id = input("Enter backup ID: ").strip()
        if backup_id:
            monitor_backup(pc, backup_id)
        return
    
    # Create new backup (choice 1)
    # Get index name
    index_name = input("Index name: ").strip()
    if not index_name:
        print("Error: Index name is required.")
        return
    
    # Generate default backup name with timestamp
    default_backup_name = f"{index_name}-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    backup_name = input(f"Backup name [{default_backup_name}]: ").strip() or default_backup_name
    
    # Get optional description
    description = input("Description (optional): ").strip() or f"Backup of {index_name}"
    
    print(f"\nCreating backup...")
    print(f"  Index: {index_name}")
    print(f"  Backup name: {backup_name}")
    print(f"  Description: {description}")
    
    try:
        backup = pc.create_backup(
            index_name=index_name,
            backup_name=backup_name,
            description=description
        )
        
        print(f"\nBackup initiated!")
        print(f"  Backup ID: {backup.backup_id}")
        print(f"  Initial status: {backup.status}")
        
        # Monitor until complete
        monitor_backup(pc, backup.backup_id)
        
    except Exception as e:
        print(f"\nBackup failed: {e}")


if __name__ == "__main__":
    main()
