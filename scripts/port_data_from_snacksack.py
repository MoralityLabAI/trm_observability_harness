import paramiko
import os
from pathlib import Path

def fetch_files():
    hostname = "snacksack-ms-7d32.tail3156cd.ts.net"
    username = "patrick"
    password = "heypyatt"
    
    local_data_dir = Path("data/ported_from_snacksack")
    local_data_dir.mkdir(parents=True, exist_ok=True)
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, username=username, password=password)
        print(f"Connected to {hostname}. Searching for Qwopus data...")
        
        # Find jsonl and log files modified in the last 24 hours
        # Focus on Qwopus and tesseract related paths
        commands = [
            "find /home/patrick/Qwopus -name '*.jsonl' -o -name '*.log' -mtime -1",
            "find /home/patrick/tesseract-jobs -name '*.jsonl' -o -name '*.log' -mtime -1",
            "find /home/patrick/logs -name '*.jsonl' -o -name '*.log' -mtime -1"
        ]
        
        all_files = []
        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            all_files.extend(stdout.read().decode().strip().split('\n'))
            
        all_files = [f for f in all_files if f.strip()]
        
        if not all_files:
            print("No recent data files found on snacksack.")
            return

        print(f"Found {len(all_files)} potential data files. Starting transfer...")
        
        sftp = client.open_sftp()
        for remote_path in all_files:
            try:
                rel_path = remote_path.replace("/home/patrick/", "").replace("/", "_")
                local_path = local_data_dir / rel_path
                print(f"Fetching {remote_path} -> {local_path}")
                sftp.get(remote_path, str(local_path))
            except Exception as e:
                print(f"  Error fetching {remote_path}: {e}")
        
        sftp.close()
        print("Transfer complete.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    fetch_files()
