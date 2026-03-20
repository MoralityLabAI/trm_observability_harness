import paramiko
import sys
import time

def start_server():
    hostname = "snacksack-ms-7d32.tail3156cd.ts.net"
    username = "patrick"
    password = "heypyatt"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, username=username, password=password)
        print("Connected.")
        
        work_dir = "/home/patrick/Qwopus/runtime/textgen-portable-4.1.1-linux-cuda12.4/text-generation-webui-4.1.1"
        python_exe = work_dir + "/portable_env/bin/python3"
        server_py = work_dir + "/server.py"
        model_path = "/home/patrick/Qwopus/models/Qwen3.5-27B.Q4_K_M.gguf"
        model_filename = "Qwen3.5-27B.Q4_K_M.gguf"
        
        # Ensure the link exists in the EXPECTED relative path location
        client.exec_command(f"mkdir -p {work_dir}/user_data/models")
        client.exec_command(f"ln -sf {model_path} {work_dir}/user_data/models/{model_filename}")

        # Use absolute paths for everything in the command to be safe
        # The key might be that it MUST be run from work_dir for other modules to load
        command = f"cd {work_dir} && {python_exe} {server_py} --portable --api --auto-launch --model {model_filename} --loader llama.cpp --ctx-size 16384 --gpu-layers 99 --cache-type q4_0 --api --api-port 8080 --nowebui --listen --listen-port 8081 --listen-host 0.0.0.0"

        print("Killing any existing processes...")
        client.exec_command("pkill -9 -f server.py")
        time.sleep(2)

        print(f"Starting server in {work_dir}...")
        log_path = "/home/patrick/Qwopus/server.log"
        # Using a subshell to ensure the cd and command run together correctly in the background
        full_cmd = f"nohup bash -c '{command}' > {log_path} 2>&1 &"
        client.exec_command(full_cmd)
        
        print("Waiting for server to initialize...")
        max_wait = 180
        start_time = time.time()
        while time.time() - start_time < max_wait:
            time.sleep(15)
            
            stdin, stdout, stderr = client.exec_command(f"cat {log_path}")
            log_out = stdout.read().decode()
            
            if "HTTP server listening" in log_out or "Model loaded" in log_out:
                stdin, stdout, stderr = client.exec_command("netstat -tulpn | grep 8080")
                if "8080" in stdout.read().decode():
                    print("Server is listening on port 8080.")
                    return True
                else:
                    print(f"Log shows activity (Elapsed: {int(time.time() - start_time)}s), waiting for port bind...")
            
            stdin, stdout, stderr = client.exec_command("ps aux | grep server.py | grep -v grep")
            ps_out = stdout.read().decode()
            if "server.py" in ps_out:
                print(f"Process is running... (Elapsed: {int(time.time() - start_time)}s)")
                lines = log_out.strip().split('\n')
                if lines:
                    print(f"Last log line: {lines[-1]}")
            else:
                print("Process failed. Log content:")
                print(log_out)
                return False
                
        print("Timed out.")
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    if start_server():
        sys.exit(0)
    else:
        sys.exit(1)
