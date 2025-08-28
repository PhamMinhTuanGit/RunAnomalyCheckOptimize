import subprocess

def check_docker_compose_status(compose_file_path):
    try:
        # Build và chạy container
        print("Building and starting containers...")
        subprocess.run(
            ["docker-compose", "-f", compose_file_path, "up", "-d", "--build"],
            check=True
        )
        
        # Kiểm tra trạng thái container
        print("Checking container status...")
        result = subprocess.run(
            ["docker-compose", "-f", compose_file_path, "ps"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("Docker Compose Status:\n")
        print(result.stdout)
        
        # Optional: kiểm tra logs (ví dụ 10 dòng)
        print("Showing last 10 lines of logs...")
        logs = subprocess.run(
            ["docker-compose", "-f", compose_file_path, "logs", "--tail", "10"],
            check=True,
            capture_output=True,
            text=True
        )
        print(logs.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)

if __name__ == "__main__":
    compose_path = "feature_repo/docker-compose.yaml"  # Đường dẫn tới file docker-compose.yaml của bạn
    check_docker_compose_status(compose_path)
