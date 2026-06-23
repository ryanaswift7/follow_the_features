import time
import requests
import subprocess

class LocalVllmServer:
    """Spins up vLLM serve via subprocess and gracefully kills it when done."""
    
    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct", port=8000, tp=2):
        self.port = port
        self.cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--max-model-len", "8192"
        ]
        self.process = None

    def __enter__(self):
        print(f"\nStarting local vLLM judge server on port {self.port} with TP={self.cmd[self.cmd.index('--tensor-parallel-size')+1]}...")
        self.process = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        health_url = f"http://localhost:{self.port}/health"
        while True:
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    print("vLLM Server is online and ready!")
                    break
            except requests.exceptions.ConnectionError:
                pass
            
            if self.process.poll() is not None:
                raise RuntimeError("vLLM server process crashed during startup.")
            time.sleep(5)
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Shutting down local vLLM judge server and releasing VRAM...")
        self.process.terminate()
        self.process.wait()