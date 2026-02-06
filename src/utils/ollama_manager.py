import os
import subprocess
import time

import requests


class OllamaManager:
    def __init__(self, ollama_judge, port="11434"):
        self.port = port
        self.host = f"127.0.0.1:{self.port}"
        self.process = None
        self.ollama_judge = ollama_judge

    def __enter__(self):
        print(f"--- Starting Ollama on port {self.port} ---")
        # Define the environment for the subprocess
        env = os.environ.copy()
        env["OLLAMA_HOST"] = self.host

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,  # This forces Ollama to use your custom port
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        # Wait for the server to wake up on the custom port
        for i in range(10):
            try:
                response = requests.get(f"http://{self.host}/api/tags")
                if response.status_code == 200:
                    return self
            except requests.exceptions.ConnectionError:
                time.sleep(2)
        raise RuntimeError(f"Ollama failed on {self.host}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            print(f"--- Cleaning up VRAM and stopping Ollama on port {self.port} ---")
            try:
                # 1. Force the model to unload from VRAM immediately
                # We tell Ollama to set the keep_alive to 0, which triggers a VRAM flush.
                unload_url = f"http://127.0.0.1:{self.port}/api/generate"
                try:
                    requests.post(
                        unload_url,
                        json={"model": self.ollama_judge, "keep_alive": 0},
                        timeout=2,
                    )
                except Exception:
                    # If the server is already hanging, just skip to killing it
                    pass

                # 2. Kill the process
                # On Linux/Mac, we use killpg to ensure the GPU 'runner' child dies too.
                # On Windows, we use the process handle.
                if os.name != "nt":
                    import signal

                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()

                # 3. Wait for the OS to release the resources
                self.process.wait(timeout=5)

                # 4. Optional: Force a CUDA cache clear in Python
                # Only relevant if your main script is also using the GPU
                import torch

                torch.cuda.empty_cache()

                print("VRAM cleared and process terminated.")

            except Exception as e:
                print(f"Cleanup failed: {e}")
