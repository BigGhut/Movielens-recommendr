import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def use_mock_data(request):
    return request.config.getoption("--use-mock-data")

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

@pytest.fixture(scope="session")
def mock_server(data_dir, model_dir):
    port = find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["MODEL_DIR"] = str(model_dir)

    # A pipe fills up and deadlocks the server once uvicorn logs enough lines.
    with tempfile.TemporaryFile() as log_handle:
        process = subprocess.Popen(
            [sys.executable, "tests/e2e/mock_server.py", "--host", host, "--port", str(port)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        def server_log() -> str:
            log_handle.seek(0)
            return log_handle.read().decode(errors="replace")

        started = False
        for _ in range(30):
            try:
                response = requests.get(f"{url}/health", timeout=1)
                if response.status_code == 200:
                    started = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)
        if not started:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(f"Mock server failed to start on {url}.\n{server_log()}")

        try:
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

class MockAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_health(self, delay: float = 0.0):
        return self.session.get(f"{self.base_url}/health", params={"delay": delay})
        
    def get_recommendations(self, user_id: int, delay: float = 0.0):
        return self.session.get(f"{self.base_url}/recommend/{user_id}", params={"delay": delay})
        
    def get_movie(self, movie_id: int):
        return self.session.get(f"{self.base_url}/movie/{movie_id}")
        
    def get_item(self, item_id: int):
        return self.session.get(f"{self.base_url}/item/{item_id}")

@pytest.fixture(scope="session")
def api_client(mock_server):
    return MockAPIClient(mock_server)


@pytest.fixture(scope="session")
def data_dir(use_mock_data, tmp_path_factory):
    if use_mock_data:
        path = tmp_path_factory.mktemp("data")
        subprocess.run(
            [
                sys.executable,
                "tests/e2e/mock_preprocess.py",
                "--raw-dir",
                str(path / "raw"),
                "--processed-dir",
                str(path),
            ],
            check=True,
        )
        return path
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("data/processed")

@pytest.fixture(scope="session")
def model_dir(use_mock_data, tmp_path_factory):
    if use_mock_data:
        path = tmp_path_factory.mktemp("models")
        # The mock server treats missing files as a degraded catalog.
        # Present, empty files select the normal and cold-start branches.
        (path / "item_index.faiss").write_bytes(b"")
        (path / "reranker.lgb").write_bytes(b"")
        return path
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("models")

@pytest.fixture(scope="session")
def preprocess_script(use_mock_data):
    if use_mock_data:
        return "tests/e2e/mock_preprocess.py"
    return "src/data/preprocessing.py"

@pytest.fixture(scope="session")
def retrieval_train_script(use_mock_data):
    if use_mock_data:
        return "tests/e2e/mock_retrieval_train.py"
    return "src/retrieval/train.py"

@pytest.fixture(scope="session")
def reranking_train_script(use_mock_data):
    if use_mock_data:
        return "tests/e2e/mock_reranking_train.py"
    return "src/reranking/train.py"

@pytest.fixture(scope="session")
def evaluate_script(use_mock_data):
    if use_mock_data:
        return "tests/e2e/mock_evaluate.py"
    return "evaluate.py"

