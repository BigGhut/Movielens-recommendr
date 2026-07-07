import socket
import subprocess
import sys
import time
import os
from pathlib import Path
import pytest
import requests

def pytest_addoption(parser):
    parser.addoption(
        "--use-mock-data",
        action="store_true",
        default=False,
        help="Use mock/synthetic data for fast local testing",
    )

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
    
    import os
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["MODEL_DIR"] = str(model_dir)
    
    # Start the mock server in a background process
    process = subprocess.Popen(
        [sys.executable, "tests/e2e/mock_server.py", "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    
    # Wait for the server to start up by polling the /health endpoint
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.1)
    else:
        # If it failed to start, terminate and print output
        process.terminate()
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"Mock server failed to start on {url}.\n"
            f"STDOUT: {stdout.decode()}\n"
            f"STDERR: {stderr.decode()}"
        )
        
    yield url
    
    # Terminate the process on teardown
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
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if use_mock_data:
        return tmp_path_factory.mktemp("data")
    return Path("data/processed")

@pytest.fixture(scope="session")
def model_dir(use_mock_data, tmp_path_factory):
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    if use_mock_data:
        return tmp_path_factory.mktemp("models")
    return Path("models")

@pytest.fixture(scope="session")
def preprocess_script(use_mock_data):
    if use_mock_data:
        return "tests/e2e/mock_preprocess.py"
    return "src/data/preprocess.py"

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

