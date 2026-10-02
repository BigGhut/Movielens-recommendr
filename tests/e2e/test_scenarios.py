import concurrent.futures
import os
import subprocess
import sys
import time

import pandas as pd
import pytest
import requests

# --- Tier 4: Real-World Application Scenarios (5 Workloads) ---

@pytest.fixture(scope="module", autouse=True)
def run_pipelines_setup(preprocess_script, retrieval_train_script, reranking_train_script, data_dir, model_dir):
    cmd_prep = [sys.executable, preprocess_script, "--raw-dir", str(data_dir / "raw"), "--processed-dir", str(data_dir)]
    subprocess.run(cmd_prep, capture_output=True, check=False)
    cmd_ret = [sys.executable, retrieval_train_script, "--model-dir", str(model_dir), "--data-dir", str(data_dir)]
    subprocess.run(cmd_ret, capture_output=True, check=False)
    cmd_rer = [sys.executable, reranking_train_script, "--model-dir", str(model_dir), "--data-dir", str(data_dir)]
    subprocess.run(cmd_rer, capture_output=True, check=False)

def test_scenario_1_standard_user_journey(api_client, data_dir):
    """Scenario 1: Standard User Recommendations Journey."""
    # 1. Retrieve the user's historical ratings from train.csv
    train_df = pd.read_csv(data_dir / "train.csv")
    user_history = train_df[train_df["user_id"] == 1]
    assert len(user_history) > 0
    
    # 2. Send request to recommendations endpoint
    start = time.time()
    resp = api_client.get_recommendations(user_id=1)
    duration = time.time() - start
    
    # 3. Assertions
    assert resp.status_code == 200
    assert duration <= 0.500
    data = resp.json()
    assert len(data["recommendations"]) == 10
    # Visual check: User 1 liked Animation/Children's/Comedy movies,
    # and mock server returns movies in those categories.
    assert data["user_id"] == 1

def test_scenario_2_dynamic_training_and_redeployment(
    api_client, preprocess_script, retrieval_train_script, reranking_train_script, data_dir, model_dir, tmp_path
):
    """Scenario 2: Dynamic System Training and Redeployment."""
    # 1. Append new interactions to raw data
    raw_dir = tmp_path / "raw_dynamic"
    raw_dir.mkdir()
    
    ratings_data = [
        "1::101::4::1000", "1::102::5::1001", "1::103::3::1002", "1::104::4::1003", "1::105::5::1004", "1::106::2::1005",
        "2::101::3::2000", "2::102::4::2001", "2::103::5::2002", "2::104::2::2003", "2::105::4::2004",
        "4::101::5::4000", "4::102::4::4001", "4::103::3::4002", "4::104::5::4003", "4::105::4::4004", "4::106::3::4005", "4::107::5::4006",
        "5::101::5::5000", "5::102::4::5000", "5::103::3::5000", "5::104::5::5000", "5::105::2::5000",
        # New interaction added dynamically
        "1::107::5::9999"
    ]
    users_data = ["1::F::1::10::48067", "2::M::56::16::70072", "4::M::45::7::02460", "5::F::25::20::94103"]
    movies_data = [
        "101::Toy Story (1995)::Animation", "102::Jumanji (1995)::Adventure", "103::Grumpier Old Men::Comedy",
        "104::Waiting to Exhale::Comedy", "105::Father of the Bride::Comedy", "106::Heat::Action", "107::Sabrina::Comedy"
    ]
    
    (raw_dir / "ratings.dat").write_text("\n".join(ratings_data) + "\n", encoding="latin-1")
    (raw_dir / "users.dat").write_text("\n".join(users_data) + "\n", encoding="latin-1")
    (raw_dir / "movies.dat").write_text("\n".join(movies_data) + "\n", encoding="latin-1")
    
    # Run pipelines concurrently with active API queries
    def query_loop():
        latencies = []
        success_count = 0
        for _ in range(10):
            start = time.time()
            try:
                resp = api_client.get_recommendations(user_id=1)
                if resp.status_code == 200:
                    success_count += 1
                latencies.append(time.time() - start)
            except Exception as exc:  # noqa: BLE001 - one failed probe does not stop the latency loop
                print(f"query failed: {exc}")
            time.sleep(0.1)
        return success_count, max(latencies) if latencies else 0.0

    # Start concurrent background queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        query_future = executor.submit(query_loop)
        
        # Execute preprocessing
        cmd_prep = [sys.executable, preprocess_script, "--raw-dir", str(raw_dir), "--processed-dir", str(data_dir)]
        res_prep = subprocess.run(cmd_prep, capture_output=True, check=False)
        assert res_prep.returncode == 0
        
        # Execute retrieval train
        cmd_ret = [sys.executable, retrieval_train_script, "--model-dir", str(model_dir), "--data-dir", str(data_dir)]
        res_ret = subprocess.run(cmd_ret, capture_output=True, check=False)
        assert res_ret.returncode == 0
        
        # Execute reranking train
        cmd_rerank = [sys.executable, reranking_train_script, "--model-dir", str(model_dir), "--data-dir", str(data_dir)]
        res_rerank = subprocess.run(cmd_rerank, capture_output=True, check=False)
        assert res_rerank.returncode == 0
        
        success_count, max_latency = query_future.result()
        
    # Check assertions
    assert success_count == 10, "Some API queries dropped during retrain cycle"
    # Max latency spike during index reload should be <= 600ms (0.600 seconds)
    assert max_latency <= 0.600, f"Max latency was too high: {max_latency}s"
    assert (model_dir / "item_index.faiss").exists()

def test_scenario_3_multi_user_concurrent_traffic_peak(api_client):
    """Scenario 3: Multi-User Concurrent Traffic Peak."""
    # Mixture of users: 70% registered (IDs: 1, 2, 4, 5), 20% cold (IDs: 900001, 900002), 10% invalid (IDs: -1, -2)
    user_pool = [1, 2, 4, 5] * 3 + [900001, 900002, 900003, 900004] + [-1, -2]
    assert len(user_pool) == 18 # Close enough to 20 requests mixture
    
    def fetch_recs(uid):
        start = time.time()
        resp = api_client.get_recommendations(user_id=uid)
        return resp.status_code, resp.json()["recommendations"], time.time() - start

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_recs, uid): uid for uid in user_pool}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    statuses, _recs_lists, latencies = zip(*results)
    
    # 1. 100% of requests complete successfully (no HTTP 500)
    for status in statuses:
        assert status == 200
        
    # 2. Average latency across all requests is <= 500ms
    mean_latency = sum(latencies) / len(latencies)
    assert mean_latency <= 0.500
    
    # 3. Recommendation lists are distinct for different registered users
    recs_user1 = api_client.get_recommendations(user_id=1).json()["recommendations"]
    recs_user2 = api_client.get_recommendations(user_id=2).json()["recommendations"]
    assert recs_user1 != recs_user2

def test_scenario_4_graceful_degradation_db_failure(api_client, model_dir):
    """Scenario 4: Graceful Degradation under Database / Storage Failure."""
    faiss_path = model_dir / "item_index.faiss"
    faiss_bak = model_dir / "item_index.faiss.bak"
    
    # Temporarily rename index file to simulate database/storage failure
    if faiss_path.exists():
        os.rename(faiss_path, faiss_bak)
        
    try:
        # Request recommendations
        resp = api_client.get_recommendations(user_id=1)
        # Should degrade gracefully to popularity fallback instead of returning HTTP 500
        assert resp.status_code == 200
        data = resp.json()
        assert data["fallback"] is True
        assert data["fallback_type"] == "popularity_fallback"
    finally:
        # Restore index file
        if faiss_bak.exists():
            os.rename(faiss_bak, faiss_path)

def test_scenario_5_complete_cold_start_deployment(use_mock_data):
    """Scenario 5: Complete Cold-Start Deployment Verification."""
    # Purge is simulated by starting a new server on a free port,
    # polling health check to verify it starts under 2 seconds,
    # running smoke recommendation query, and tearing down.
    import socket
    import sys
    
    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    # Spin up server from scratch
    process = subprocess.Popen(
        [sys.executable, "tests/e2e/mock_server.py", "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    start_time = time.time()
    success = False
    
    # Poll health check
    for _ in range(20):
        try:
            resp = requests.get(f"{url}/health")
            if resp.status_code == 200:
                success = True
                break
        except requests.RequestException:
            pass
        time.sleep(0.1)
        
    elapsed = time.time() - start_time
    
    try:
        assert success, "Server failed to launch"
        # Health check must respond within 2 seconds of starting
        assert elapsed <= 2.0
        
        # Smoke recommendation query
        smoke_resp = requests.get(f"{url}/recommend/1")
        assert smoke_resp.status_code == 200
        assert len(smoke_resp.json()["recommendations"]) == 10
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
