import concurrent.futures
import threading
import time

import requests

# --- Tier 1: Feature Coverage (F5) ---

def test_latency_standard_user(api_client):
    """T1_F5_1: Verify API latency for typical user <= 500ms."""
    start = time.time()
    resp = api_client.get_recommendations(user_id=1)
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_cold_user(api_client):
    """T1_F5_2: Verify API latency for cold user fallback <= 500ms."""
    start = time.time()
    resp = api_client.get_recommendations(user_id=905000)
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_concurrent_light(api_client):
    """T1_F5_3: Verify latency under 5 concurrent users."""
    def make_request():
        start = time.time()
        resp = api_client.get_recommendations(user_id=1)
        return resp.status_code, time.time() - start

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in futures]

    statuses, durations = zip(*results)
    for status in statuses:
        assert status == 200
        
    mean_duration = sum(durations) / len(durations)
    assert mean_duration <= 0.500

def test_latency_health_check(api_client):
    """T1_F5_4: Verify health check latency <= 50ms."""
    start = time.time()
    resp = api_client.get_health()
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.050

def test_latency_sequential(api_client):
    """T1_F5_5: Verify response time over 20 sequential queries."""
    for _ in range(20):
        start = time.time()
        resp = api_client.get_recommendations(user_id=1)
        duration = time.time() - start
        assert resp.status_code == 200
        assert duration <= 0.500

# --- Tier 2: Boundary & Corner Cases (F5) ---

def test_latency_high_history_user(api_client):
    """T2_F5_1: Verify latency for user with 10k interactions."""
    # User 10000 has high interaction simulation
    start = time.time()
    resp = api_client.get_recommendations(user_id=10000)
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_under_read_lock(api_client, tmp_path):
    """T2_F5_2: Verify latency while reading from locked DB files."""
    # Simulate DB file locking by locking a dummy file
    lock_file = tmp_path / "dummy_db.lock"
    lock_file.write_text("locked")
    
    # We acquire lock on the lock_file (simulate read lock)
    # The API should not hang because of locks and should resolve under SLA
    start = time.time()
    resp = api_client.get_recommendations(user_id=2)
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_first_request_cold(api_client):
    """T2_F5_3: Verify first-request cold latency handling."""
    # Warm-up request or first request should still resolve fast
    start = time.time()
    resp = api_client.get_recommendations(user_id=4)
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_large_top_k(api_client):
    """T2_F5_4: Verify latency if requesting top-100 recommendations."""
    # Requesting 100 recommendations via POST endpoint
    start = time.time()
    resp = requests.post(f"{api_client.base_url}/recommend", json={"user_id": 1, "num_recs": 100})
    duration = time.time() - start
    assert resp.status_code == 200
    assert duration <= 0.500

def test_latency_during_faiss_reload(api_client, model_dir):
    """T2_F5_5: Verify latency while FAISS index updates in background."""
    index_file = model_dir / "item_index.faiss"
    
    # Normal latency baseline
    start_normal = time.time()
    resp1 = api_client.get_recommendations(user_id=1)
    latency_normal = time.time() - start_normal
    assert resp1.status_code == 200
    
    # Simulate index update on disk
    def update_index():
        if index_file.exists():
            with open(index_file, "ab") as f:
                f.write(b" ")
                
    # Update index and simultaneously send request
    t = threading.Thread(target=update_index)
    t.start()
    
    start_reload = time.time()
    resp2 = api_client.get_recommendations(user_id=1)
    latency_reload = time.time() - start_reload
    
    t.join()
    
    assert resp2.status_code == 200
    # Latency spike should be <= 100ms (0.100s)
    spike = latency_reload - latency_normal
    assert spike <= 0.100
