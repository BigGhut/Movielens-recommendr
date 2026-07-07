import sys
import json
import subprocess
import pytest

def test_health_check(api_client):
    response = api_client.get_health()
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_recommend_regular_users(api_client):
    # Test recommendations for user 1
    resp1 = api_client.get_recommendations(user_id=1)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["user_id"] == 1
    assert len(data1["recommendations"]) == 10
    assert data1["fallback"] is False
    
    # Test recommendations for user 2
    resp2 = api_client.get_recommendations(user_id=2)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["user_id"] == 2
    assert len(data2["recommendations"]) == 10
    assert data2["fallback"] is False
    
    # Recommendations for different users should be distinct
    assert data1["recommendations"] != data2["recommendations"]

def test_recommend_cold_start_user(api_client):
    # User ID >= 900000 triggers cold start fallback
    resp = api_client.get_recommendations(user_id=950000)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 950000
    assert len(data["recommendations"]) == 10
    assert data["fallback"] is True
    assert data["fallback_type"] == "textual_description"

def test_invalid_movie_not_found(api_client):
    # Movie ID >= 900000 or negative returns 404
    resp = api_client.get_movie(movie_id=950000)
    assert resp.status_code == 404
    
    resp_neg = api_client.get_movie(movie_id=-5)
    assert resp_neg.status_code == 404

    resp_valid = api_client.get_movie(movie_id=100)
    assert resp_valid.status_code == 200

def test_api_latency_success(api_client):
    # Delay is small (e.g. 0.05s), response time should be <= 0.5s
    resp = api_client.get_recommendations(user_id=1, delay=0.05)
    assert resp.status_code == 200
    assert resp.elapsed.total_seconds() <= 0.5

def test_api_latency_forced_failure(api_client):
    # Delay is large (e.g. 0.6s), should fail the <= 0.5s assertion
    resp = api_client.get_recommendations(user_id=1, delay=0.6)
    assert resp.status_code == 200
    with pytest.raises(AssertionError):
        assert resp.elapsed.total_seconds() <= 0.5

def test_mock_evaluate_script(tmp_path):
    output_json = tmp_path / "eval_results.json"
    data_dir = tmp_path / "data"
    
    # Run the mock evaluation script
    cmd = [
        sys.executable,
        "tests/e2e/mock_evaluate.py",
        "--data-dir", str(data_dir),
        "--output", str(output_json),
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Evaluation script failed: {res.stderr}"
    
    # Verify evaluation JSON exists
    assert output_json.exists()
    
    with open(output_json, "r") as f:
        metrics = json.load(f)
        
    # Check that it contains popularity, retrieval-only, and full-pipeline models
    for model in ["Popularity", "Retrieval-Only", "Full-Pipeline"]:
        assert model in metrics
        for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
            assert metric in metrics[model]
            val = metrics[model][metric]
            assert 0.0 <= val <= 1.0
            
    # Verify model comparison: Full-Pipeline should be strictly better than Retrieval-Only,
    # which should be better than or equal to Popularity
    assert metrics["Full-Pipeline"]["NDCG@10"] > metrics["Retrieval-Only"]["NDCG@10"]
    assert metrics["Full-Pipeline"]["NDCG@10"] > metrics["Popularity"]["NDCG@10"]
    assert metrics["Retrieval-Only"]["NDCG@10"] >= metrics["Popularity"]["NDCG@10"]

def test_use_mock_data_flag(use_mock_data):
    # Simple check that the CLI option fixture works
    assert isinstance(use_mock_data, bool)
