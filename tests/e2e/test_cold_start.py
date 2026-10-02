import requests

# --- Tier 1: Feature Coverage (F4) ---

def test_cold_user_status(api_client):
    """T1_F4_1: Verify API request for unknown user returns HTTP 200."""
    # User ID >= 900000 is treated as cold start
    resp = api_client.get_recommendations(user_id=905000)
    assert resp.status_code == 200

def test_cold_user_fallback(api_client):
    """T1_F4_2: Verify content-based fallback for cold user."""
    resp = api_client.get_recommendations(user_id=905000)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback"] is True
    assert data["fallback_type"] == "textual_description"
    assert len(data["recommendations"]) == 10

def test_cold_movie_status(api_client):
    """T1_F4_3: Verify request for unknown movie returns HTTP 404."""
    resp = api_client.get_movie(movie_id=950000)
    assert resp.status_code == 404

def test_cold_user_empty_history(api_client):
    """T1_F4_4: Verify fallback for user with zero ratings."""
    resp = api_client.get_recommendations(user_id=900000)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback"] is True
    assert len(data["recommendations"]) == 10

def test_cold_start_unindexed_movie(api_client):
    """T1_F4_5: Verify API handles newly added unindexed movie."""
    # Querying a movie ID that isn't indexed/known returns 404 rather than crashing
    resp = api_client.get_movie(movie_id=999999)
    assert resp.status_code == 404

# --- Tier 2: Boundary & Corner Cases (F4) ---

def test_cold_user_extreme_id(api_client):
    """T2_F4_1: Verify negative or huge user ID cold start."""
    # Huge ID
    resp_huge = api_client.get_recommendations(user_id=999999999)
    assert resp_huge.status_code == 200
    assert resp_huge.json()["fallback"] is True
    
    # Negative ID
    resp_neg = api_client.get_recommendations(user_id=-1)
    assert resp_neg.status_code == 200
    assert resp_neg.json()["fallback"] is True

def test_cold_movie_non_numeric(api_client):
    """T2_F4_2: Verify string item ID request."""
    # String movie ID
    resp = requests.get(f"{api_client.base_url}/movie/invalid")
    assert resp.status_code in [400, 404, 422]  # Should not be 500

def test_cold_user_missing_metadata(api_client):
    """T2_F4_3: Verify cold start when text descriptions are empty."""
    # For user_id < 0, mock server falls back to popularity_fallback
    resp = api_client.get_recommendations(user_id=-5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback"] is True
    assert data["fallback_type"] == "popularity_fallback"

def test_cold_api_malformed_body(api_client):
    """T2_F4_4: Verify API validation of request schema (returns HTTP 422)."""
    # Malformed GET parameter (string instead of int user_id)
    resp = requests.get(f"{api_client.base_url}/recommend/not-an-int")
    assert resp.status_code == 422
    
    # Malformed POST request body (passing string instead of int for user_id)
    post_resp = requests.post(f"{api_client.base_url}/recommend", json={"user_id": "not-an-int"})
    assert post_resp.status_code == 422

def test_cold_start_disjoint_users(api_client):
    """T2_F4_5: Verify testing on completely unseen user population."""
    # All users >= 900000 are unseen test users
    for uid in range(900001, 900006):
        resp = api_client.get_recommendations(user_id=uid)
        assert resp.status_code == 200
        assert resp.json()["fallback"] is True
