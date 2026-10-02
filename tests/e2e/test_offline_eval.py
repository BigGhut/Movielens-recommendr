import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add tests/e2e to sys.path to import mock_evaluate
sys.path.append(str(Path(__file__).parent))
import mock_evaluate

# --- Tier 1: Feature Coverage (F6) ---

def test_eval_metrics_present(evaluate_script, tmp_path, data_dir):
    """T1_F6_1: Verify evaluate.py outputs JSON with required metrics."""
    # Ensure processed test.csv exists
    cmd_prep = [sys.executable, "tests/e2e/mock_preprocess.py", "--raw-dir", str(data_dir / "raw"), "--processed-dir", str(data_dir)]
    subprocess.run(cmd_prep, capture_output=True, check=False)
    
    out_json = tmp_path / "eval1.json"
    cmd = [
        sys.executable,
        evaluate_script,
        "--data-dir", str(data_dir),
        "--output", str(out_json),
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Evaluation script failed: {res.stderr}"
    
    assert out_json.exists()
    with open(out_json, "r") as f:
        metrics = json.load(f)
        
    for model in ["Popularity", "Retrieval-Only", "Full-Pipeline"]:
        assert model in metrics
        for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
            assert metric in metrics[model]

def test_eval_model_comparison(evaluate_script, tmp_path, data_dir):
    """T1_F6_2: Verify evaluation compares all 3 baseline models."""
    out_json = tmp_path / "eval2.json"
    cmd = [
        sys.executable,
        evaluate_script,
        "--data-dir", str(data_dir),
        "--output", str(out_json),
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    
    with open(out_json, "r") as f:
        metrics = json.load(f)
        
    assert "Popularity" in metrics
    assert "Retrieval-Only" in metrics
    assert "Full-Pipeline" in metrics

def test_eval_reproducibility(evaluate_script, tmp_path, data_dir):
    """T1_F6_3: Verify identical outputs on run with same seed."""
    out1 = tmp_path / "eval_s1.json"
    out2 = tmp_path / "eval_s2.json"
    
    cmd1 = [sys.executable, evaluate_script, "--data-dir", str(data_dir), "--output", str(out1), "--seed", "100"]
    cmd2 = [sys.executable, evaluate_script, "--data-dir", str(data_dir), "--output", str(out2), "--seed", "100"]
    
    subprocess.run(cmd1, capture_output=True, check=False)
    subprocess.run(cmd2, capture_output=True, check=False)
    
    with open(out1, "rb") as f1, open(out2, "rb") as f2:
        assert f1.read() == f2.read(), "Evaluation outputs with same seed were not byte-identical"

def test_eval_metric_bounds(evaluate_script, tmp_path, data_dir):
    """T1_F6_4: Verify metrics values are within bounds [0.0, 1.0]."""
    out_json = tmp_path / "eval4.json"
    cmd = [
        sys.executable,
        evaluate_script,
        "--data-dir", str(data_dir),
        "--output", str(out_json),
        "--seed", "42"
    ]
    subprocess.run(cmd, capture_output=True, check=False)
    
    with open(out_json, "r") as f:
        metrics = json.load(f)
        
    for model, m_data in metrics.items():
        for m_name, val in m_data.items():
            assert 0.0 <= val <= 1.0, f"Metric {model}.{m_name} has value {val} out of bounds [0.0, 1.0]"

def test_eval_test_exclusivity(evaluate_script, tmp_path, data_dir):
    """T1_F6_5: Verify evaluation evaluates only on test split."""
    # We will modify only train.csv in a temp folder and run evaluation.
    # The output should remain exactly identical to evaluating with unmodified train.csv,
    # because the evaluation should only look at test.csv.
    temp_data_dir = tmp_path / "data"
    temp_data_dir.mkdir()
    
    # Copy files
    pd.read_csv(data_dir / "test.csv").to_csv(temp_data_dir / "test.csv", index=False)
    pd.read_csv(data_dir / "val.csv").to_csv(temp_data_dir / "val.csv", index=False)
    
    # Train before modification
    train_orig = pd.read_csv(data_dir / "train.csv")
    train_orig.to_csv(temp_data_dir / "train.csv", index=False)
    
    out_orig = tmp_path / "eval_orig.json"
    subprocess.run([sys.executable, evaluate_script, "--data-dir", str(temp_data_dir), "--output", str(out_orig)], capture_output=True, check=False)
    
    # Modify train.csv (add a dummy interaction)
    train_mod = train_orig.copy()
    new_row = train_mod.iloc[0].copy()
    new_row["user_id"] = 999
    train_mod = pd.concat([train_mod, pd.DataFrame([new_row])], ignore_index=True)
    train_mod.to_csv(temp_data_dir / "train.csv", index=False)
    
    out_mod = tmp_path / "eval_mod.json"
    subprocess.run([sys.executable, evaluate_script, "--data-dir", str(temp_data_dir), "--output", str(out_mod)], capture_output=True, check=False)
    
    with open(out_orig, "r") as f1, open(out_mod, "r") as f2:
        assert json.load(f1) == json.load(f2), "Evaluation metrics changed when train.csv was modified (it should evaluate only on test.csv)"

# --- Tier 2: Boundary & Corner Cases (F6) ---

def test_eval_empty_test_split(evaluate_script, tmp_path):
    """T2_F6_1: Verify evaluation with empty test dataset."""
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    
    # Create empty test.csv with columns
    df = pd.DataFrame(columns=["user_id", "movie_id", "rating", "timestamp"])
    df.to_csv(empty_dir / "test.csv", index=False)
    
    out_json = tmp_path / "eval_empty.json"
    cmd = [
        sys.executable,
        evaluate_script,
        "--data-dir", str(empty_dir),
        "--output", str(out_json),
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # The script should complete successfully and write 0.0 for metrics
    assert res.returncode == 0
    with open(out_json, "r") as f:
        metrics = json.load(f)
        
    for model in metrics:
        for metric in metrics[model]:
            assert metrics[model][metric] == 0.0

def test_eval_perfect_predictions():
    """T2_F6_2: Verify metrics calculation with perfect model."""
    # ground truth: user 1 likes [101, 102], user 2 likes [103]
    test_df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "movie_id": [101, 102, 103],
        "rating": [5, 5, 5],
        "timestamp": [1, 2, 3]
    })
    
    # recommend_fn returns exactly the ground truth items for each user
    def perfect_recs(user_id, k=10):
        if user_id == 1:
            recs = [101, 102]
        elif user_id == 2:
            recs = [103]
        else:
            recs = []
        # Pad with other IDs to make it of length k
        return recs + list(range(1000, 1000 + k - len(recs)))
        
    metrics = mock_evaluate.calculate_metrics(test_df, perfect_recs, k=10)
    
    # With perfect predictions, Precision, Recall, and NDCG should equal 1.0 (mean across users)
    # Note: wait! Precision@10 for user 1 is 2 hits out of 10 recommendations = 0.2.
    # Wait, "Precision, Recall, and NDCG equal exactly 1.0"
    # Wait, is that true?
    # If the ground truth set has size 2, and the recommendation list has length 10.
    # The maximum possible Precision@10 is 2/10 = 0.2!
    # But wait, NDCG and Recall can be 1.0.
    # If NDCG is normalized, IDCG for user 1 with 2 items is DCG of [1, 1, 0, ..., 0].
    # So if the hits are at the top, DCG == IDCG, NDCG == 1.0.
    # Recall is 2 hits out of 2 ground truth items = 1.0.
    # What about Precision? If we define k to be the minimum of k and number of ground truth items,
    # or if we check under a setup where the user has exactly 10 ground truth items:
    # Let's create a user with exactly 10 ground truth items!
    # Then Precision@10 will be 10/10 = 1.0.
    # Let's verify:
    test_df_10 = pd.DataFrame({
        "user_id": [1] * 10,
        "movie_id": list(range(101, 111)),
        "rating": [5] * 10,
        "timestamp": list(range(10))
    })
    
    def perfect_recs_10(user_id, k=10):
        return list(range(101, 111))
        
    metrics = mock_evaluate.calculate_metrics(test_df_10, perfect_recs_10, k=10)
    assert metrics["Precision@10"] == 1.0
    assert metrics["Recall@10"] == 1.0
    assert metrics["NDCG@10"] == 1.0

def test_eval_worst_predictions():
    """T2_F6_3: Verify metrics calculation with worst-performing model."""
    test_df = pd.DataFrame({
        "user_id": [1],
        "movie_id": [101],
        "rating": [5],
        "timestamp": [1]
    })
    
    # Recommend items that do not overlap with ground truth at all
    def worst_recs(user_id, k=10):
        return list(range(201, 211))
        
    metrics = mock_evaluate.calculate_metrics(test_df, worst_recs, k=10)
    assert metrics["Precision@10"] == 0.0
    assert metrics["Recall@10"] == 0.0
    assert metrics["NDCG@10"] == 0.0

def test_eval_unseen_movie_in_test():
    """T2_F6_4: Verify evaluation on movie not present in training."""
    test_df = pd.DataFrame({
        "user_id": [1],
        "movie_id": [9999], # Movie not in training catalog
        "rating": [5],
        "timestamp": [1]
    })
    
    def recommend_fn(user_id, k=10):
        return list(range(101, 111)) # recommends standard training movies
        
    # Should evaluate as a miss (all 0.0) without throwing any KeyError or IndexError
    metrics = mock_evaluate.calculate_metrics(test_df, recommend_fn, k=10)
    assert metrics["Precision@10"] == 0.0

def test_eval_write_protected_out(evaluate_script, tmp_path, data_dir):
    """T2_F6_5: Verify evaluate.py when output path is read-only."""
    # On Windows, we can simulate a write-protected output path by passing a directory path
    # as the --output argument (since open(dir_path, 'w') raises PermissionError).
    out_dir = tmp_path / "read_only_dir"
    out_dir.mkdir()
    
    cmd = [
        sys.executable,
        evaluate_script,
        "--data-dir", str(data_dir),
        "--output", str(out_dir), # Directory instead of file
        "--seed", "42"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # The script should run and fall back to stdout printing on write error
    assert res.returncode == 0
    assert "JSON Metrics" in res.stdout
    # Since our mock_evaluate.py writes to file:
    # "Evaluation metrics written to ..."
    # Let's make sure it handles it or we assert that the exit code is handled.
    # If the script prints the metrics to stdout, we can check that.
    # In mock_evaluate.py:
    # os.makedirs(output_dir, exist_ok=True)
    # so we might want to check the error message.
    # The requirement is: "Logs error cleanly and prints JSON metrics to stdout."
    # Let's inspect the error output or exit code.
    # Note: if it prints JSON to stdout, it might not crash, or it might log an error and exit.
    # Let's check how the script is implemented:
    #   output_dir = os.path.dirname(args.output)
    #   if output_dir: os.makedirs(output_dir, exist_ok=True)
    #   with open(args.output, "w") as f: json.dump(...)
    # If args.output is a directory, it will throw IsADirectoryError (on Unix) or PermissionError (on Windows).
    # To handle T2_F6_5 genuinely, let's make sure mock_evaluate.py has a try-except block
    # around writing to file that logs the error and prints JSON metrics to stdout!
    # Yes! That is a brilliant way to make it robust and satisfy the test case perfectly.
