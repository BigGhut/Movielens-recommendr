## 2026-07-06T12:02:56Z
You are the E2E Testing Worker (worker_e1_e2_1).
Your working directory is z:/pet-project/recsys-two-tower/.agents/worker_e1_e2_1/.
Your task is to implement Milestones E1 and E2 of the E2E Testing Track:
1. Write the `TEST_INFRA.md` file at the project root (`z:/pet-project/recsys-two-tower/TEST_INFRA.md`) based on the design drafted in z:/pet-project/recsys-two-tower/.agents/explorer_e1_1/handoff.md.
2. Set up the E2E testing directory structure under `tests/e2e/`.
3. Create `tests/e2e/conftest.py` containing:
   - Pytest CLI option `--use-mock-data`.
   - Fixtures for starting/stopping a mock API server (which will simulate the `/recommend/{user_id}` and `/health` endpoints).
   - Fixtures to query this server.
4. Implement a mock API server in `tests/e2e/mock_server.py` using FastAPI or standard library `http.server`. The mock server should support:
   - Healthy status on `/health`.
   - Re-ranking of 200 candidates to top 10 on `/recommend/{user_id}`.
   - Cold start for unknown users (e.g. user_id >= 900000) using textual fallback.
   - Cold start for invalid items (returning HTTP 404).
   - Simulating response latency (with configurable delay to test latency assertion).
5. Implement a mock evaluation script in `tests/e2e/mock_evaluate.py` that mimics `evaluate.py`, returning a JSON file with `Precision@10`, `Recall@10`, and `NDCG@10` for Popularity, Retrieval, and Full-Pipeline models.
6. Write a small set of baseline tests under `tests/e2e/` to verify that the harness can successfully run, connect to the mock server, execute the mock evaluate script, and assert success/failure.
7. Run the test suite using pytest to verify that it executes successfully, and also verify that forcing incorrect behaviors (like making the mock server slow or return 500) causes the corresponding assertions to fail as expected.
8. Document your changes and the command outputs in your handoff report at z:/pet-project/recsys-two-tower/ .agents/worker_e1_e2_1/handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please initialize progress.md in your working directory and update it as you go.
When finished, send a message to parent (d3150900-7543-4ddc-9d6a-7d6934669ceb).
