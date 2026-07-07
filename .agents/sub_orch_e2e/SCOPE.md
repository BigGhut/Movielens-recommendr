# Project: Two-Tower RecSys MovieLens-1M
# Scope: E2E Testing Track

## Architecture
- **E2E Testing Track**: Requirement-driven, opaque-box testing of the recommendation system.
- **Interfaces tested**:
  1. CLI evaluation: `evaluate.py` should run and generate a metrics JSON.
  2. HTTP API endpoints: `GET /health` and `POST /recommend` / `GET /recommend/{user_id}` (handling normal users, cold-start users, invalid movies, and performance latency).
  3. Preprocessing / Data split integrity constraints: verification that split has no leakage, users with < 5 interactions are filtered, and splits are non-overlapping.
- **Verification framework**: `pytest` running tests against the API (either live server or test client) and codebase outputs.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | E1: E2E Test Infra Design | Design and publish `TEST_INFRA.md` at project root specifying features, methodology, and test layout | None | DONE |
| 2 | E2: Test Runner & Harness Setup | Set up pytest, mock data/APIs for initial test execution to ensure test harness runs and can fail/pass | E1 | DONE |
| 3 | E3: 4-Tier Test Cases | Implement 4 tiers of test cases (Tier 1: Feature, Tier 2: Boundary, Tier 3: Cross-Feature, Tier 4: Real-world Workloads) | E2 | DONE |
| 4 | E4: Final Verification & Ready | Run the test suite against mocks/actual code, verify correct/incorrect behavior, publish `TEST_READY.md` | E3 | IN_PROGRESS (sub_orch_e2e) |

## Interface Contracts
- **Test Runner Command**: `pytest tests/e2e` or similar command.
- **Input Requirements**: Original Request requirements (temporal split, retrieval model candidate generation, re-ranking GBDT model, cold-start fallback, HTTP API response latency).
- **Output Requirements**: `TEST_INFRA.md` and `TEST_READY.md` at project root.
