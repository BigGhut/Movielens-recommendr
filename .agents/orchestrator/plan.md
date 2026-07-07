# Project Plan: RecSys Two-Tower MovieLens-1M

## Orchestrator Strategy
We are executing under the **Project Pattern** with a **Dual Track** architecture:
1. **E2E Testing Track**: Designed independently from user requirements to construct a comprehensive opaque-box test suite (Tiers 1-4).
2. **Implementation Track**: Implements the data pipeline, retrieval, re-ranking, API, UI, and Docker setup in milestones.

## Plan Steps

### Phase 1: Planning & Setup
- [x] Step 1.1: Initialize orchestrator briefing, progress, context, and plan files.
- [x] Step 1.2: Perform global project decomposition and write `PROJECT.md`.
- [ ] Step 1.3: Launch the E2E Testing Track to define requirements, metrics, and create tests.
- [ ] Step 1.4: Launch the Implementation Track to execute milestones.

### Phase 2: Dual Track Execution
- **E2E Testing Track**:
  - Milestones for feature inventory, test runner infrastructure, and writing Tier 1-4 tests.
  - Output: `TEST_READY.md` and `TEST_INFRA.md`.
- **Implementation Track**:
  - Milestone 1: Data Pipeline.
  - Milestone 2: Retrieval Model.
  - Milestone 3: Re-ranking Model.
  - Milestone 4: API, UI, and Docker.
  - Milestone 5: Verification (Integrity & E2E Tests + Tier 5 Adversarial Coverage Hardening).

### Phase 3: Final Acceptance & Handover
- [ ] Step 3.1: Run Forensic Integrity Audit and full E2E test verification.
- [ ] Step 3.2: Compile handoff.md and handover report.
