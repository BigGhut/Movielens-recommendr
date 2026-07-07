---
name: ml-data-validator
description: Data validation and sanity checking protocols for machine learning datasets. Use this skill to implement data schema verification, handle missing values, check for data drift, and write assertions for incoming data features.
---

# ML Data Validator Skill

Focuses on validating data schemas, quality, and distributions at pipeline boundaries.

## Key Rules
1. **Schema Check**:
   - Verify column names, data types, and check for unexpected nulls using libraries like Pandera or Great Expectations.
2. **Distribution & Drift**:
   - Check training vs. inference datasets for feature drift.
   - Detect extreme outliers or invalid ranges (e.g., negative prices, age > 150).
3. **Pipeline Assertions**:
   - Assert input/output shapes before and after transformation steps.
