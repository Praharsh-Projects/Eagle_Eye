# Canonical Eagle Eye application freeze

This repository revision is the canonical public source publication of the Eagle Eye React/Vite operations workspace served by FastAPI. The canonical launcher is `./run_eagle_eye.sh`, and the interface opens at `http://127.0.0.1:8000/overview`. Streamlit remains a QA fallback only.

The source comes from product revision `eagle_eye_computational_eval_v3`, derived from the corrected `benchmark_recovery_v2` baseline. The local protected record contains 156 files and has baseline SHA-256 `fddefb2f00c11f76e0079c937e0a20b5c24a41f44a2fd7837036f50461fd3d49`.

- `protected_application.sha256` is the complete local protected-file receipt.
- `github_source_snapshot.sha256` binds the protected application-source files included in this public repository.
- Git tag `eagle-eye-computational-eval-v3` binds the complete public source tree.

The public repository deliberately excludes `.env`, supplied or generated private data, the local documentary index, caches, built assets, test output, model caches, and thesis workspaces. It includes the public demo fallback. These exclusions avoid publishing credentials, restricted movement traces, or large generated evidence while preserving the canonical application code.

This publication does not create a new model revision, alter R12 evidence, or rerun any evaluation.
