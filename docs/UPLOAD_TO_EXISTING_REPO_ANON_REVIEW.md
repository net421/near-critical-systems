# Updating the existing anonymous repository

The previous submission used:

`https://github.com/anonymousresearchrep-cmd/near-critical-systems`

That repository currently contains files for an older first-passage/DFR
manuscript. Do not add this package beside those files. Replace the repository
root so that only the contents of this package remain on the default branch.

## Recommended Git workflow

1. Extract `github_repository_buffer_threshold_policy.zip`.
2. Clone the existing anonymous repository.
3. Remove its tracked manuscript-specific files.
4. Copy the extracted package contents into the repository root.
5. Confirm that `README.md`, `src/`, `scripts/`, `tests/`, `results/`, and
   `docs/` are at the root, not inside another folder.
6. Use the repository-local anonymous Git identity:

```bash
git config user.name "Anonymous Researcher"
git config user.email "anonymousresearchrep@gmail.com"
```

7. Run the checks before committing:

```bash
python -m pip install -e .
python -m pip install pytest
python scripts/verify_definitive_results.py
python -m pytest tests -q
```

8. Commit and push the replacement from the anonymous account.

Do not use a personal Git name or email. Do not upload the ZIP as a single file;
GitHub will not unpack it. Upload or push the extracted contents.
