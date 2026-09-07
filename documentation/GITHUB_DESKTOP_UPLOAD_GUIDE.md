# GitHub Desktop upload guide for this cleaned repository

This cleaned tree is intended to be copied into a **clone of the existing GitHub repository**, not published as a new unrelated repository.

## Safest beginner workflow

1. Keep a backup of your current working folder.
2. Open GitHub Desktop and sign in.
3. Clone the existing repository `efatme/leakage-audited-multion-electrode-ml.v1` into a new empty local folder.
4. In GitHub Desktop, create a new branch named `paper/cmt-stage-learnability` from `main`.
5. Close Jupyter notebooks that are editing files in the repository.
6. Copy the contents of this cleaned package into the cloned repository folder. Allow Windows to merge/replace files.
7. Return to GitHub Desktop and inspect the **Changes** tab.
8. Confirm that local ZIPs, notebook checkpoints, and CMT runtime `_checkpoints` are not listed.
9. Use commit message: `Add frozen CMT stage-dependent learnability analysis`.
10. Click **Commit to paper/cmt-stage-learnability**.
11. Click **Push origin** / **Publish branch**.
12. Open the repository on GitHub in a browser and verify that `papers/cmt/`, `README.md`, `CHANGELOG.md`, and the frozen Stage F summary are visible.

Do not merge to `main` until the branch has been checked on GitHub.

## Files intentionally excluded from Git history

- local `.zip` patch/completed-run bundles;
- Jupyter `.ipynb_checkpoints/`;
- CMT runtime fit `_checkpoints/`;
- Quantum ESPRESSO restart/scratch files;
- pseudopotential binaries (`*.UPF`);
- local environment/secrets.

## Large-file note

The repository already contains some historical CSV outputs larger than 50 MB but below GitHub's 100 MB hard per-file limit. Do not duplicate or repackage them. New submission ZIP/TIFF bundles should normally be attached to a GitHub Release rather than committed as another large archive.
