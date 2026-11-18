# Backup scope

This branch preserves the canonical WSL source as of 2026-11-18. The separate
Windows checkout is preserved on `backup/windows-thesis-20261118`; it is not a
replacement for the canonical WSL training and evaluation implementation.

`reports/human_eval_20261105/` contains the final compact report, summary tables,
training curves and all 69 plotted evaluation trajectories (32 matched cases and
37 additional, unmatched cases). Original source hashes and paths are retained
as provenance; those machine-local paths are not downloadable assets.

Raw human recordings, checkpoints, runtime environments, credentials, local
assistant configuration and bulk experiment artifacts are not included. The
report/gallery is a results snapshot, not a complete reproducibility archive.

Unpublished local modifications to third-party dependencies, including
`external/unitree_rl_mjlab`, `safe-locomotion`, `safe_rl` and `external/LOOP`, are
also excluded pending publication permission. In particular, the current gentle
locomotion integration depends on a local Unitree observation-adaptation patch.
A fresh checkout of this branch alone does not include that dependency fix or
the required model weights. Dependency patches remain in the local backup only;
no upstream repository has been modified by this publication.

Historical experiment launchers are retained as source, not instructions to
restart canceled cluster jobs or retired autonomous research workflows.
