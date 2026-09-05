# Official training baseline with activation offload

The training implementation and 512 Stage 1/2 presets were aligned with
bytedance/LatentSync `main` on 2026-09-05.

Training losses, gradient checkpointing, and AMP updates follow the official implementation.
The official validation scorer is used when its optional modules are available;
otherwise a warning is emitted and validation videos are still generated. Checkpoints
contain weights and step counts and are published atomically to avoid auto-selecting
interrupted saves. Both stages use
AdamW at 1e-5, 10,000,000 maximum steps, and a 10,000-step save interval.

Windows single-GPU startup, valid timestamp directory names, and local dataset/cache
paths are retained. The offload preset additionally wraps pixel-loss computation in
selective saved-activation CPU offload. Standard presets do not enable offload.

Stage 2 retains automatic checkpoint search: first select the greatest step in its
own output directory (including fixed and timestamped layouts), otherwise select
the latest Stage 1 checkpoint and reset the stage step count. An explicit path is
also supported. There is no optimizer-state sidecar restore, checkpoint pruning,
within-epoch replay, or custom AMP retry loop. A restored step count does not restore
optimizer or scheduler state. The recovery preset explicitly selects checkpoint-1000;
Stage 1 selects `checkpoints/latentsync_unet.pt`. Each launch creates a timestamped directory.

For a controlled comparison, preserve the same input paths and checkpoint. Running
`tools.write_fileslist` without `--skip_config_update` still changes validation paths
and training budgets; use that flag to preserve these official preset values.

This alignment does not establish the cause of previous NaNs. GPU validation must
be performed on the training machine. The broader preprocessing/inference Windows
adaptations have not been reverted.
