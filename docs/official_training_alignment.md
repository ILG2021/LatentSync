# Official training baseline with activation offload

The training implementation and 512 Stage 1/2 presets were aligned with
bytedance/LatentSync `main` on 2026-09-05.

Training losses, gradient checkpointing, AMP updates, validation scoring, and
weight-only checkpoint saving follow the official implementation. Both stages use
AdamW at 1e-5, 10,000,000 maximum steps, and a 10,000-step save interval.

Windows single-GPU startup, valid timestamp directory names, and local dataset/cache
paths are retained. The offload preset additionally wraps pixel-loss computation in
selective saved-activation CPU offload. Standard presets do not enable offload.

There is no automatic checkpoint search, optimizer-state sidecar restore, checkpoint
pruning, within-epoch replay, or custom AMP retry loop. `resume_ckpt_path` must name
a weights file. A restored step count does not restore optimizer or scheduler state.
The recovery preset explicitly selects checkpoint-1000; other presets select the
official `checkpoints/latentsync_unet.pt`. Each launch creates a timestamped directory.

For a controlled comparison, preserve the same input paths and checkpoint. Running
`tools.write_fileslist` without `--skip_config_update` still changes validation paths
and training budgets; use that flag to preserve these official preset values.

This alignment does not establish the cause of previous NaNs. GPU validation must
be performed on the training machine. The broader preprocessing/inference Windows
adaptations have not been reverted.
