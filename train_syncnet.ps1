# Official PyTorch Windows wheels may be built without libuv. torchrun reads this
# setting before scripts.train_syncnet is imported, so it must be set by the launcher.
$env:USE_LIBUV = "0"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25678 -m scripts.train_syncnet `
    --config_path "configs/syncnet/syncnet_16_pixel_attn.yaml"
