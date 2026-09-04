# Official PyTorch Windows wheels may be built without libuv. torchrun reads this
# setting before scripts.train_unet is imported, so it must be set by the launcher.
$env:USE_LIBUV = "0"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25679 -m scripts.train_unet `
    --unet_config_path "configs/unet/stage1_512.yaml"
