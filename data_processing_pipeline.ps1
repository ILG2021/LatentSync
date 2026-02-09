python -u -m preprocess.data_processing_pipeline `
    --total_num_workers 1 `
    --per_gpu_num_workers 1 `
    --resolution 512 `
    --sync_conf_threshold 3 `
    --temp_dir temp `
    --input_dir my_data
