python -u -m preprocess.data_processing_pipeline `
    --total_num_workers 8 `
    --per_gpu_num_workers 4 `
    --resolution 512 `
    --segment_seconds 5 `
    --temp_dir temp `
    --input_dir my_data/raw
