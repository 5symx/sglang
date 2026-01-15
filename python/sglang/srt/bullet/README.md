# bullet usage
## benchmark
launch server
```bash
python -m sglang.launch_server   --host 127.0.0.1   --port 8001   --model-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B   --mem-fraction-static 0.92   --chunked-prefill-size 4096   --served-model-name Qwen3-30B-A3B   --enable-mixed-chunk   --kt-method AMXINT8   --kt-weight-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-INT8   --kt-cpuinfer 32   --kt-threadpool-count 2   --kt-num-gpu-experts 32   --kt-max-deferred-experts-per-token 0

python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 8001   --num-prompts 20 --random-input-len 1024 --random-output-len 16 --random-range-ratio 1.0 --dataset-name random --model /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B --dataset-path /home/ymx/BulletServe/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1
```
### result
tpc set to 71/16/1 no significant difference for single request benchmark.

## args setting in server_args
enable_bullet_engine: bool = False / True

## RoadMap
tp_worker.py
- add forward_stream for forward_bullet
- shared_mng.py - set fixed TPC number  - in bullet_utils.py
- sm_ctrl : with `csrc/` sm configuration
