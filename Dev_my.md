# 13012026
4090 - sglang
L40 - kt-kernel
```bash
python -m sglang.launch_server   --host 0.0.0.0   --port 8000   --model-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B   --mem-fraction-static 0.92   --chunked-prefill-size 4096   --served-model-name Qwen3-30B-A3B   --enable-mixed-chunk   --kt-method AMXINT8   --kt-weight-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-INT8   --kt-cpuinfer 32   --kt-threadpool-count 2   --kt-num-gpu-experts 32   --kt-max-deferred-experts-per-token 2 

python -m sglang.bench_serving   --backend sglang   --host 0.0.0.0   --port 8000   --num-prompts 1 --random-input-len 1024 --random-output-len 16 --random-range-ratio 1.0 --dataset-name random --model /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B --dataset-path /home/ymx/BulletServe/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

output 

Prefill batch, #new-seq: 1, #new-token: 1024, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, 
[2026-01-13 11:27:28] Shared Manager without lib return total_tpc : shared_array_264493
[2026-01-13 11:27:28] Prefill TPC 71, policy default

todo: add P+D with sm partition.

# 14-01-2026
```bash
python -m sglang.launch_server   --host 127.0.0.1   --port 8001   --model-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B   --mem-fraction-static 0.92   --chunked-prefill-size 4096   --served-model-name Qwen3-30B-A3B   --enable-mixed-chunk   --kt-method AMXINT8   --kt-weight-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-INT8   --kt-cpuinfer 32   --kt-threadpool-count 2   --kt-num-gpu-experts 32   --kt-max-deferred-experts-per-token 2  --disable-radix-cache --disable-chunked-prefix-cache

python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 8001   --num-prompts 2 --random-input-len 1024 --random-output-len 16 --random-range-ratio 1.0 --dataset-name random --model /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B --dataset-path /home/ymx/BulletServe/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```
CUTLASS error - reduce input length for fewer GPU memory usage.

```bash
python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 8001   --num-prompts 2 --random-input-len 128 --random-output-len 16 --random-range-ratio 1.0 --dataset-name random --model /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B --dataset-path /home/ymx/BulletServe/dataset/ShareGPT_V3_unfiltered_cleaned_split.json 
```

set --request-rate 1
```bash
python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 8001   --num-prompts 20 --random-input-len 1024 --random-output-len 16 --random-range-ratio 1.0 --dataset-name random --model /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B --dataset-path /home/ymx/BulletServe/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1
```

profiling 
```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 40 --duration 20 python -m sglang.launch_server   --host 127.0.0.1   --port 8001   --model-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B   --mem-fraction-static 0.92   --chunked-prefill-size 4096   --served-model-name Qwen3-30B-A3B   --enable-mixed-chunk   --kt-method AMXINT8   --kt-weight-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-INT8   --kt-cpuinfer 32   --kt-threadpool-count 2   --kt-num-gpu-experts 32   --kt-max-deferred-experts-per-token 2  --disable-radix-cache --disable-chunked-prefix-cache 
```

## multi-stream for kt - error
remove multi-stream from bullet
for bullet test cuda_stream for bullet forward  - check!
set forward_stream to default stream in torch.cuda - segmantfault.

add with torch.cuda.stream(self.forward_stream) to each self.model_runner in forward_batch_generation_bullet()
set back with forward_stream - synchronize with this stream when CPU/GPU comm
cuda graph replay error --disable-cuda-graph. - error for cuda error

remove set_stream_mask for cuda replay error check! with forward_stream is default_stream
remove all torch.cuda.stream(self.forward_stream) for testing - check!
-set forward with stream / default error / non-default check!

non-default with forward / assert in kt_rp_wrapper set_sm in tp_worker  - request 1 check / 2 error cutlass / 2 check!!!????
support forward with forward_stream

## TPC set
set TPC for 71 - no performance different.
change tpc to 10 for performance check - no difference.

set tpc to 16 - test  - check!?
flashinfer paged_kernel length - change attention kernel into fa3 to test. - error device-error for apply(forward)

fa3 request 2 
16/71 no difference 1 no difference

## accuracy test
launch server
```bash
python -m sglang.launch_server   --host 127.0.0.1   --port 8001   --model-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B   --mem-fraction-static 0.92   --chunked-prefill-size 4096   --served-model-name Qwen3-30B-A3B   --enable-mixed-chunk   --kt-method AMXINT8   --kt-weight-path /data0/ymx/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-INT8   --kt-cpuinfer 32   --kt-threadpool-count 2   --kt-num-gpu-experts 16   --kt-max-deferred-experts-per-token 0
```

deffer expert test
```bash
python3 benchmark/gsm8k/bench_sglang.py --num-questions 1 # 20
#score 0.95~ / Latency: 23.514 s Output throughput: 143.869 token/s - 2 deffer expert
#score 0.95~ / Latency: 15.450 s / Output throughput: 159.931 token/s - 0 deffer expert
python3 -m sglang.test.run_eval --port 8001 --eval-name gpqa --num-examples 10 --repeat 2 --thinking-mode qwen3
#Repeat: 2, mean: 0.200 Scores: ['0.200', '0.200'] - 0 expert deffer
#Repeat: 2, mean: 0.300 Scores: ['0.300', '0.300'] - 2 expert deffer
python3 -m sglang.test.run_eval --port 8001 --eval-name mmlu --num-examples 10 --repeat 2 --thinking-mode qwen3
# score  Repeat: 2, mean: 0.600 Scores: ['0.600', '0.600']' - 0 expert deffer
# score  Repeat: 2, mean: 0.700 Scores: ['0.700', '0.700']' - 2 expert deffer
# score  Repeat: 2, mean: 0.700 Scores: ['0.700', '0.700']' - 4 expert deffer
python3 -m sglang.test.run_eval --port 8001 --eval-name aime25 --num-examples 10 --repeat 2 --thinking-mode qwen3
#Repeat: 2, mean: 0.000 Scores: ['0.000', '0.000'] - 0 expert deffer

python3 -m sglang.test.run_eval --port 8001 --eval-name mmlu --num-examples 20 --repeat 5 --thinking-mode qwen3
#Repeat: 5, mean: 0.840 Scores: ['0.850', '0.850', '0.800', '0.850', '0.850'] - 4 expert
#Repeat: 5, mean: 0.840 Scores: ['0.850', '0.850', '0.800', '0.850', '0.850'] - 0 expert
python3 -m sglang.test.run_eval --port 8001 --eval-name gpqa --num-examples 20 --repeat 5 --thinking-mode qwen3
#Repeat: 5, mean: 0.280 Scores: ['0.350', '0.250', '0.350', '0.250', '0.200'] - 0 expert deffer
#Repeat: 5, mean: 0.230 Scores: ['0.200', '0.200', '0.300', '0.250', '0.200'] - 2 expert deffer
#Repeat: 5, mean: 0.210 Scores: ['0.100', '0.300', '0.200', '0.250', '0.200'] - 4 expert deffer

```

# 15-01-2026
## get topk_output for FusedMoe to get the activate expert id
when to activate moe
load model - add log in FusedMoe to test for kt - check!

```bash
--log-level debug 2>&1 | tee output-115.log
```





