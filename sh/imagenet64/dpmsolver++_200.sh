#!/usr/bin/env bash

# 원하는 기본 변수 설정
DEVICES='0'
data="imagenet64_128"
sampleMethod='dpmsolver++'   # sample_type in diffusion.py
type="dpmsolver"          # dpm_solver_type in diffusion.py
order=3
method="multistep"
DIS="logSNR"
steps=200

# 실험 결과를 저장할 디렉토리(workdir)
workdir="samples/64x64_diffusion/${sampleMethod}_order${order}_${steps}"
echo "===== Running with order=${order}, steps=${steps} ====="
CUDA_VISIBLE_DEVICES="${DEVICES}" python main_npz.py \
    --config "${data}.yml" \
    --exp "${workdir}" \
    --sample \
    --fid \
    --timesteps "${steps}" \
    --eta 0 \
    --ni \
    --skip_type "${DIS}" \
    --sample_type "${sampleMethod}" \
    --dpm_solver_order "${order}" \
    --dpm_solver_method "${method}" \
    --dpm_solver_type "${type}" \
    --port 12351