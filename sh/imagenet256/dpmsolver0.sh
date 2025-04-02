#!/usr/bin/env bash

# 원하는 기본 변수 설정
DEVICES='0'
data="imagenet256_guided"
sampleMethod='dpmsolver++'
type="dpmsolver"
method="multistep"
DIS="time_uniform"
order=3

for scale in 2.0
do
    for steps in 5 6 8 10 12 15 20
    do
        # 실험 결과를 저장할 디렉토리(workdir)
        workdir="/data/experiments_dpm-solver/${data}/${sampleMethod}_${DIS}_order${order}_${steps}_${type}_${scale}"
        echo "===== Running with order=${order}, steps=${steps}, scale=${scale} ====="
        CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
            --config "${data}.yml" \
            --exp "${workdir}" \
            --sample \
            --fid \
            --timesteps "${steps}" \
            --eta 0 \
            --ni \
            --verbose "critical" \
            --skip_type "${DIS}" \
            --sample_type "${sampleMethod}" \
            --dpm_solver_order "${order}" \
            --dpm_solver_method "${method}" \
            --dpm_solver_type "${type}" \
            --scale "${scale}" \
            --thresholding \
            --port 12351
    done
done