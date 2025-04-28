# 원하는 기본 변수 설정
DEVICES='0,1,2,3,4,5,6,7'
#DEVICES='0'
data="imagenet64"
type="data_prediction"          # dpm_solver_type in diffusion.py
method="multistep"
DIS="logSNR"

for order in 4 5 6 7 8
do
for steps in 5 6 8 10 12 15 20 25 30 35 40
do
    # 실험 결과를 저장할 디렉토리(workdir)
    sampleMethod='unipc'
    workdir="samples/64x64_diffusion/${sampleMethod}_order${order}_${steps}"
    echo "===== Running with order=${order}, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
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

done
done