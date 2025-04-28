# 원하는 기본 변수 설정
DEVICES='0,1,2,3,4,5,6,7'
#DEVICES='0'
data="imagenet64"
type="data_prediction"          # dpm_solver_type in diffusion.py
method="multistep"
DIS="logSNR"

for steps in 5 6 8 10 12 15 20 25 30 35 40
do
    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order7_${steps}"
    echo "===== Running with method=${sampleMethod}, order=7, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "7" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 

    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order8_${steps}"
    echo "===== Running with method=${sampleMethod}, order=8, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "8" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 

    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order9_${steps}"
    echo "===== Running with method=${sampleMethod}, order=9, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "9" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 

    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order10_${steps}"
    echo "===== Running with method=${sampleMethod}, order=10, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "10" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 

    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order11_${steps}"
    echo "===== Running with method=${sampleMethod}, order=11, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "11" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 

    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order12_${steps}"
    echo "===== Running with method=${sampleMethod}, order=12, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_M=128_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "12" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351 
done
