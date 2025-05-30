# 원하는 기본 변수 설정
DEVICES='0,1,2,3,4,5,6,7'
data="imagenet64"
type="data_prediction"          # dpm_solver_type in diffusion.py
method="multistep"
DIS="logSNR"

for steps in 5 6 8 10 12 15 20 25
do
    # UniPC
    sampleMethod='unipc'
    workdir="samples/64x64_diffusion/${sampleMethod}_order3_${steps}"
    echo "===== Running with method=${sampleMethod}, order=3, steps=${steps} ====="
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
        --dpm_solver_order "3" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351

    # DPM-Solver++
    sampleMethod='dpmsolver++'
    workdir="samples/64x64_diffusion/${sampleMethod}_order3_${steps}"
    echo "===== Running with method=${sampleMethod}, order=3, steps=${steps} ====="
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
        --dpm_solver_order "3" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "dpmsolver" \
        --port 12351    

    # RBF-order3
    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order3_${steps}"
    echo "===== Running with method=${sampleMethod}, order=3, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "3" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351        

    # RBF-order4
    sampleMethod='rbf_ecp_marginal'
    workdir="samples/64x64_diffusion/${sampleMethod}_order4_${steps}"
    echo "===== Running with method=${sampleMethod}, order=4, steps=${steps} ====="
    CUDA_VISIBLE_DEVICES="${DEVICES}" python main.py \
        --config "${data}.yml" \
        --exp "${workdir}" \
        --sample \
        --fid \
        --timesteps "${steps}" \
        --scale_dir "/data/guided-diffusion/scale/rbf_ecp_marginal_64" \
        --eta 0 \
        --ni \
        --skip_type "${DIS}" \
        --sample_type "${sampleMethod}" \
        --dpm_solver_order "4" \
        --dpm_solver_method "${method}" \
        --dpm_solver_type "${type}" \
        --port 12351            
done
