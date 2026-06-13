#!/bin/bash

CLUSTER_NAME="wice"
JOB_PARTITION="batch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

savedir="${REPO_ROOT}/runs"
mkdir -p "${savedir}/logfile"
N=50
MPI_WORKERS=30
tag="_20260607"

# I:   ca/co=25, g=10  |  II: ca/co=50, g=10  |  III: ca/co=50, g=2
scenario_list=(I II III)

p_mino_list=(0.1 0.2)
rho_list=(0.8 0.5)
rho_cross_list=(0.5 0.2)
Bayes_risk_list=(0.15 0.25)

scenario_dims() {
  case "$1" in
    I)   echo "25 25 10" ;;
    II)  echo "50 50 10" ;;
    III) echo "50 50 2" ;;
  esac
}

for scenario in "${scenario_list[@]}"; do
  read -r case_ca case_co case_g <<< "$(scenario_dims "$scenario")"
  for p_mino in "${p_mino_list[@]}"; do
    for rho in "${rho_list[@]}"; do
      for rho_cross in "${rho_cross_list[@]}"; do
        if (( $(echo "$rho_cross < $rho" | bc -l) )); then
          for Bayes_risk in "${Bayes_risk_list[@]}"; do
            case="simu_s${scenario}_N${N}_ca${case_ca}_co${case_co}_n1000_pmino${p_mino}_rho${rho}_rhoc${rho_cross}_B${Bayes_risk}_g${case_g}_${CLUSTER_NAME}_${JOB_PARTITION}"
            final_filename="${savedir}/result/${case}${tag}_N=${N}.csv"
            mkdir -p "${savedir}/logfile/${case}/hpc"
            if [ -f "$final_filename" ]; then
              echo "File exists: $final_filename"
              continue
            fi

            JOB_NAME=$case
            if squeue -M ${CLUSTER_NAME} -u $USER -o "%j" -h | grep -q "$JOB_NAME$"; then
              echo "job '$JOB_NAME' exists"
              continue
            fi

            cat > job_script.slurm <<EOF
#!/bin/bash -l
#SBATCH --cluster=${CLUSTER_NAME} --partition=${JOB_PARTITION} --nodes=1 --ntasks=$((MPI_WORKERS + 1)) --cpus-per-task=1
#SBATCH --time=00-2:00:00
#SBATCH --account=lp_stat_cluster
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${savedir}/logfile/${JOB_NAME}/hpc/${JOB_NAME}.log
export R_LIBS_USER=/vsc-hard-mounts/leuven-data/356/vsc35603/miniconda3/envs/Renv/lib/R/library
export ECSLR_MPI_WORKERS=${MPI_WORKERS}
source ~/.bashrc
module load cluster/${CLUSTER_NAME}/${JOB_PARTITION}
module load OpenMPI/4.1.1-GCC-10.3.0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_orte_base_help_aggregate=0
cd "${SCRIPT_DIR}"
mpirun --bind-to none -np 1 /vsc-hard-mounts/leuven-data/356/vsc35603/miniconda3/envs/Renv/bin/Rscript benchmark_simulation.R ${scenario} ${p_mino} ${rho} ${rho_cross} ${Bayes_risk} ${tag}
EOF
            sbatch job_script.slurm
            rm job_script.slurm
          done
        fi
      done
    done
  done
done
