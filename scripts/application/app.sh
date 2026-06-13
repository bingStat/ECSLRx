#!/usr/bin/env bash

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-wice}"
JOB_PARTITION="${JOB_PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-lp_stat_cluster}"
RENV_RSCRIPT="${RENV_RSCRIPT:-/data/leuven/356/vsc35603/miniconda3/envs/Renv/bin/Rscript}"
RENV_LIB="${RENV_LIB:-/data/leuven/356/vsc35603/miniconda3/envs/Renv/lib/R/library}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${WORKTREE_ROOT}/runs}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "${SCRIPT_DIR}/../../package/ECSLRx" && pwd)}"
PACKAGE_LIB="${PACKAGE_LIB:-${WORKTREE_ROOT}/Rlib}"
RESULT_ROOT="${RESULT_ROOT:-${RUNS_ROOT}/result}"
TEMP_ROOT="${TEMP_ROOT:-${RUNS_ROOT}/temp}"
LOG_ROOT="${LOG_ROOT:-${RUNS_ROOT}/logfile}"
HPC_LOG_ROOT="${HPC_LOG_ROOT:-${LOG_ROOT}}"
APPLICATION_FILE_TAG="${APPLICATION_FILE_TAG:-}"

mkdir -p "${PACKAGE_LIB}" "${RESULT_ROOT}" "${TEMP_ROOT}" "${LOG_ROOT}" "${HPC_LOG_ROOT}"

expand_values() {
    local spec=$1
    local values=()
    local token

    spec=${spec//,/ }
    for token in ${spec}; do
        if [[ ${token} =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]}
            local end=${BASH_REMATCH[2]}
            local i
            if (( start <= end )); then
                for ((i = start; i <= end; i++)); do
                    values+=("${i}")
                done
            else
                for ((i = start; i >= end; i--)); do
                    values+=("${i}")
                done
            fi
        else
            values+=("${token}")
        fi
    done

    printf '%s\n' "${values[@]}"
}

install_ecslrx_package() {
    local renv_bin
    local renv_r
    renv_bin="$(dirname "${RENV_RSCRIPT}")"
    renv_r="${renv_bin}/R"
    echo "Installing ECSLRx from ${PACKAGE_ROOT} into ${PACKAGE_LIB}..."
    R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \
        R_LIBS="${PACKAGE_LIB}:${RENV_LIB}" R_LIBS_USER="${RENV_LIB}" PATH="${renv_bin}:${PATH}" \
        "${renv_r}" CMD INSTALL --preclean --library="${PACKAGE_LIB}" "${PACKAGE_ROOT}"
}

cli_case_spec=${1:-}
cli_k_spec=${2:-}

if [ -n "${APPLICATION_CASE_VALUES:-}" ]; then
    mapfile -t case_values < <(expand_values "${APPLICATION_CASE_VALUES}")
elif [ -n "${cli_case_spec}" ]; then
    mapfile -t case_values < <(expand_values "${cli_case_spec}")
else
    case_values=(6)
fi
datasets=(MTCC KTCC KVIC BMTL DCCC UGC)
if [ -n "${APPLICATION_K_VALUES:-}" ]; then
    mapfile -t k_values < <(expand_values "${APPLICATION_K_VALUES}")
elif [ -n "${cli_k_spec}" ]; then
    mapfile -t k_values < <(expand_values "${cli_k_spec}")
else
    k_values=(1 2 3 4 5 6 7 8 9 10)
fi

submit_case_job() {
    local casenum=$1
    local k=$2
    local case_name=${datasets[$((casenum - 1))]}
    local job_output_root="${HPC_LOG_ROOT}/${case_name}/hpc"
    local job_name="case_${casenum}_${k}_multi"
    local job_id=""
    mkdir -p "${job_output_root}"

    if squeue --clusters="${CLUSTER_NAME}" -u "${USER}" -o "%j" -h | grep -q "${job_name}$"; then
        echo "job '${job_name}' exists" >&2
        job_id=$(squeue --clusters="${CLUSTER_NAME}" -u "${USER}" -o "%A %j" -h | awk -v name="${job_name}" '$2 == name {print $1; exit}')
    else
        local job_script="job_${job_name}.slurm"
        cat > "${job_script}" <<EOF
#!/bin/bash -l
#SBATCH --partition=${JOB_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01-00:00:00
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_output_root}/${job_name}.out
#SBATCH --error=${job_output_root}/${job_name}.err

export R_PROFILE_USER=/dev/null
export R_ENVIRON_USER=/dev/null
export R_LIBS=${PACKAGE_LIB}:${RENV_LIB}
export R_LIBS_USER=${RENV_LIB}
export APPLICATION_PACKAGE_LIB=${PACKAGE_LIB}
export APPLICATION_FILE_TAG=${APPLICATION_FILE_TAG}
export APPLICATION_RUN_ROOT=${RUNS_ROOT}
cd "${SCRIPT_DIR}"
"${RENV_RSCRIPT}" --vanilla benchmark_application.R ${casenum} ${k}
EOF
        job_id=$(sbatch --clusters="${CLUSTER_NAME}" -A "${ACCOUNT}" "${job_script}" | awk '{print $4}')
        rm -f "${job_script}"
    fi

    if [ -n "${job_id:-}" ]; then
        printf '%s' "${job_id}"
    fi
}

submit_summary_job() {
    local casenum=$1
    local case_name=${datasets[$((casenum - 1))]}
    local dependency=${2:-}
    local tag_file_hint=${3:-}
    local k_values_string=${4:-}
    local job_output_root="${HPC_LOG_ROOT}/${case_name}/hpc"
    local summary_job="summary_case_${casenum}"
    local summary_script="summary_${summary_job}.slurm"
    mkdir -p "${job_output_root}"

    cat > "${summary_script}" <<EOF
#!/bin/bash -l
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=${summary_job}
#SBATCH --output=${job_output_root}/${summary_job}.out
#SBATCH --error=${job_output_root}/${summary_job}.err
EOF
    if [ -n "${dependency}" ]; then
        echo "#SBATCH --dependency=${dependency}" >> "${summary_script}"
    fi

    cat >> "${summary_script}" <<EOF

export R_PROFILE_USER=/dev/null
export R_ENVIRON_USER=/dev/null
export R_LIBS=${PACKAGE_LIB}:${RENV_LIB}
export R_LIBS_USER=${RENV_LIB}
export RENV_RSCRIPT=${RENV_RSCRIPT}
export APPLICATION_RUN_ROOT=${RUNS_ROOT}
export APPLICATION_CASE_NAME=${case_name}
export APPLICATION_K_VALUES="${k_values_string}"
EOF

    if [ -n "${tag_file_hint}" ]; then
        echo "export APPLICATION_FILE_TAG_HINT=${tag_file_hint}" >> "${summary_script}"
    fi

cat >> "${summary_script}" <<'EOF'
"${RENV_RSCRIPT}" --vanilla - <<'RS'
run_root <- Sys.getenv("APPLICATION_RUN_ROOT")
case_name <- Sys.getenv("APPLICATION_CASE_NAME")
k.all <- as.integer(strsplit(Sys.getenv("APPLICATION_K_VALUES"), "\\s+")[[1]])
tag_hint <- Sys.getenv("APPLICATION_FILE_TAG_HINT", "")
if (nzchar(tag_hint) && !startsWith(tag_hint, "_")) {
  tag_hint <- paste0("_", tag_hint)
}

if (!nzchar(run_root) || !nzchar(case_name)) {
  stop("Missing APPLICATION_RUN_ROOT or APPLICATION_CASE_NAME")
}

savedir <- run_root
temp_dir <- file.path(savedir, "temp", case_name)
result_dir <- file.path(savedir, "result")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

candidate_files <- if (nzchar(tag_hint)) {
  file.path(temp_dir, paste0(case_name, tag_hint, "_k=", k.all[1], ".csv"))
} else {
  list.files(temp_dir, pattern = paste0("^", case_name, ".*_k=", k.all[1], "\\.csv$"), full.names = TRUE)
}

candidate_files <- candidate_files[file.exists(candidate_files)]
if (length(candidate_files) == 0L) {
  stop("No application temp files found in: ", temp_dir)
}

if (!nzchar(tag_hint)) {
  info <- file.info(candidate_files)
  candidate_files <- candidate_files[order(info$mtime, info$size, decreasing = TRUE)]
}

first_file <- basename(candidate_files[1])
tag <- sub(paste0("^", case_name), "", first_file)
tag <- sub("_k=.*$", "", tag)

files <- file.path(temp_dir, paste0(case_name, tag, "_k=", k.all, ".csv"))
files <- files[file.exists(files)]
if (length(files) == 0L) {
  stop("No application temp files found for tag: ", tag)
}

read_one <- function(path) {
  data <- read.csv(path, check.names = FALSE)
  if (ncol(data) > 0L && grepl("^Unnamed", names(data)[1])) {
    data <- data[, -1, drop = FALSE]
  }
  data
}

result <- do.call(rbind, lapply(files, read_one))
result <- result[order(result$k, result$model), , drop = FALSE]

full_file <- file.path(result_dir, paste0(case_name, tag, ".csv"))
write.csv(result, full_file, row.names = FALSE)

cat("Wrote summary:", full_file, "\n")
RS
EOF

    sbatch --clusters="${CLUSTER_NAME}" -A "${ACCOUNT}" "${summary_script}"
    rm -f "${summary_script}"
}

install_ecslrx_package

for casenum in "${case_values[@]}"; do
    summary_job_ids=()
    for k in "${k_values[@]}"; do
        job_id="$(submit_case_job "${casenum}" "${k}")"
        if [ -n "${job_id}" ]; then
            summary_job_ids+=("${job_id}")
        fi
    done

    dependency=""
    if [ "${#summary_job_ids[@]}" -gt 0 ]; then
        dependency="afterok:$(IFS=:; echo "${summary_job_ids[*]}")"
    fi

    submit_summary_job "${casenum}" "${dependency}" "${APPLICATION_FILE_TAG}" "${k_values[*]}"
done
