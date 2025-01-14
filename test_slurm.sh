#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables

CONFIG_FILE=/cluster/pegasus/jschmidt/logs/goliath/RGCA/runs/rgca.AXE977/2024-11-27_13-28-18_5cm_perturbed_lpos/config.yml
TIME='1-00:00:00'


###-----------------------------------------------------------------

ROOT_DIR=/rhome/jschmidt/projects/goliath

declare -a SIDS=(
    "AXE977"
)
declare -a DATA_ROOTS=(
    "/cluster/pegasus/jschmidt/goliath/m--20230306--0707--AXE977--pilot--ProjectGoliath--Head"
)

###-----------------------------------------------------------------
# Loop over for creation of runs
for (( i=0; i<"${#SIDS[@]}"; i++ )); do

SID="${SIDS[i]}"
DATA_ROOT="${DATA_ROOTS[i]}"

JOB_NAME=test_RGCA_${SID}
LOG_DIR=/cluster/pegasus/jschmidt/logs/goliath/RGCA/
CKPT_DIR=/cluster/pegasus/jschmidt/logs/goliath/RGCA/${SID}_${RUN_ID}/

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=submit
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=%x_%j_%N.log

source /rhome/jschmidt/.bashrc
source activate rgca
cd ${ROOT_DIR}

mkdir -p ${CKPT_DIR}

nvidia-smi

# Run training
srun python -m ca_code.scripts.run_test \
    ${CONFIG_FILE} \
    sid=${SID} \
    data.root_path=${DATA_ROOT} \
    test.test_path=${CKPT_DIR}

srun python -m ca_code.scripts.run_vis_relight \
    ${CONFIG_FILE} \
    sid=${SID} \
    data.root_path=${DATA_ROOT} \
    test_path=${CKPT_DIR}"
EOL

###-----------------------------------------------------------------
# Print the script content in green
echo $SCRIPT
echo -e "\033[0;32m"
cat $SCRIPT
echo -e "\033[0m"

###-----------------------------------------------------------------
# Submit the job
sbatch $SCRIPT

###Optionally, remove the temporary script file
rm -f $SCRIPT

done
