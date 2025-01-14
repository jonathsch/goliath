#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables

CONFIG_FILE=config/rgca_becominglit_nvs.yml
TIME='3-00:00:00'


###-----------------------------------------------------------------

ROOT_DIR=/rhome/jschmidt/projects/goliath

declare -a SIDS=(
    "'1001'"
)
declare -a DATA_ROOTS=(
    "/cluster/valinor/jschmidt/becominglit"
)

###-----------------------------------------------------------------
# Loop over for creation of runs
for (( i=0; i<"${#SIDS[@]}"; i++ )); do

SID="${SIDS[i]}"
SEQUENCE="EXP-2"
DATA_ROOT="${DATA_ROOTS[i]}"

JOB_NAME=RGCA_BL_${SID}
RUN_ID=ALL_NVS_$(date '+%Y-%m-%d_%H-%M-%S')
# CKPT_DIR=/cluster/pegasus/jschmidt/logs/goliath/RGCA/${SID}_${RUN_ID}/

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=submit
#SBATCH --job-name=train_rgca_nvs
#SBATCH --nodes=1
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100:1
#SBATCH --output=./slurm_logs/slurm_out_%x_%j.txt

source /rhome/jschmidt/.bashrc
source activate rgca
cd ${ROOT_DIR}

nvidia-smi

# Run training
srun python -m ca_code.scripts.run_train \
    ${CONFIG_FILE} \
    train.run_id=${RUN_ID} \
    data.root_path=${DATA_ROOT} \
    sid=${SID}
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
