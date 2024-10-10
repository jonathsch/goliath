#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables

CONFIG_FILE=config/rgca_example.yml
TIME='3-00:00:00'


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

JOB_NAME=RGCA_${SID}
RUN_ID=$(date '+%Y-%m-%d_%H-%M-%S')
CKPT_DIR=outputs/goliath/RGCA/${SID}_${RUN_ID}/

###-----------------------------------------------------------------
# Create a temporary script file
SCRIPT=$(mktemp)

# Write the SLURM script to the temporary file
cat > $SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=submit
#SBATCH --job-name=train_goliath
#SBATCH --nodes=1
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=rtx_a6000:1
#SBATCH --output=./slurm_out_%x_%j.txt

source /rhome/jschmidt/.bashrc
source activate rgca
cd ${ROOT_DIR}

mkdir -p ${CKPT_DIR}

nvidia-smi

# Run training
srun python -m ca_code.scripts.run_train \
    ${CONFIG_FILE} \
    sid=${SID} \
    train.run_id=${RUN_ID} \
    data.root_path=${DATA_ROOT} \
    test.test_path=${CKPT_DIR}

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
