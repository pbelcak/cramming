#!/bin/sh
JOB_NAME="$@"
# replace spaces with _:
JOB_NAME=${JOB_NAME// /_}
# replace = with _: 
JOB_NAME=${JOB_NAME//=/_}
# replace / with _:
JOB_NAME=${JOB_NAME//\//_}
# replace -- with _:
JOB_NAME=${JOB_NAME//--/}
JOB_NAME=${JOB_NAME//.py/}
JOB_NAME=${JOB_NAME//config/}
JOB_NAME=${JOB_NAME//experiment_name_/}
# remove _ if it is the first character
JOB_NAME=${JOB_NAME/#_}

max_jobname_length=180
if [[ ${#JOB_NAME} -gt $max_jobname_length ]]; then
  JOB_NAME=${JOB_NAME:0:max_jobname_length}
else
  JOB_NAME=$JOB_NAME
fi

# Store the arguments in a variable
args=""
# Iterate over each argument
for arg in "$@"; do
  # Append the argument to the args variable with a space
  args="$args $arg"
done

PROJECT_PATH=$PB/lutification
CODE_PATH=${PROJECT_PATH}/experiments/vision

submit_job --gpu 8 --nodes 4 --partition=grizzly,polar,polar2,polar3,polar4 --duration 4 --autoresume_before_timelimit 30 -n $JOB_NAME --image nvcr.io/nvidia/pytorch:23.11-py3 --command 'source cluster/prepare_job.sh; source cluster/secrets.sh; PYTHONPATH='"$CODE_PATH"':'"$PROJECT_PATH"':${PYTHONPATH} torchrun --nproc_per_node $SUBMIT_GPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes $NUM_NODES --node_rank $NODE_RANK train.py '"$args" 