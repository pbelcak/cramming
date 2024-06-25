# interactive dbg sesh
srun --account=nvr_lpr_llm --partition=interactive,grizzly,polar,polar2,polar3,polar4  --time=04:00:00 --container-image nvcr.io/nvidia/pytorch:23.11-py3 -n 1 --gpus 8 --cpus-per-gpu 16 --container-mounts=$HOME:/home,/lustre:/lustre --pty /bin/bash
# THEN
cd $CRAM
source cluster/prepare_job.sh
source cluster/secrets.sh

PROJECT_PATH=$CRAM

# interactive dry run to test dependencies
I=0
JOB_NAME=test_DRY_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} python pretrain.py name=${JOB_NAME} arch=hf-bert-base train=bert-base data=sanity-check-2 dryrun=True impl.microbatch_size=2

# interactive bert reproduction dry run
I=0
JOB_NAME=A6000amp_b512_bert_base_original_final_DRY_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node=8 --standalone pretrain.py name=${JOB_NAME} dryrun=True wandb.tags=[bertbase,original,final,bookcorpus] arch=hf-bert-base train=bert-original impl.microbatch_size=256 data=bookcorpus-wikipedia

# interactive cramming-bert dry run
I=0
JOB_NAME=A6000amp_b8192_cb_o4_final_DRY_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} python pretrain.py name=${JOB_NAME} dryrun=True wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=512 data=pile-readymade

# interactive cramming-bert run
I=0
JOB_NAME=A6000amp_b8192_cb_o4_final_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} python pretrain.py name=${JOB_NAME} wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=512 data=pile-readymade