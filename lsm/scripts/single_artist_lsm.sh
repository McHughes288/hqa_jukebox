#!/bin/bash -eux

# assumes we are on cam2aml01.aml.speechmatics.io
GPUQ="-q gpu.q"
VENV=/cantab/dev/inbetweeners/hydra/venv_stable
train_data=/perish/data/music/train.dbl
val_data=/perish/data/music/val.dbl
CUDA_WRAPPER=/usr/bin/cuda_wrapper
CODE_DIR=${HOME}/git/hqa_jukebox
EXPNAME=$(basename $BASH_SOURCE)
WORK_ROOT=/cantab/dev/inbetweeners/hqa_jukebox/exp/${USER}_body_${EXPNAME}

###############
## set here
amp="O1"
n_gpus=1

msg="Single artist lsm"
hqa_path=/cantab/dev/inbetweeners/hqa_jukebox/exp/johnh_body_full_stack.sh/20200615_full_stack.sh_l3/model.pt
checkpoint=

window_size=32000
batch_size=16  # per gpu
minimum_batch_size=16
steps=400000
seed=1

hidden_dim=256
num_layers=3

learning_rate=4e-4

## end set
###############

for layer in 3; do
    # hqa_path=${hqa_root}_l${layer}/model.pt
    # WORK_DIR=${WORK_ROOT}/$(date +"%Y%m%d")_${EXPNAME}_ws${window_size}_h${hidden_dim}_l${num_layers}_hqa${layer}

    mkdir -p "$WORK_DIR"
    rsync --quiet -avhz --exclude "*ipynb*" --exclude "venv" --exclude ".git" --exclude "**/__pycache__" "$CODE_DIR"/* "$WORK_DIR"/code

    cat <<EOF >"${WORK_DIR}"/launch.qsh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ $GPUQ
#$ -terse
#$ -w e
#$ -wd $WORK_DIR/code
#$ -pe local $n_gpus
#$ -p $priority
#$ -notify
#$ -o ${WORK_DIR}/results.log
#$ -sync n
#$ -N "jb_lsm_${layer}"
set -e pipefail;

# job info
hostname && date
echo
echo "sge_job_id:  \${JOB_ID}"
echo "sge_queue:   \${QUEUE}"
echo "user:        \${USER}"
echo "sge_tmp_dir: \${TMPDIR}"
echo "sge_request: \${REQUEST}"
echo "reason:      ${msg}"
echo "sge_wd:      \$(pwd)"
echo

function cleanup {
  err=\$?
  echo "\$(date) Received code \${err}. Cleaning up..."
  rm -f ${WORK_DIR}/inflight ${WORK_DIR}/done
  exit \$err
}
trap cleanup EXIT INT QUIT TERM

echo "\$(date -u) starting \${JOB_ID}" >> ${WORK_DIR}/sge_job_id

echo "\${JOB_ID}" > ${WORK_DIR}/inflight \
&& cd $WORK_DIR/code \
&& VIRTUAL_ENV_DISABLE_PROMPT=true source $VENV/bin/activate \
&& pip3 freeze &> ${WORK_DIR}/pip_freeze.log

# export CUDA_HOME=/usr/local/cuda-10.1/

$CUDA_WRAPPER $n_gpus time python3.7 -m lsm.train \
    --expdir=${WORK_DIR} \
    --hqa_path=${hqa_path} \
    --window_size=${window_size} \
    --train_data=${train_data} \
    --val_data=${val_data} \
    --log_tb_viz_every=1000 \
    --batch_size=${batch_size} \
    --seed=${seed} \
    --n_gpus=${n_gpus} \
    --steps=${steps} \
    --minimum_batch_size=${minimum_batch_size} \
    --amp=${amp} \
    --save_every=10000 \
    --val_every=5000 \
    --hidden_dim=${hidden_dim} \
    --num_layers=${num_layers} \
    --lr=${learning_rate}  \
&& sleep 1s \
&& touch ${WORK_DIR}/done \
&& echo "Done"
EOF
    if [[ -f "$WORK_DIR/done" ]]; then
        >&2 echo "${WORK_DIR} is already done. Skipping!"
    else
        chmod +x "${WORK_DIR}"/launch.qsh
        echo "Launching: ${WORK_DIR}/launch.qsh"
        echo "Job launched"
    fi
done
echo "Done"
