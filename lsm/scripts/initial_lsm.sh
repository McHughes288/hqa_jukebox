#!/bin/bash -eux

# assumes we are on cam2aml01.aml.speechmatics.io
GPUQ="-q gpu.q"
VENV=/home/willw/venv_unstable/bin/activate
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

msg="Go back to pred z_e but with no tanh models"
run_name="save_trained"
hqa_root= #TODO
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

for layer in 4_small_decoder 4_smaller_cd 4_smaller_cg; do
    hqa_path=${hqa_root}_l${layer}/model.pt
    WORK_DIR=${WORK_ROOT}/$(date +"%Y%m%d")_${EXPNAME}_ws${window_size}_h${hidden_dim}_l${num_layers}_hqa${layer}

    if [[ -f "$WORK_DIR/done" ]]; then
        (echo >&2 "$(tput setaf 3)$WORK_DIR is already done. Skipping!$(tput sgr0)")
        continue
    fi
    if [[ -f "$WORK_DIR/inflight" ]]; then
        (echo >&2 "$(tput setaf 3)$WORK_DIR is already inflight. Skipping!$(tput sgr0)")
        continue
    fi

    (
        mkdir -p "$WORK_DIR"
        rsync --quiet -avhz --exclude "apis" --exclude "functests" --exclude "*ipynb*" --exclude "build" --exclude "venv" --exclude "unittests" --exclude "logs" --exclude ".git" --exclude "**/__pycache__" --exclude ".pytest_cache" --exclude "exp" --exclude "htmlcov" "$CODE_DIR"/* "$WORK_DIR"/code

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
echo "\${JOB_ID}" > ${WORK_DIR}/inflight

export CUDA_HOME=/usr/local/cuda-10.1/
source $VENV \
&& $CUDA_WRAPPER $n_gpus time python3.7 -m lsm.train \
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
        chmod +x "${WORK_DIR}"/launch.qsh

        set -x
        qsub -sync n -o "${WORK_DIR}"/results.log -N "${EXPNAME}" -sc directory="${WORK_DIR}" "${WORK_DIR}/launch.qsh"
        set +x
    ) &
    sleep 12
done

sleep 0.5 && wait && echo "All queued up"
