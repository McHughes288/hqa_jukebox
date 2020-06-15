#!/bin/bash -eux

GPUQ="-q gpu.q@cam2aml01.aml.speechmatics.io"
VENV=/cantab/dev/inbetweeners/hydra/venv_stable/bin/activate
train_data=/perish/data/hqa_audio/train_npr_03022020.dbl
val_data=/perish/data/hqa_audio/test_npr_03022020.dbl
CUDA_WRAPPER=./scripts/cuda_wrapper
CODE_DIR=${HOME}/git/hydra
EXPNAME=$(basename $BASH_SOURCE)

###############
## set here

seed=1
priority=500
WORK_ROOT=/cantab/dev/inbetweeners/hydra/${USER}_body_${EXPNAME}_s${seed}

amp="O1"
n_gpus=4

msg="Run from 44khz with commitment loss"
prev_model=
checkpoint=

window_size=16384
batch_size=4  # per gpu
minimum_batch_size=2
steps=150000

enc_strides=2
codebook_groups=1

codebook_slots=512
codebook_dim=128

entropy_beta=5e-5
commit_beta=8e-2

enc_hidden_dim=128
dec_n_residual=128
dec_n_skip=128
dec_dilation_depth=10
dec_n_repeat=3

learning_rate=4e-4
decay_temp=True
temp_decay_proportion=0.4
gs_temp=0.4

## end set
###############
prev_job_id=

mkdir -p "${WORK_ROOT}"/
( cd $CODE_DIR && echo "$(date -u) $(git describe --always --abbrev=40 --dirty)")>> "${WORK_ROOT}"/git_sha
rsync --quiet -avhz --exclude "apis" --exclude "data_edgecase" --exclude "best_models" --exclude "models" --exclude "functests" --exclude "*ipynb*" --exclude "build" --exclude "venv" --exclude "unittests" --exclude "logs" --exclude ".git" --exclude "**/__pycache__" --exclude ".pytest_cache" --exclude "exp" --exclude "htmlcov" "$CODE_DIR"/ "${WORK_ROOT}"/code

for layer in 5 6; do

    case $layer in
    0) prev_model=
       window_size=16384
       checkpoint=
       ;;
    1) prev_model=/cantab/dev/inbetweeners/hydra/samr_body_44khz_commitment_loss.sh_s1/20200508_44khz_commitment_loss.sh_l0/model.pt
       window_size=16384
       checkpoint=/cantab/dev/inbetweeners/hydra/samr_body_44khz_exp_decay_temp.sh_s1/20200508_44khz_exp_decay_temp.sh_l1/checkpoint.pt.step140000
       ;;
    2) prev_model=${WORK_ROOT}/layer1.pt
       window_size=32768
       checkpoint=
       ;;
    3) prev_model=${WORK_ROOT}/layer2.pt
       window_size=65536
       checkpoint=
       ;;
    4) prev_model=${WORK_ROOT}/layer3.pt
       window_size=131072
       checkpoint=
       ;;
    5) prev_model=${WORK_ROOT}/layer4.pt
       window_size=262144
       checkpoint=
       ;;
    6) prev_model=${WORK_ROOT}/layer5.pt
       window_size=524288
       checkpoint=
       ;;
    esac

    WORK_DIR=${WORK_ROOT}/20200508_${EXPNAME}_l${layer}
    if [[ -f "${WORK_DIR}/model.pt" ]]; then
        echo "${WORK_DIR} is already done. Skipping!"
        continue
    fi
    mkdir -p "$WORK_DIR"
    touch ${WORK_DIR}/results.log
    echo "${prev_model}" > ${WORK_DIR}/prev_model

        cat <<EOF >"${WORK_DIR}"/launch.qsh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ $GPUQ
#$ -terse
#$ -w e
#$ -wd $WORK_ROOT/code
#$ -pe local $n_gpus
#$ -p $priority
#$ -notify
#$ -o ${WORK_DIR}/results.log
#$ -sync n
#$ -N "l${layer}"
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
echo "pstree_pid:  \$\$"
echo

echo "\$(date -u) starting \${JOB_ID}" >> ${WORK_DIR}/sge_job_id
export CUDA_HOME=/usr/local/cuda-10.1/
source $VENV
pip3 freeze

$CUDA_WRAPPER $n_gpus time python3.7 -m body.research.hqa_audio.train \
    --expdir=${WORK_DIR} \
    --window_size=${window_size} \
    --enc_strides=${enc_strides} \
    --prev_model=${prev_model} \
    --train_data=${train_data} \
    --val_data=${val_data} \
    --log_tb_viz_every=1000 \
    --log_tb_every=100 \
    --batch_size=${batch_size} \
    --codebook_slots=${codebook_slots} \
    --codebook_dim=${codebook_dim} \
    --codebook_groups=${codebook_groups} \
    --seed=${seed} \
    --enc_hidden_dim=${enc_hidden_dim} \
    --dec_n_residual=${dec_n_residual} \
    --dec_n_skip=${dec_n_skip} \
    --dec_dilation_depth=${dec_dilation_depth} \
    --dec_n_repeat=${dec_n_repeat} \
    --decay_temp=${decay_temp} \
    --steps=${steps} \
    --minimum_batch_size=${minimum_batch_size} \
    --amp=${amp} \
    --save_every=10000 \
    --val_every=5000 \
    --gs_temp=${gs_temp} \
    --lr=${learning_rate}  \
    --entropy_beta=${entropy_beta} \
    --commit_beta=${commit_beta} \
    --checkpoint=${checkpoint} \
    --temp_decay_proportion=${temp_decay_proportion} \
    --checkpoint_autoload \
&& sleep 3s \
&& touch ${WORK_DIR}/done \
&& ln -s ${WORK_DIR}/model.pt ${WORK_ROOT}/layer${layer}.pt \
&& echo "Done"
EOF
    echo "Launching layer $layer"
    chmod +x "${WORK_DIR}"/launch.qsh
    job_id=$(qsub -hold_jid "${prev_job_id:-\'\'}" "${WORK_DIR}/launch.qsh";)
    prev_job_id=${job_id}
done

echo "All launced"
