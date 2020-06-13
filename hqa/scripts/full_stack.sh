#!/bin/bash -eux

# assumes we are on cam2aml01.aml.speechmatics.io
GPUQ="-q gpu.q@cam2aml01.aml.speechmatics.io"
VENV=/cantab/dev/inbetweeners/hydra/venv_stable/bin/activate
train_data=/perish/data/music/train.dbl
val_data=/perish/data/music/hqa_val.dbl
CUDA_WRAPPER=/usr/bin/cuda_wrapper
CODE_DIR=${HOME}/git/hqa_jukebox
EXPNAME=$(basename $BASH_SOURCE)

dry_run=nodry_run
while getopts 'd' OPTIONS; do
  case $OPTIONS in
   d) dry_run=dry_run ;;
  esac
done
###############
## set here

seed=1
priority=600
WORK_ROOT=//cantab/dev/inbetweeners/hqa_jukebox/exp/${USER}_body_${EXPNAME}

# stable params
amp_level="O1"
n_gpus=4
msg="full hqa jukebox stack"

log_every=50
logviz_every=1000
val_every=5000
save_every=10000

# changable params as increase layers
window_size=16384
batch_size=4 # per gpu
minimum_batch_size=4
steps=200000
enc_strides=2,2
codebook_slots=512
codebook_dim=64
codebook_groups=2
gs_temp=0.66
decay_temp=True
temp_decay_proportion=0.7
temp_min=0.0001
enc_hidden_dim=128
enc_kernel_size=8
enc_num_layers=4
dec_n_residual=128
dec_n_skip=128
dec_dilation_depth=10
dec_n_repeat=3
learning_rate=4e-4
entropy_beta=5e-5
commit_beta=3e-2

## end set
###############
prev_job_id=


mkdir -p "${WORK_ROOT}"/
( cd $CODE_DIR && echo "$(date -u) $(git describe --always --abbrev=40 --dirty)")>> "${WORK_ROOT}"/git_sha
rsync --quiet -avhz --exclude "data_edgecase" --exclude "*ipynb*" --exclude "venv" --exclude ".git" --exclude "**/__pycache__" --exclude "htmlcov" "$CODE_DIR"/ "${WORK_ROOT}"/code

for layer in 0 1 2 3 4; do

    case $layer in
    0) prev_model=
       window_size=16000
       codebook_dim=128
       codebook_groups=1
       enc_strides=5
       dec_dilation_depth=10
       dec_n_repeat=3
       enc_kernel_size=8
       enc_num_layers=4
       ;;
    1) prev_model=${WORK_ROOT}/layer0.pt
       window_size=24000
       codebook_dim=128
       codebook_groups=2
       enc_strides=4
       dec_dilation_depth=10
       dec_n_repeat=3
       enc_kernel_size=8
       enc_num_layers=4
       ;;
    2) prev_model=${WORK_ROOT}/layer1.pt
       window_size=32000
       codebook_dim=128
       codebook_groups=3
       enc_strides=2
       dec_dilation_depth=6
       dec_n_repeat=5
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    3) prev_model=${WORK_ROOT}/layer2.pt
       window_size=48000
       codebook_dim=128
       codebook_groups=5
       enc_strides=2
       dec_dilation_depth=6
       dec_n_repeat=5
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    4) prev_model=${WORK_ROOT}/layer3.pt
       window_size=64000
       codebook_dim=128
       codebook_groups=10
       enc_strides=2
       dec_dilation_depth=6
       dec_n_repeat=5
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    4_small_decoder) prev_model=${WORK_ROOT}/layer3.pt
       window_size=64000
       codebook_dim=128
       codebook_groups=10
       enc_strides=2
       dec_dilation_depth=4
       dec_n_repeat=1
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    4_smaller_cd) prev_model=${WORK_ROOT}/layer3.pt
       window_size=64000
       codebook_dim=64
       codebook_groups=10
       enc_strides=2
       dec_dilation_depth=6
       dec_n_repeat=5
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    4_smaller_cg) prev_model=${WORK_ROOT}/layer3.pt
       window_size=64000
       codebook_dim=128
       codebook_groups=6
       enc_strides=2
       dec_dilation_depth=6
       dec_n_repeat=5
       enc_kernel_size=2
       enc_num_layers=1
       ;;
    esac

    WORK_DIR=${WORK_ROOT}/20200514_${EXPNAME}_l${layer}
    if [[ -f "${WORK_DIR}/model.pt" ]]; then
        echo "${WORK_DIR} is already done. Skipping!"
        continue
    fi
    mkdir -p "$WORK_DIR"
    echo "${prev_model}" > ${WORK_DIR}/prev_model

    if [ $dry_run == "dry_run" ]; then
      model_out=${WORK_DIR}/dry_run/model.pt
    else
      model_out=${WORK_DIR}/model.pt
    fi

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
#$ -N "LONG_5_l${layer}"
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

$CUDA_WRAPPER $n_gpus time python3.7 -m hqa.train \
    --amp=${amp_level} \
    --batch_size=${batch_size} \
    --checkpoint_autoload \
    --codebook_dim=${codebook_dim} \
    --codebook_groups=${codebook_groups} \
    --codebook_slots=${codebook_slots} \
    --dec_dilation_depth=${dec_dilation_depth} \
    --dec_n_repeat=${dec_n_repeat} \
    --dec_n_residual=${dec_n_residual} \
    --dec_n_skip=${dec_n_skip} \
    --decay_temp=${decay_temp} \
    --enc_strides=${enc_strides} \
    --enc_hidden_dim=${enc_hidden_dim} \
    --enc_kernel_size=${enc_kernel_size} \
    --enc_num_layers=${enc_num_layers} \
    --expdir=${WORK_DIR} \
    --gs_temp=${gs_temp} \
    --temp_decay_proportion=${temp_decay_proportion} \
    --temp_min=${temp_min} \
    --log_every=${log_every} \
    --log_tb_every=${logviz_every} \
    --log_tb_viz_every=${logviz_every} \
    --lr=${learning_rate} \
    --minimum_batch_size=${minimum_batch_size} \
    --prev_model=${prev_model} \
    --save_every=${save_every} \
    --seed=${seed} \
    --steps=${steps} \
    --train_data=${train_data} \
    --val_data=${val_data} \
    --val_every=${val_every} \
    --window_size=${window_size} \
    --entropy_beta=${entropy_beta} \
    --commit_beta=${commit_beta} \
    --${dry_run} \
&& sleep 3s \
&& touch ${WORK_DIR}/done \
&& ln -s $model_out ${WORK_ROOT}/layer${layer}.pt \
&& echo "Done"
EOF
    echo "Launching layer $layer"
    chmod +x "${WORK_DIR}"/launch.qsh
    job_id=$(qsub -hold_jid "${prev_job_id:-\'\'}" "${WORK_DIR}/launch.qsh";)
    prev_job_id=${job_id}
done

echo "All launched."
