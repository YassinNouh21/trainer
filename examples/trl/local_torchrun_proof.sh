#!/bin/sh
# Proof: TRL's sft script runs distributed under bare `torchrun`, driven ONLY by
# the PET_* env vars the Kubeflow torch plugin injects. No topology flags, no
# `trl` CLI, no `accelerate launch`, no adapter, no plugin change.
#
# This is the torchrun-first direction agreed on the SDK call: bypass the TRL
# CLI (which is hardwired to accelerate's launcher) and launch it as `torchrun -m
# trl.scripts.sft` — the runtime manifest's exact command. torchrun reads PET_*
# natively and exports RANK/WORLD_SIZE to the script, where in-process
# accelerate picks them up (PartialState env detection).
#
# Runs in a Linux container (the pods' environment; macOS breaks on an MPS
# quirk unrelated to any of this). Simulates two Kubeflow pods as two torchrun
# launches with exactly the five variables the plugin sets. Requires docker.
#
# The world-size fingerprint: transformers computes epoch = step/steps_per_epoch,
# and the 17-example dataset shards 17 -> 9 steps/epoch only if the sampler sees
# num_replicas=2. So first-step epoch 0.0588 (1/17) means world 1, and 0.1111
# (1/9) means a real world of 2. A rank that failed to rendezvous would hang,
# not complete — completion of both concurrent launches is itself rendezvous
# proof.
#
# Observed result (2026-08-23, trl 1.10.0 / torch 2.13 / transformers 5.15):
#   control  PET_NNODES=1 : first-step epoch 0.05882  (world 1)
#   node0+1  PET_NNODES=2 : first-step epoch 0.1111   (world 2), both completed

set -eu
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat >"$WORK/inner.sh" <<'INNER'
set -eu
cd /w
export PIP_CACHE_DIR=/w/.pipcache HF_HOME=/w/.hf
pip install -q trl peft 2>&1 | tail -1 || true
python -c "import trl, torch; print('versions: trl', trl.__version__, '| torch', torch.__version__)"
mkdir -p logs

# Tiny random model + tiny dataset from TRL's own test fixtures: a few MB,
# a CPU step takes well under a second.
TRAIN_ARGS="--model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
  --dataset_name trl-internal-testing/zen \
  --dataset_config standard_language_modeling \
  --per_device_train_batch_size 1 \
  --max_steps 4 \
  --logging_steps 1 \
  --report_to none \
  --use_cpu"

run_node() {  # $1=nnodes $2=node_rank $3=log $4=outdir
    PET_NNODES="$1" \
    PET_NODE_RANK="$2" \
    PET_NPROC_PER_NODE=1 \
    PET_MASTER_ADDR=127.0.0.1 \
    PET_MASTER_PORT=29500 \
    torchrun -m trl.scripts.sft $TRAIN_ARGS --output_dir "logs/$4" >"logs/$3" 2>&1
}

echo "=== control: single node (PET_NNODES=1) ==="
run_node 1 0 control.log out-control
grep -q "Training completed" logs/control.log && echo "control: completed"

echo "=== proof: two nodes, PET_* env only, zero flags ==="
run_node 2 1 node1.log out-n1 &
N1=$!
run_node 2 0 node0.log out-n0
wait "$N1"
grep -q "Training completed" logs/node0.log && echo "node0: completed"
grep -q "Training completed" logs/node1.log && echo "node1: completed"

echo "--- first-step epoch: control (0.0588 = 1/17 means world 1) ---"
grep -o "'epoch': '[0-9.]*'" logs/control.log | head -1
echo "--- first-step epoch: node0 (0.1111 = 1/9 means world 2) ---"
grep -o "'epoch': '[0-9.]*'" logs/node0.log | head -1
# HF Trainer logs metrics only on rank 0, so node1 has no loss lines by design;
# its completion (instead of a rendezvous hang) is its evidence.
echo "PROOF-DONE"
INNER

docker run --rm -v "$WORK":/w python:3.11-slim sh /w/inner.sh
