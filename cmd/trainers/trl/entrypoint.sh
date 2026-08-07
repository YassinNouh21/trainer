#!/bin/sh
# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The Torch plugin sets PET_* for torchrun, but `trl` hands off to accelerate,
# which reads topology from flags only. Translate, then exec. No training logic.
# Set TRL_SKIP_ACCELERATE_ARGS=1 to skip the translation and run bare.

set -eu

METHOD="${TRL_METHOD:-sft}"

if [ "${TRL_SKIP_ACCELERATE_ARGS:-0}" = "1" ]; then
    echo "[entrypoint] trl ${METHOD} $*"
    exec trl "${METHOD}" "$@"
fi

NNODES="${PET_NNODES:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
NPROC="${PET_NPROC_PER_NODE:-1}"

# The plugin sets "auto" when GPUs are requested; accelerate needs a number.
if [ "${NPROC}" = "auto" ]; then
    PY=$(command -v python || command -v python3)
    NPROC=$("${PY}" -c 'import torch; print(torch.cuda.device_count() or 1)')
fi

# --num_processes is the total across machines, not per node.
set -- \
    --num_processes "$((NPROC * NNODES))" \
    --num_machines "${NNODES}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${PET_MASTER_ADDR:-localhost}" \
    --main_process_port "${PET_MASTER_PORT:-29500}" \
    "$@"

echo "[entrypoint] trl ${METHOD} $*"
exec trl "${METHOD}" "$@"
