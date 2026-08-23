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

# Same base as cmd/trainers/torchtune so the two runtimes are comparable.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY cmd/trainers/trl/requirements.txt .

RUN pip install -r requirements.txt

# The runtime command is torchrun invoking TRL's training script as a module
# (trl/scripts/sft.py is a plain script: TrlParser + SFTTrainer, no launcher).
# torchrun reads the PET_* env the torch plugin injects natively, so there is
# no adapter and no `accelerate launch` anywhere in the path. Other methods are
# their module name: trl.scripts.dpo, trl.scripts.grpo, ...
ENTRYPOINT ["torchrun", "-m", "trl.scripts.sft"]
