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

# Installed on PATH, not referenced by absolute path, so a runtime's command is
# ["trl-launch"] — a name the torch plugin can match on, exactly as it matches
# ["tune", "run"] for torchtune.
COPY cmd/trainers/trl/trl-launch /usr/local/bin/trl-launch
RUN chmod +x /usr/local/bin/trl-launch

ENTRYPOINT ["trl-launch"]
