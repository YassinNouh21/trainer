# TRL PoC — does TRL work with the Torch plugin?

Answers the action item *"check that the TRL CLI is going to work with the torch
plugin"*, scoped on the community call as a PoC with no SDK changes. There is no
Go in this branch — only a Dockerfile, a runtime manifest, and a proof script.

**Short answer:** yes — by launching TRL's training script with **torchrun**,
which consumes the plugin's `PET_*` natively. Zero translation, zero adapter,
zero plugin changes. The TRL *CLI* (which hands off to `accelerate launch`)
is bypassed and goes under limitations.

## The design

With `mlPolicy.torch` set, the Torch plugin injects `PET_NNODES`,
`PET_NPROC_PER_NODE`, `PET_NODE_RANK`, `PET_MASTER_ADDR` and `PET_MASTER_PORT`.
Those are torchrun's own env interface (`torch.distributed.argparse_util`
derives `PET_{DEST}` for every flag), so the runtime command is simply:

```yaml
command: [torchrun, -m, trl.scripts.sft]
args: [--model_name_or_path=..., --dataset_name=..., ...]
```

This works because of what each layer is:

- `trl/scripts/sft.py` is a **plain training script** — `TrlParser` argument
  parsing plus `SFTTrainer(...).train()`, no launcher logic. The `trl` CLI
  itself is only a wrapper that resolves this same file and hands it to
  accelerate's `launch_command` (`trl/cli/accelerate_launcher.py`).
- torchrun reads `PET_*`, spawns the script, and exports the standard
  `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `MASTER_ADDR` / `MASTER_PORT`.
- In-process accelerate (the library transformers depends on — not the
  launcher) detects that env in `PartialState`: `LOCAL_RANK` set → multi-GPU
  (nccl); `WORLD_SIZE > 1` on CPU → multi-CPU (gloo).

Other post-training methods are just the module name: `trl.scripts.dpo`,
`trl.scripts.grpo`, `trl.scripts.kto` — one runtime manifest per method/model,
no new mechanism.

## Verified practically

`local_torchrun_proof.sh` (in this directory, needs only docker) runs TRL's
real `sft.py` in a Linux container as two bare `torchrun -m trl.scripts.sft`
launches — the exact command in the runtime manifest — with **only** the five
`PET_*` variables set. No topology flags, no CLI. Observed
(trl 1.10.0 / torch 2.13 / transformers 5.15):

| Run | First-step epoch | Meaning |
|---|---|---|
| control, `PET_NNODES=1` | `0.05882` = 1/17 | world size 1 |
| two nodes, `PET_NNODES=2` | `0.1111` = 1/9 | **world size 2** — the 17-example dataset sharded to 9 steps/epoch, which only happens when the sampler sees 2 replicas |

Both node processes reached `Training completed`; a rank that failed to
rendezvous would hang, not complete. The run used `--use_cpu`, so this needs no
GPUs to reproduce.

On the cluster, `probe-rendezvous` (bare torchrun reading the plugin's actual
injected `PET_*` across two pods) reached `world_size=2 allreduce=2.0`,
confirming the plugin→torchrun half on real infrastructure. The remaining step
is the combined run — this image, two pods — once cluster access is back.

## Limitation: the TRL CLI / `accelerate launch`

The TRL CLI cannot be used as the runtime command. `trl sft` forwards to
`accelerate launch`, and accelerate-the-launcher reads topology from **flags
only** — it has no reference to `PET_*` anywhere. Measured on two cluster pods
(`trl-demo`): even with the flags translated correctly per pod, the job trained
at world size 1 on each pod and still reported `Succeeded` — a silent failure.
Two root causes in the launcher: every distributed branch is guarded
`and not args.cpu`, and `--multi_gpu` is only inferred when the *local* device
count exceeds 1 (never true at one GPU per pod).

Launching the script with torchrun sidesteps the launcher entirely, so none of
that applies. Supporting `accelerate launch` as a first-class launcher (its
config file is also the route to FSDP/DeepSpeed settings) is deferred to a
future discussion, along with any launcher-adapter mechanism.

## Layout

```
cmd/trainers/trl/Dockerfile          # trl + deps on the torchtune base; ENTRYPOINT torchrun -m trl.scripts.sft
manifests/base/runtimes/trl/         # ClusterTrainingRuntime, framework: trl
examples/trl/trl-trainjob.yaml       # plain TrainJob
examples/trl/local_torchrun_proof.sh # the docker-based proof described above
```

## Running it

```bash
sh examples/trl/local_torchrun_proof.sh   # local proof, needs only docker

docker build -f cmd/trainers/trl/Dockerfile -t <your-registry>/trl-trainer:poc .
docker push <your-registry>/trl-trainer:poc

kubectl apply -k manifests/base/runtimes/trl/
kubectl apply -f examples/trl/trl-trainjob.yaml

kubectl get jobset trl-sft-qwen -o yaml   # PET_* injected, command not rewritten
kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f
```

The world-size check in the logs: with 2 nodes the per-step `epoch` increment
must be twice the single-node value (dataset sharded across ranks). Completion
alone is not evidence — see the limitation above.

## Comparability

The runtime mirrors `manifests/base/runtimes/torchtune/qwen2_5/qwen2_5_1.5B.yaml`
— same model, dataset and initializers — so TRL is the only variable, and keeps
torchtune's one-runtime-per-base-model granularity.

## Known gaps

- No entry in `.github/workflows/build-and-push-images.yaml`; no Helm packaging.
- Proven locally with trl 1.10.0; the image pins 1.9.2 (same `trl/scripts/`
  layout) — re-run the proof when bumping the pin.
- FSDP/DeepSpeed configs normally arrive via an accelerate config file; that
  surface is part of the deferred accelerate discussion.
