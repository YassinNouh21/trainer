# TRL PoC — does the TRL CLI work with the Torch plugin?

Answers the action item *"check that the TRL CLI is going to work with the torch
plugin"*, scoped on the community call as a PoC with no SDK changes. There is no
Go in this branch — only a Dockerfile, an adapter, and a runtime manifest.

**Short answer:** yes, but only with an adapter, and a naive adapter fails
*silently*. The plugin itself needs no changes.

## The question

With `mlPolicy.torch` set, the Torch plugin injects `PET_NNODES`,
`PET_NPROC_PER_NODE`, `PET_NODE_RANK`, `PET_MASTER_ADDR` and `PET_MASTER_PORT`
for `torchrun` to consume. TRL uses its own CLI. Does it recognise them?

## Answer: no, and it fails silently

`trl <method>` delegates to `accelerate launch`, and accelerate resolves
topology from **flags, not the environment** — `accelerate/commands/launch.py`
has no `os.environ` reads for topology. Its defaults:

| Flag | Default | Source |
|---|---|---|
| `--num_processes` | `torch.cuda.device_count()` | `launch.py:1317` |
| `--num_machines` | `1` | `launch.py:1340` |

Measured with `PET_NNODES=2 PET_NPROC_PER_NODE=4` and nothing else:

```
num_processes=1  process_index=0  distributed_type=DistributedType.NO
RANK=None        WORLD_SIZE=None
```

So single-node multi-GPU happens to work, because accelerate auto-detects local
GPUs and coincidentally lands on the right number. Multi-node is **silently
wrong**: every pod trains independently, with no error and no hang.

## What makes it work

`trl-launch` translates the plugin's environment into
`--num_processes/--num_machines/--machine_rank/--main_process_ip/--main_process_port`
and execs the TRL CLI. TRL forwards flags its own parser does not recognise
through to accelerate, which is what lets this work without any control-plane
change. Details it has to get right:

- `--num_processes` is the **total across all machines**, while
  `PET_NPROC_PER_NODE` is per-node — so it multiplies by `PET_NNODES`.
- the plugin leaves `numProcPerNode` as the literal `"auto"` when GPUs are
  requested (`torch.go:121,133`). `torchrun` accepts that; accelerate does not.
- **`--multi_gpu` must be passed explicitly.** Accelerate auto-enables the
  multi-GPU path only when the *local* process sees `torch.cuda.device_count() > 1`.
  At one GPU per pod it never infers it, so without the flag every pod runs at
  world size 1 — and exits 0.
- **CPU multi-node is impossible through accelerate.** Every distributed branch
  in `launch_command` is guarded `and not args.cpu`, so `--use_cpu` falls
  through to `simple_launcher`, which never sets `RANK`/`WORLD_SIZE`. The
  adapter refuses the combination rather than producing N isolated trainings.

## Verified on a cluster

The rendered JobSet showed the plugin injecting all five `PET_*` variables and
port 29500, leaving `command` untouched, and the adapter producing:

```
[trl-launch] trl sft --multi_gpu --num_processes 2 --num_machines 2 --machine_rank 0 \
  --main_process_ip <job>-node-0-0.<job> --main_process_port 29500 ...
```

Both branches of the `numProcPerNode` logic were exercised: `"auto"` with GPUs
requested, and a plain integer derived from the CPU request without them.

### The rendezvous, and the silent failure

Two jobs were run on two CPU nodes, differing only in what consumed `PET_*`:

| Job | Consumer | Result |
|---|---|---|
| `trl-demo` | `trl sft` → accelerate, first version of the adapter | Flags translated **correctly** — `machine_rank` 0 and 1, matching `main_process_ip` — yet `epoch` advanced `1/52002` per step instead of `2/52002`. **World size 1.** Both pods `Succeeded`, TrainJob `Complete`. |
| `probe-rendezvous` | bare `torchrun` reading `PET_*` | `[Gloo] Rank 1 is connected to 1 peer ranks`, `rank=1 world_size=2 allreduce=2.0`. **Real rendezvous.** |

The probe is a bare `torchrun` over a script that joins a Gloo process group and
all-reduces, each rank contributing `1.0`. A sum of `2.0` can only come from two
processes that actually exchanged data, so it cannot be faked.

That pair is the whole finding. The plugin's `PET_*` is correct and sufficient —
torchrun proves it on the same configuration. What fails is accelerate, which
does not read `PET_*` at all, and it fails **without an error**. The first
version of this adapter translated the flags perfectly and still produced two
isolated trainings reported as one distributed job. The three guards in
`trl-launch` exist because of that run.

Design write-up and the equivalent analysis for Axolotl, LlamaFactory, and
Unsloth: [`proposals/2839-dynamic-llm-trainer/README.md`](../../proposals/2839-dynamic-llm-trainer/README.md), Part I.

## Layout

```
cmd/trainers/trl/Dockerfile          # trl + accelerate + peft on the torchtune base
cmd/trainers/trl/trl-launch          # PET_* -> accelerate flags; no training logic
manifests/base/runtimes/trl/         # ClusterTrainingRuntime, framework: trl
examples/trl/trl-trainjob.yaml       # plain TrainJob
```

## Running it

The runtime points at `ghcr.io/kubeflow/trainer/trl-trainer`, which is not
published yet — build and push it to a registry your cluster can reach, and
update the image in the runtime to match.

```bash
docker build -f cmd/trainers/trl/Dockerfile -t <your-registry>/trl-trainer:poc .
docker push <your-registry>/trl-trainer:poc

kubectl apply -k manifests/base/runtimes/trl/
kubectl apply -f examples/trl/trl-trainjob.yaml
```

`TRL_SKIP_ACCELERATE_ARGS=1` bypasses the translation and is the control run —
the literal "out of the box" measurement. Run it first, then the translated one,
and compare `Num processes` in accelerate's startup banner against the GPUs
requested.

```bash
kubectl get jobset trl-sft-qwen -o yaml   # env injected, command not rewritten
kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f
```

## Comparability

The runtime mirrors `manifests/base/runtimes/torchtune/qwen2_5/qwen2_5_1.5B.yaml`
— same model, dataset and initializers — so TRL is the only variable. It keeps
torchtune's one-runtime-per-base-model granularity, which is the proposed answer
to the second action item.

## Known gaps

- No entry in `.github/workflows/build-and-push-images.yaml`.
- No Helm packaging; that belongs with the runtime-release write-up.
- TRL defaults to bf16, so a CPU-only run needs `--use_cpu --bf16=False`.
- Which accelerate parameters are reachable this way is still open. Topology
  flags work; FSDP and DeepSpeed normally arrive via an accelerate config file,
  which the runtime would have to mount.
