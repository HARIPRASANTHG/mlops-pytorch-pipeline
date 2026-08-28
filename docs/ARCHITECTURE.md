# Architecture

Open [`architecture.html`](architecture.html) for the interactive version — click
a node to see its files, commands and configuration, hover to trace its edges,
and use the stage chips to isolate one layer.

## Stages

**1. Local dev (Windows 11 + RTX 5070 Ti).** `src/` holds the four modules every
later stage reuses. `configs/training_config.yaml` is the single source of truth
for hyperparameters. Training runs on the GPU through the cu128 PyTorch build
(sm_120 kernels) with AMP, and writes `checkpoints/classifier_v1.pt` plus
`metrics.json`. `scripts/export_test_images.py` exports 200 labelled test PNGs
for the serving UI.

**2. Git + CI.** Seven feature branches merge into `develop` via PRs, and
`develop` is released to `main` weekly. GitHub Actions runs ruff, pytest, both
image builds, and `kubeconform` over `k8s/`.

**3. Docker.** `Dockerfile.train` is multi-stage — deps into `/opt/venv` in stage
one, only the venv plus `src/` and `configs/` in stage two — and runs as a
non-root user with volumes for data and checkpoints. `Dockerfile.serve` installs
inference deps only, runs gunicorn as UID 1001 on port 8080, and self-probes with
a `HEALTHCHECK`. Both images are side-loaded into Minikube with
`minikube image load`.

**4. Kubernetes.** Namespace `ml-training` holds a ConfigMap (mounted read-only
at `/app/configs`), two PVCs, the training Job (2 CPU / 4Gi), the serving
Deployment (2 replicas, probes on `/health`, `maxUnavailable: 0`), a ClusterIP
Service on 80 → 8080, and an HPA scaling 2 → 6 at 70% CPU.

## Data and control flow

```
config + code ──► GPU training run ──► checkpoint
      │                                    │
      └──► git ──► CI ──► images ──► minikube image load
                                        │
        ConfigMap ──► Job ──► PVC(checkpoints) ──► Deployment ──► Service ──► client
                       └───► PVC(data cache)  ──────────┘
```

The checkpoint PVC is the only coupling between training and serving: the Job
writes it read-write, the Deployment mounts the same claim read-only. Nothing
else crosses that boundary, which is why the two images can have completely
different dependency sets.

## Decisions worth defending in review

| Decision | Reason |
|---|---|
| CIFAR-adapted ResNet stem | The stock 7×7 stride-2 conv + maxpool discards most of a 32×32 image before the first residual block. |
| Two training requirement files | The in-cluster image stays CPU and small; the laptop gets cu128 wheels with sm_120 kernels. |
| `/health` re-stats the checkpoint | Serving can be deployed before training finishes — pods report 503, then become Ready with no restart. |
| `fsGroup` on both pod specs | Hostpath PVCs mount root-owned; without it the non-root container cannot write the checkpoint. |
| `maxSurge: 1`, `maxUnavailable: 0` | The endpoint list never empties during a rollout. |
| `imagePullPolicy: IfNotPresent` | Images are side-loaded, not pulled from a registry; `Always` would give `ErrImagePull`. |
| Architecture stored in the checkpoint | Serving reconstructs the right model without a rebuild or an env var. |
