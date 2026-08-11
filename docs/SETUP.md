# Setup & run guide — Windows 11 + RTX 5070 Ti + Minikube

Everything below was written for this exact machine: Windows 11, NVIDIA GeForce
RTX 5070 Ti Laptop GPU (Blackwell, compute capability **sm_120**), Docker Desktop
with the WSL2 backend, and Minikube on the Docker driver.

---

## 0. Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.11 (3.10+ works) | `python --version` |
| NVIDIA driver | **570 or newer** | `nvidia-smi` |
| Docker Desktop | latest, WSL2 backend | `docker version` |
| kubectl | 1.29+ | `kubectl version --client` |
| Minikube | 1.33+ | `minikube version` |
| Git | 2.40+ | `git --version` |

```powershell
winget install Python.Python.3.11
winget install Docker.DockerDesktop
winget install Kubernetes.kubectl
winget install Kubernetes.minikube
winget install Git.Git
```

`nvidia-smi` reports the *driver's* CUDA version (e.g. 12.9). You do **not** need
a separate CUDA Toolkit install — the PyTorch wheel ships its own CUDA runtime.

---

## 1. Python environment and the sm_120 problem

The RTX 50-series is Blackwell, compute capability sm_120. The default PyPI
`torch` wheel is built with kernels up to sm_90, so it installs fine and then
fails at the first CUDA op:

```
NVIDIA GeForce RTX 5070 Ti Laptop GPU with CUDA capability sm_120 is not
compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 ... sm_90.
```

The fix is the **cu128** build, which is what `requirements/train-gpu.txt` pins.

```powershell
cd mlops-pytorch-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# GPU (training on this laptop)
pip install -r requirements/train-gpu.txt

# Verify before doing anything else
python scripts/verify_gpu.py
```

Expected:

```
torch          : 2.9.1+cu128
built for CUDA : 12.8
cuda available : True
arch list      : ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
device         : NVIDIA GeForce RTX 5070 Ti Laptop GPU
capability     : sm_120
GPU is ready.
```

If `sm_120` is missing from the arch list, you are on a non-cu128 wheel:

```powershell
pip uninstall -y torch torchvision
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

If that exact version has been superseded, drop the pin and take whatever the
cu128 channel currently offers:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

---

## 2. Train on the GPU

```powershell
python src/train.py --config configs/training_config.yaml
```

Output is one JSON object per line:

```json
{"event":"run_start","device":"cuda","gpu":"NVIDIA GeForce RTX 5070 Ti Laptop GPU","torch_version":"2.9.1+cu128","mixed_precision":true,"epochs":20,"batch_size":128,"ts":1754038800.12}
{"epoch":1,"train_loss":1.4021,"train_accuracy":0.4915,"val_loss":1.1183,"val_accuracy":0.6042,"lr":0.000994,"epoch_seconds":27.4,"ts":1754038828.9}
{"event":"checkpoint_saved","path":"checkpoints\\classifier_v1.pt","val_accuracy":0.6042,"ts":1754038829.1}
```

Pipe it through `jq` if you want a table: `python src/train.py | jq -c 'select(.epoch)'`.

Tuning notes for this GPU:

- `batch_size: 128` with `mixed_precision: true` is a comfortable fit; 256 also
  fits if you raise `learning_rate` to ~0.002.
- If DataLoader workers hang on Windows, set `num_workers: 0` in the config.
- Watch utilisation in a second terminal: `nvidia-smi dmon -s um`.

Then export the sample images the serving UI picks from:

```powershell
python scripts/export_test_images.py --count 200
```

---

## 3. Serve locally

```powershell
pip install -r requirements/serve.txt
$env:MODEL_PATH = "./checkpoints/classifier_v1.pt"
$env:SAMPLE_DIR = "./data/test_images"
python src/serve.py
```

- UI: <http://localhost:8080>
- Health: `curl.exe http://localhost:8080/health`
- Predict: `curl.exe -X POST http://localhost:8080/predict -F "image=@test_image.png"`

The curl examples all reference `test_image.png`. Create one from the exported
samples (any of them works, and the filename tells you the true label):

```powershell
Copy-Item (Get-ChildItem data/test_images/*.png | Get-Random) test_image.png
```

---

## 4. Docker

```powershell
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker images | Select-String mlops
```

Run training with mounted volumes (PowerShell uses `${PWD}`, not `$(pwd)`):

```powershell
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-train:v1
```

Run serving:

```powershell
docker run --rm -p 8080:8080 `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/data:/app/data `
  mlops-serve:v1

docker ps            # STATUS column shows (healthy) once HEALTHCHECK passes
```

### GPU inside Docker (optional)

Docker Desktop's WSL2 backend exposes the GPU with `--gpus all`. Build a GPU
training image first:

```powershell
docker build -f docker/Dockerfile.train -t mlops-train-gpu:v1 `
  --build-arg REQS=requirements/train-gpu.txt .
docker run --rm --gpus all -v ${PWD}/data:/app/data -v ${PWD}/checkpoints:/app/checkpoints mlops-train-gpu:v1
```

(Or simply swap the `COPY requirements/train.txt` line to `train-gpu.txt`.)

---

## 5. Minikube

```powershell
minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=40g
minikube addons enable metrics-server        # needed by the HPA
minikube status
```

Minikube runs its own container runtime, so images built on the host must be
side-loaded:

```powershell
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
minikube image ls | Select-String mlops
```

That is why every manifest sets `imagePullPolicy: IfNotPresent` — with the
default `Always`, the kubelet would try Docker Hub and fail with `ErrImagePull`.

Apply in order:

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml

kubectl get jobs -n ml-training -w
kubectl logs -f job/pytorch-training -n ml-training

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment/model-serving -n ml-training
```

Test:

```powershell
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# separate terminal
curl.exe http://localhost:8080/health
curl.exe -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

`scripts/deploy_minikube.ps1` does all of the above in one shot.

### About GPU on Minikube

Minikube's Docker driver on Windows cannot pass the laptop GPU through to the
node — GPU support requires a Linux host (`minikube start --gpus all`) or a
cloud node pool with the NVIDIA device plugin. The in-cluster Job therefore runs
on CPU with 3 epochs, while the real GPU training happens on the host and inside
Docker with `--gpus all`. `k8s/training-job-gpu.yaml` is the manifest for a
GPU-capable cluster; validate it locally without applying:

```powershell
kubectl apply -f k8s/training-job-gpu.yaml --dry-run=server
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `sm_120 is not compatible` | non-cu128 wheel | `pip install -r requirements/train-gpu.txt --force-reinstall` |
| `ErrImagePull` / `ImagePullBackOff` | image not in the Minikube runtime | `minikube image load mlops-*:v1` |
| Job pod `Pending`, events show insufficient cpu | node smaller than the 2-CPU request | `minikube start --cpus=4 --memory=8192` |
| `PermissionError: /app/checkpoints` | PVC owned by root | keep `fsGroup` in the pod securityContext |
| Serving pods stuck `0/2 Ready` | no checkpoint on the PVC yet | let the Job finish; `/health` returns 503 until then |
| `no metrics available` on HPA | metrics-server off | `minikube addons enable metrics-server`, wait ~60 s |
| DataLoader workers hang on Windows | multiprocessing spawn | set `num_workers: 0` |
| `docker run` volume path errors | `$(pwd)` is bash syntax | use `${PWD}` in PowerShell |
