# Experiments

## Running batch experiments on lightning

### 1. Clone the repo to a fresh lightning studio

```bash
git clone https://github.com/MedARC-AI/fmri-fm.git
cd fmri-fm
uv sync
```

### 2. Configure your environment

Create a `~/fmri-fm/.env` file with the following settings

```bash
# your user name on lightning teamspace shared storage
# overriding default 'volunteer' user name
SHARE_USER=yourname

# Git settings for pushing commits
GITHUB_TOKEN="github_pat_XXXX"
GIT_COMMITTER_EMAIL="your.name@email.com"
GIT_AUTHOR_EMAIL="your.name@email.com"

# R2 data access key
AWS_ACCESS_KEY_ID="XXXX"
AWS_SECRET_ACCESS_KEY="XXXX"
AWS_ENDPOINT_URL_S3="https://XXXX.r2.cloudflarestorage.com"

# Dataset cache directory on local machine storage
DATA_CACHE_DIR="/tmp/datasets"

# Prevent hf from bloating the studio directory with cache
HF_HOME="/tmp/huggingface"
```

### 3. Set up your experiment

Configs and launch scripts. See [`pretrain_mae`](pretrain_mae) for example.

### 4. Run interactively

You can provision a GPU instance and run locally

```bash
bash pretrain_mae/launch_pretrain.sh
```

### 5. Run with a batch job

You can also submit batch jobs with the [lightning SDK](https://lightning.ai/docs/overview/sdk/batch).

```python
from lightning_sdk import Studio, Machine, Job

studio = Studio(name="user-pretrain", teamspace="medarc", org="medarc")

job = Job.run(
    command=f"bash fmri-fm/experiments/pretrain_mae/launch_pretrain.sh",
    name="pretrain_mae",
    machine=Machine.H100,
    studio=studio,
    interruptible=True,
)
```

You can submit jobs from a jupyter notebook for easier interaction with running jobs. See for example [`pretrain_mae/submit.ipynb`](pretrain_mae/submit.ipynb).
