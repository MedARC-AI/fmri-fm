# Experiments

## Experiment contributing guide

Contributed experiments are the "atoms" of a paper. They should include:

- all scripts and configs needed to reproduce results
- pruned output runs excluding checkpoints, logged images, and other large files
- notebooks or scripts needed to make publication ready figures or tables
- the generated publication ready figures/tables themselves
- a `README.md` with a rough draft writeup of the setup and results (with embedded figures/tables), plus any other documentation.

In addition, full experiment outputs should be backed up to `s3://medarc/fmri-fm/experiments`.

### Experiment contributing workflow

1. Fork the repo and create your branch from `main`.
2. [Install the project](README.md#installation), including [pre-commit hooks](https://pre-commit.com/#3-install-the-git-hook-scripts).
3. Copy a recent previous experiment to get the correct structure.
4. Edit configs and run scripts.
5. Open a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) to get feedback on the experiment design.
6. Run the experiment. Watch the logs/wandb.
7. Analyze results, make plots/tables.
8. Write up setup and results in a `README.md`.
9. Back up results to `s3://medarc/fmri-fm/experiments`.
10. Update PR and request review.

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
GIT_AUTHOR_NAME="Your Name"
GIT_COMMITTER_NAME="Your Name"
GIT_AUTHOR_EMAIL="your.name@email.com"
GIT_COMMITTER_EMAIL="your.name@email.com"

# R2 data access key
AWS_ACCESS_KEY_ID="XXXX"
AWS_SECRET_ACCESS_KEY="XXXX"
AWS_ENDPOINT_URL_S3="https://XXXX.r2.cloudflarestorage.com"

# Wandb
WANDB_API_KEY="XXXX"

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

studio = Studio(name="fmri-fm-pretrain", teamspace="medarc", org="medarc")

job = Job.run(
    command=f"bash fmri-fm/experiments/pretrain_mae/launch_pretrain.sh",
    name="pretrain_mae",
    machine=Machine.H100,
    studio=studio,
    interruptible=True,
)
```

You can submit jobs from a jupyter notebook for easier interaction with running jobs. See for example [`pretrain_mae/submit.ipynb`](pretrain_mae/submit.ipynb).
