# FlashTensor AE

[![DOI](https://zenodo.org/badge/893756798.svg)](https://zenodo.org/badge/latestdoi/893756798)

## Abstract

This artifact supports the reproduction of results presented in paper #80 at PPoPP'25, titled FlashTensor: Optimizing Tensor Programs by Leveraging Fine-grained Tensor Property, including Figure 12, Figure 13 and Figure 14.

The repository includes:
* Backup logs in `fig12/backup_logs`, `fig13/backup_logs` and `fig14/backup_logs`. These files were used to generate Figure 12, Figure 13 and Figure 14 in the paper.
* Reproduction Scripts in `fig12/`, `fig13/` and `fig14/`. These scripts were used for a in-depth reproduction by executing baselines and our work.
* Source code in `3rd/` and `deepgengraph_exp`. `3rd/deepgengraph` is our work named `FlashTensor` in paper. `3rd/tvm` includes the [TVM](https://github.com/apache/tvm) baseline (commit `64969035fd4f3c1ddcc23caa84567bf90e33889c`). `deepgengraph_exp` provides additional baselines, including PyTorch, Inductor and TensorRT.

### About Missing Baseline in the Artifact

This artifact does not include reproductions for two baseliens, `Korch` and `EinNet`, shown in Figure 12 (end-to-end and core module performance). These baselines were conducted by their respective authors with private code modifications, which are unavailable for inclusion. However, as these baselines are not the top-performing methods in our experiments, their absence does not impact the validity of our experimental results.

## Getting Started

### Log into the provided cluster

We have provided a ssh private key for AE reviewers, named `id_rsa_ppopp25_ae`, to access the provided cluster.

```bash
ssh -p 40422 -i ./id_rsa_ppopp25_ae ppopp25_ae@166.111.68.163
```

The logged cluster is named `yes`, where there are 8xA100 scheduled by slurm. By `ssh fuse0`, we can log into another cluster named `fuse0`, where there are 3xH100 also scheduled by slurm.

### Use tmux for Long-Running Scripts

We strongly recommend running all scripts inside [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) to prevent interruptions.

Basic usages of tmux are as follows:
```bash
# Create a session named 'ae'
tmux new -s ae
# Now we are in the sesion
RUN/SOME/SCRIPTS 


# Detach session
# Type Ctrl + b, then D 
Ctrl-b D

# List all sessions
tmux ls
# Reattach to a session named 'ae'
tmux a -t ae
```

### Quick Reproduction: Plot from Backup Logs (~2 minutes)

Reproduce figures using pre-generated logs:

```bash
cd ~/ppopp25_ae

# for Figure 12
cd ./fig12
# plot Figure 12 (generate fig12_e2e.pdf and fig12_kernel.pdf in the current directory)
./plot.sh ./backup_logs
cd ..

# for Figure 13
cd ./fig13
# plot Figure 13 (generate fig13.pdf in the current directory)
./plot.sh ./backup_logs
cd ..

# for Figure 14
cd ./fig14
# plot Figure 14 (generate fig14.pdf in the current directory)
./plot.sh ./backup_logs
cd ..
```

Generated PDFs (`fig12_e2e.pdf`, `fig12_kernel.pdf`, `fig13.pdf` and `fig14.pdf`) will be saved in their respective directories. Use `scp` to download them locally.

### In-depth Reproduction: Plot from Real Run (~23 hours)

```bash
cd ~/ppopp25_ae

# for Figure 12
cd ./fig12
# run on A100 machine (~6.5 hours, logs will be saved in ./ae_logs)
./run.sh
# log into H100 machine
ssh fuse0
cd ~/ppopp25_ae/fig12
# run on H100 machine (~16 hours, logs will be saved in ./ae_logs)
./run.sh
# exit from H100 machine
exit
# copy logs from H100 machine
rsync -av --progress fuse0:/home/ppopp25_ae/ppopp25_ae/fig12/ae_logs/ ./ae_logs
# plot Figure 12 (generate fig12_e2e.pdf and fig12_kernel.pdf in the current directory)
./plot.sh ./ae_logs
cd ..

# for Figure 13
cd ./fig13
# run on H100 machine (~10 minutes, logs will be saved in ./ae_logs)
ssh fuse0
cd ~/ppopp25_ae/fig13
./run.sh
# exit from H100 machine
exit
# copy logs from H100 machine
rsync -av --progress fuse0:/home/ppopp25_ae/ppopp25_ae/fig13/ae_logs/ ./ae_logs
# plot Figure 13 (generate fig13.pdf in the current directory)
./plot.sh ./ae_logs
cd ..


# for Figure 14
cd ./fig14
# run on A100 machine (~10 minutes, logs will be saved in ./ae_logs)
./run.sh
# run on H100 machine (~10 minutes, logs will be saved in ./ae_logs)
ssh fuse0
cd ~/ppopp25_ae/fig14
./run.sh
# exit from H100 machine
exit
# copy logs from H100 machine
rsync -av --progress fuse0:/home/ppopp25_ae/ppopp25_ae/fig14/ae_logs/ ./ae_logs
# plot Figure 14 (generate fig14.pdf in the current directory)
./plot.sh ./ae_logs
cd ..
```

Generated PDFs (`fig12_e2e.pdf`, `fig12_kernel.pdf`, `fig13.pdf` and `fig14.pdf`) will be saved in their respective directories. Use `scp` to download them locally.

#### Potential Issues and Solutions

1. Long-running slurm tasks for TVM (Figure 12 on A100 Cluster).

  Due to internal problems, some TVM autotuner tasks may exceed 3.5 hours. To solve the problem, we should first use `scancel` to scancel the slurm task, and then use `run_single.sh` to rerun TVM case:
  ```bash
  # find which case we should rerun
  ps -aux | grep srun | grep ppopp25_ae
  # see the current task id
  squeue
  # cancel the task to make the `run.sh` continue
  scancel <task_id>
  # rerun the specific TVM case using `run_single.sh`
  ./run_single.sh kernel roco tvm
  ```

2. Warning occurs in `./plot.sh`.

  Warnings like `Warning: ./ae_logs/kernel.100.roco.tvm.log does not have avg time` indicates missing logs. To solve the problem, we should also use `run_single.sh` to run specific TVM case. For example:
  ```bash
  ./run_single.sh kernel roco tvm
  ```

For AE Reviewers: If you encounter any issues, feel free to contact use directly through HotCRP comments.

## Installation

For AE Reviewers: We strongly recommend using the provided environment because some network issues may occur in the provided cluster (like connecting to Github and installing pip packages).

```bash
# After log into the provided cluster

# backup existing environement (Note that Python Venv contains absolute paths, making backups non-functional directly after moving)
mv ~/ppopp25_ae ~/ppopp25_ae_backup
git clone https://github.com/monellz/FlashTensor-AE.git ppopp25_ae

# install on A100 machine
cd ~/ppopp25_ae
./script/install.sh

# install on H100 machine
ssh fuse0
mv ~/ppopp25_ae ~/ppopp25_ae_backup
git clone https://github.com/monellz/FlashTensor-AE.git ppopp25_ae

cd ~/ppopp25_ae
./script/install.sh
```
