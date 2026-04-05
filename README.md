<p align="center">
  <a href="http://appfl.rtfd.io"><img src="https://github.com/APPFL/APPFL/blob/main/docs/_static/logo/logo_small.png?raw=true" alt="APPFL logo" style="width: 40%; height: auto;"></a>
</p>

<p align="center" style="font-size: 18px;">
    <b>Federated Learning across HPC Facilities with APPFL and Globus</b>
</p>

This repository contains the code for deploying federated learning (FL) workloads across geographically distributed High Performance Computing (HPC) facilities using the [APPFL](https://github.com/APPFL/APPFL) framework with [Globus Compute] and [Globus Transfer] for orchestration.

---

## Table of Contents

- [Installation](#installation)
- [Running the Code](#running-the-code)
  - [Step 1: Set Up Globus Compute Endpoints](#step-1-set-up-globus-compute-endpoints)
  - [Step 2: Configure the YAML Files](#step-2-configure-the-yaml-files)
  - [Step 3: Configure Globus Transfer](#step-3-configure-globus-transfer)
  - [Step 4: Run the Experiment](#step-4-run-the-experiment)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/APPFL/APPFL_HPC_public.git
cd APPFL_HPC_public
pip install -e .
```

This will install all required dependencies automatically.

---

## Running the Code

We walk through the steps to run federated fine-tuning of Llama-2-7B on the [SMolInstruct](https://huggingface.co/datasets/osunlp/SMolInstruct) chemistry dataset using FedAvg as an example. The relevant configuration files are located in:

```
examples/resources/config_gc/llm_chem/
```

### Step 1: Set Up Globus Compute Endpoints

Each participating HPC facility requires two Globus Compute endpoints, and the server requires one. Please refer to the [Globus Compute Quickstart Guide](https://globus-compute.readthedocs.io/en/stable/quickstart.html) for installation and configuration instructions.

**On each client facility (repeat for each HPC system):**

1. **MPI Engine** — used to launch distributed training jobs via MPI. Note its `endpoint_id` for the client YAML configuration in the next step.
2. **Regular Engine** — used to stage model parameters before and after training. Note its `endpoint_id` as the `transfer_endpoint_id` in the client YAML configuration.

**On the server:**

1. **Regular Engine** — used to coordinate aggregation. Note its `endpoint_id` as the `server_endpoint_id` in the server YAML configuration.

---

### Step 2: Configure the YAML Files

Navigate to the example configuration directory:

```
examples/resources/config_gc/llm_chem/
```

There are two YAML files to configure for the FedAvg experiment:

#### Client Configuration: `clients_fedavg.yaml`

Edit this file to fill in the Globus Compute endpoint IDs for each client facility. Also configure training parameters as needed.

#### Server Configuration: `server_fedavg.yaml`

Edit this file to fill in the server Globus Compute endpoint ID and aggregation settings. Adjust other settings as needed.

---

### Step 3: Configure Globus Transfer

Globus Transfer is used to move model parameters between the server and clients. In both `clients_fedavg.yaml` and `server_fedavg.yaml`, configure the Globus Transfer endpoint IDs and staging paths. For Proxystore configuration, please refer to the [Proxystore documtation](https://github.com/proxystore/proxystore). 
---

### Step 4: Run the Experiment

Once all endpoints and YAML files are configured, launch the experiment from the server by running:

```bash
python examples/globus_compute/run_fedavg.py
```

This script reads the server and client configurations from the YAML files, distributes the initial global model to all clients via Globus Transfer, submits local training jobs to each facility's scheduler via Globus Compute, collects model updates, and performs global aggregation after each round. Training progress and global model checkpoints will be logged to the server.

To run with a different FL algorithm, one can create separate YAML configuration files and launch scripts. 

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{li2026scalablecrossfacilityfederatedlearning,
      title={Scalable Cross-Facility Federated Learning for Scientific Foundation Models on Multiple Supercomputers}, 
      author={Yijiang Li and Zilinghan Li and Kyle Chard and Ian Foster and Todd Munson and Ravi Madduri and Kibaek Kim},
      year={2026},
      eprint={2603.19544},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.19544}, 
}

@article{li2024advances,
  title   = {Advances in APPFL: A Comprehensive and Extensible Federated Learning Framework},
  author  = {Li, Zilinghan and He, Shilan and Yang, Ze and Ryu, Minseok and Kim, Kibaek and Madduri, Ravi},
  journal = {arXiv preprint arXiv:2409.11585},
  year    = {2024}
}

@inproceedings{ryu2022appfl,
  title        = {APPFL: open-source software framework for privacy-preserving federated learning},
  author       = {Ryu, Minseok and Kim, Youngdae and Kim, Kibaek and Madduri, Ravi K},
  booktitle    = {2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  pages        = {1074--1083},
  year         = {2022},
  organization = {IEEE}
}
```

---

## Acknowledgements

This work was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research, under Contract DE-AC02-06CH11357.

We gratefully acknowledge the computing resources provided on Improv, a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory. An award of computer time was provided by the ASCR Leadership Computing Challenge (ALCC) program. This research used resources of the Argonne Leadership Computing Facility, which is a U.S. Department of Energy Office of Science User Facility operated under contract DE-AC02-06CH11357. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy User Facility using NERSC award ALCC-ERCAP0038201.