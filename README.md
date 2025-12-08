# HeteroAccel: Automated Heterogeneous Accelerator Design for Multi-DNN Workloads

**HeteroAccel** is a framework for **automated design of heterogeneous ASIC accelerators** targeting **multi-DNN inference workloads**.  
It integrates **simulated annealing optimization** with **DNN scheduling** to explore heterogeneous accelerator architectures and achieve optimal energy-delay product (EDP).  
Built on the popular **Timeloop-Accelergy** framework (included as a submodule), HeteroAccel delivers accurate metrics for **delay, energy, and area** while automating scheduling and architecture co-optimization.

---

## 🔧 Key Features

- 🧊 **Simulated Annealing Optimization**  
  Explores heterogeneous accelerator architectures for multi-DNN workloads.

- 🏗️ **Timeloop-Accelergy Integration**  
  Uses Timeloop-Accelergy for accurate hardware modeling and energy, delay, and area estimation.

- 🔀 **DNN Scheduling-Aware Design**  
  Jointly optimizes accelerator architectures and schedules DNN workloads for maximum efficiency.

- 🎯 **Energy-Delay Product Optimization**  
  Targets EDP as the primary optimization metric for accelerator performance.

- 🧩 **Baselines and Comparisons**  
  Includes SOTA HDA-Q method and homogeneous multi-accelerator baseline for comparison.

- 🐳 **Dockerized Setup**  
  Fully reproducible Docker environment for building and running experiments.

---

## 📦 Technologies Used

- Python 3  
- Simulated Annealing (SA) optimization  
- Timeloop-Accelergy (submodule) for accelerator simulation  
- YAML-based workload and argument configuration  
- Docker for environment setup and reproducibility  

---

## ⚙️ Installation

Set up the Docker-based environment:

```bash
# Build the Docker image
./setup/build_docker.sh

# Launch the Docker container
./setup/run_docker.sh
```

## 🚀 Usage

Run simulated annealing optimization:

```bash
./run/main.sh
```
Run SOTA HDA-Q comparison:
```bash
./run/sota.sh
```
Evaluate homogeneous multi-accelerator baseline:
```bash
./run/baseline.sh
```
Configuration files:
- `main.py` – Main entry point for running optimization
- `run/args.yaml` – Defines all arguments and optimizer parameters
- `run/workloads.yaml` – Defines multi-DNN workloads

## 📚 Citation

If you use this repository, please cite:
```bibtex
@article{balaskas2024heterogeneous,
  title={Heterogeneous Accelerator Design for Multi-DNN Workloads via Heuristic Optimization},
  author={Balaskas, Konstantinos and Khdr, Heba and Sikal, Mohammed Bakr and Kre{\ss}, Fabian and Siozios, Kostas and Becker, J{\"u}rgen and Henkel, J{\"o}rg},
  journal={IEEE Embedded Systems Letters},
  volume={16},
  number={4},
  pages={317--320},
  year={2024},
  publisher={IEEE}
}
```
For related work (HDA-Q method):
```bibtex
@article{spantidi2022targeting,
  title={Targeting dnn inference via efficient utilization of heterogeneous precision dnn accelerators},
  author={Spantidi, Ourania and Zervakis, Georgios and Alsalamin, Sami and Roman-Ballesteros, Isai and Henkel, Joerg and Amrouch, Hussam and Anagnostopoulos, Iraklis},
  journal={IEEE Transactions on Emerging Topics in Computing},
  volume={11},
  number={1},
  pages={112--125},
  year={2022},
  publisher={IEEE}
}
```

## EAS Lab Installation

### Clone and Build
Git with SSH:
```
git clone git@github.com:crimsonmagick/hetero-accel.git \
    && cd hetero-accel/  \
    && . ./setup/install_eas_lab.sh
```

OR

Git with HTTPS:
```
git clone https://github.com/crimsonmagick/hetero-accel.git \
    && cd hetero-accel/  \
    && . ./setup/install_eas_lab.sh
```

### Test Your Installation

Test your Timeloop + Accelergy Installation:
```
PYTHONPATH=. python3 src/timeloop.py
```

Run the full annealing simulation:
```
python3 main.py --yaml-cfg-file run/args_cfg.yaml --workload-cfg-file run/workloads.yaml
```