
# CoTra: Towards Efficient and Scalable Distributed Vector Search with RDMA

This project is an implementation of a high-performance distributed similarity vector retrieval system using **Remote Direct Memory Access (RDMA)**. The system is designed to achieve arge-scale vector searches by leveraging RDMA's high performance communication capabilities and efficient memory management.

---

## Key Features

- **High Throughput & Scalable Architecture**: RDMA-based direct memory access minimizes CPU overhead and network round trips, supports distributed deployment across multiple nodes for large-scale data.  
- **Efficient Memory Utilization**: Implements a custom memory management system to handle vector storage and retrieval efficiently.  
- **C++ Implementation**: The system is built in C++ for maximum performance and fine-grained control over RDMA operations.

---

## Requirements

- **Operating System**: Linux (with RDMA-capable hardware support, Connect-X3 or above).  
- **Compiler**: GCC 11+
- **RDMA Libraries**: libibverbs, Infiniband

---

## Installation

1. **Install Dependencies**:
We use **memcached** to manage meta data of servers.
We use **[DiskANN](https://github.com/microsoft/DiskANN)** as default graph index.
```bash
sudo apt update
sudo apt install libibverbs-dev librdmacm-dev
sudo apt install memcached libmemcached-tools
```
Please install dependencies of DiskANN: intel-oneapi

2. **Build the Project**:
```bash
mkdir build && cd build
cmake ..
make -j
```
## Download Datasets
SIFT, DEEP, and Text2Image dataset (From Neurips21 BIGANN Contest): 
https://big-ann-benchmarks.com/neurips21.html

LAION dataset:
https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/

All dataset files (raw vector, ground truth, and query file follows the same format in Neurips21 BIGANN Contest).

# Usage

## Manage Server Connection
Edit <your_config>.conf file in ./scripts

The first and second lines are the leader machine's ip and port, please select one machine of cluster as the leader.
We use **machine ID** to identify each machine, which should be user defined, and the leader machine's ID must be 0.
Then write the IPs and IDs of every machine in cluster on the following lines.
e.g.,
```bash
<leadr_machine_ip>
<leadr_machine_port>
<machine0_IP>=0
<machine1_IP>=1
<machine2_IP>=2
...
```


User can edit the scripts in ./scripts/dataset/ to adjust system configuration.
We provide different implementations discribed in paper, including **single_machine**, **global_index**, 
**shard_index(Random)**, **shard_index(Kmeans)**, and **CoTra**.

## Build Index Scripts & Options
An index script format example:
```bash
# This for restart memcache server
source ../scripts/restart_memcache.sh 
# Your server metadata file.
CONF_FILE="../scripts/<your_config>.conf"
# Clear memcache based on your config file.
clear_memcache "$CONF_FILE"

dataset_path=<Path_to_your_data_file_dir>
million=10 # Size of dataset in millions.
base_file=${dataset_path}/<Path_to_base_file_dir> # Path to the base vector data.
num_threads=<Thread_num>
index_save_path=<Path_to_index_save_path>

R=48 # Define the max degree of graph index, default is 48.
L=500 # build queue size, equivalent to efConstruction
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P

# graph_type [index option]: vamana, shared_nothing, Kshared_nothing, scalagraph_v3
# data_type [dataset type option]: uint8, int8, float
# dist_fn [distance function option]: l2, ip
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_index \
    --config_file ${CONF_FILE} \
    --graph_type shared_nothing \
    --data_type float \
    --dist_fn l2 \
    --data_path ${base_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L ${L} -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done
```


## Vector Search Scripts & Options
An anns scripts format example:
```bash
# This for restart memcache server
source ../scripts/restart_memcache.sh 
# Your server metadata file.
CONF_FILE="../scripts/<your_config>.conf"
# Clear memcache based on your config file.
clear_memcache "$CONF_FILE"

dataset_path=<Path_to_your_data_file_dir>
million=10 # Size of dataset in millions.
base_file=${dataset_path}/<Path_to_base_file_dir> # Path to the base vector data.
gt_file=${dataset_path}/<Path_to_gt_file_dir>
query_path=${dataset_path}/<Path_to_query_file_dir>
num_threads=<Thread_num>
index_save_path=<Path_to_index_save_path>

R=48 # degree limit.
index_save_dir=${index_save_path}/<Path_to_index_save_file_dir>

# Note: app_type option should be aligned with graph_type respectively.
# app_type [implementation option]: single, b2, scala_v3 
# graph_type [index option]: vamana, shared_nothing, Kshared_nothing, scalagraph_v3
# data_type [dataset type option]: uint8, int8, float
# dist_fn [distance function option]: l2, ip
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type b2 \
    --graph_type shared_nothing \
    --data_type float \
    --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 --scalagraph_v2 -s ${million} -t ${num_threads}
done

```

## Build Index & Run ANNS
```bash
# All scripts can execute simultaneously, so we suggest to use terminal tools like tmux.
# For each server, run: 
./scripts/<your_scripts>.sh
```

Enable hugepage for better performance.
```bash
grep -i huge /proc/meminfo
sudo su
xyzhi
echo 8000 > /proc/sys/vm/nr_hugepages
su xyzhi
```

# License
This project is licensed under the MIT License. See the LICENSE file for more details.