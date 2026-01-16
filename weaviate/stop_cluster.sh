#!/bin/bash

set -e

echo "开始停止 Weaviate 集群..."

# 节点配置
NODES=(
    "192.168.200.204:weaviate_node0"
    "192.168.200.205:weaviate_node1"
    "192.168.200.206:weaviate_node2"
    "192.168.200.207:weaviate_node3"
    "192.168.200.208:weaviate_node4"
    "192.168.200.209:weaviate_node5"
)

user="CONNECT\\xchen740"
weaviate_dir="/home/CONNECT/xchen740/workspace/vector-retrieve/weaviate"

for node in "${NODES[@]}"; do
    IP=${node%:*}
    NODE_NAME=${node#*:}
    
    echo "正在停止节点 ${NODE_NAME} (${IP})..."
    
    # 远程执行停止和删除容器命令
    ssh "${user}@${IP}" \
    "cd '${weaviate_dir}' && \
    podman compose -f docker-compose.yml stop '${NODE_NAME}' && \
    podman compose -f docker-compose.yml rm -f '${NODE_NAME}'"
    
    echo "节点 ${NODE_NAME} 已停止"
done

echo "集群已全部停止"