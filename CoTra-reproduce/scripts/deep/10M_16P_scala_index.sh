# ../scripts/deep/10M_16P_scala_index.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=10
base_file=${dataset_path}/base.${million}M.fbin
num_threads=32
index_save_path=/data/share/users/xyzhi/data/deep

R=48 # degree limit
L=500 # build queue size, equivalent to efConstruction
# index_save_dir=${index_save_path}/vamana_test/merged_index_deep_100M_R24_L500_A1.2_replica_2
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_index \
    --config_file ${CONF_FILE} \
    --graph_type scalagraph_v3 \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L ${L} -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
    # 2>&1 | tee ./build_logs/deep/merged_index_deep_10M_R48_L500_A1.2_B_10_M_${m}_T32_kbase_4.log
done