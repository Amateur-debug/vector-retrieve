# ../scripts/deep/vamana_index_10M_4P.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-40.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=10
base_file=${dataset_path}/base.${million}M.fbin
num_threads=1
index_save_path=/data/share/users/xyzhi/data/deep

# this scripts is curently used for 
index_save_dir=${index_save_path}/vamana_test/merged_index_deep_100M_R24_L500_A1.2_replica_2
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/index/loader \
    --config_file ${CONF_FILE} \
    --app_type single \
    --graph_type vamana \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 24 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done