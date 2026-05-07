# ../scripts/deep/1000M_16P_shnot_index.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=1000
base_file=${dataset_path}/base.${million}M.fbin
num_threads=32
index_save_path=/data/share/users/xyzhi/data/deep

# remember to change machine num
index_save_dir=${index_save_path}/share_nothing/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_index \
    --config_file ${CONF_FILE} \
    --graph_type shared_nothing \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 --scalagraph_v2 -s ${million} -t ${num_threads}
done