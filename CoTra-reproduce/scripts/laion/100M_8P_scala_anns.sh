# ../scripts/laion/100M_16P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/laion
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/laion/gt.10k.bin
query_path=/data/share/users/xyzhi/data/laion/query.10k.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/laion

R=48 # degree limit
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type scala_v3 \
    --graph_type scalagraph_v3 \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done



# 10       0.82112         0.00    43940.207
# 20       0.87954         0.00    49100.719
# 30       0.90343         0.00    32968.590
# 50       0.92867         0.00    28029.105
# 100      0.94974         0.00    17897.861
# 200      0.96066         0.00    11151.679
# 300      0.96515         0.00    7878.693
# 400      0.96768         0.00    6219.080
# 500      0.96934         0.00    5147.939
# 600      0.97034         0.00    3864.687



