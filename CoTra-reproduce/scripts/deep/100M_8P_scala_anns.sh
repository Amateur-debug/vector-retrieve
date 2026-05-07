# ../scripts/deep/100M_8P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-40.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/deep-${million}M
query_path=/data/share/users/xyzhi/data/deep/query.public.10K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/deep

# this scripts is curently used for 
index_save_dir=${index_save_path}/scalagraph/test_${million}M_8P
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
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done


# 10       0.49873         0.00    67643.438
# 20       0.65565         0.00    46455.879
# 30       0.73691         0.00    36294.875
# 50       0.77735         0.00    29160.045
# 100      0.86978         0.00    18930.180
# 200      0.92305         0.00    11565.427
# 300      0.95549         0.00    8370.085
# 400      0.96626         0.00    6672.886
# 500      0.97652         0.00    5511.203
# 600      0.98126         0.00    4579.954
# 700      0.98451         0.00    3999.758
