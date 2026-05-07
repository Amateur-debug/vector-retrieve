# ../scripts/text2image/100M_8P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-40.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/text2image
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/text2image-${million}M
query_path=/data/share/users/xyzhi/data/text2image/query.public.100K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/text2image

# this scripts is curently used for 
index_save_dir=${index_save_path}/scalagraph/test_${million}M_8P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type scala_v3 \
    --graph_type scalagraph_v3 \
    --data_type float --dist_fn mips \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done


# (0.52841, 58334.891),
# (0.64507, 50741.074),
# (0.70739, 36386.660),
# (0.77639, 27210.811),
# (0.85767, 18556.801),
# (0.91769, 10330.995),
# (0.94257, 7439.746 ),
# (0.95657, 5933.630 ),
# (0.96458, 4877.970 ),
# (0.97099, 4103.051 ),
# (0.97534, 3540.682 ),