# ../scripts/deep/10M_16P_shnot_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=10
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/deep-${million}M
query_path=/data/share/users/xyzhi/data/deep/query.public.10K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/deep

R=48
index_save_dir=${index_save_path}/share_nothing/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type b2 \
    --graph_type shared_nothing \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 --scalagraph_v2 -s ${million} -t ${num_threads}
done


10      0.88861 0 us    35327.3 /s
20      0.95166 0 us    26005.3 /s
30      0.97309 0 us    20590.3 /s
50      0.98799 0 us    14629.2 /s
100     0.99618 0 us    9189.33 /s
200     0.99858 0 us    5354.08 /s
300     0.99897 0 us    3825.26 /s
400     0.99911 0 us    3011.77 /s
500     0.9992  0 us    2480.37 /s
600     0.99921 0 us    2088.71 /s
700     0.99928 0 us    1795.39 /s