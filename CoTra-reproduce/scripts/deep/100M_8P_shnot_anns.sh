# ../scripts/deep/100M_8P_shnot_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w41-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/deep-${million}M
query_path=/data/share/users/xyzhi/data/deep/query.public.10K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/deep

# remember to change machine num
index_save_dir=${index_save_path}/share_nothing/test_${million}M_8P
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
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 --scalagraph_v2 -s ${million} -t ${num_threads}
done

# 10      0.84609 0 us    38282.9 /s
# 20      0.92624 0 us    27085.4 /s
# 30      0.95612 0 us    21104.8 /s
# 50      0.97944 0 us    14917.8 /s
# 100     0.99312 0 us    8899.87 /s
# 200     0.99759 0 us    5166.2 /s
# 300     0.99856 0 us    3702.84 /s
# 400     0.999   0 us    2877.77 /s
# 500     0.9991  0 us    2356.75 /s
# 600     0.99917 0 us    1951.74 /s
# 700     0.99919 0 us    1678.15 /s
