# ../scripts/sift/100M_8P_shnot_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-40.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/bigann
million=100
base_file=${dataset_path}/base.${million}M.u8bin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/bigann-${million}M
query_path=/data/share/users/xyzhi/data/bigann/query.public.10K.u8bin
num_threads=1
index_save_path=/data/share/users/xyzhi/data/bigann

# remember to change machine num
index_save_dir=${index_save_path}/share_nothing/test_${million}M_8P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type b2 \
    --graph_type shared_nothing \
    --data_type uint8 --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done


# 10      0.86167 0 us    64796.6 /s
# 20      0.93933 0 us    47670.6 /s
# 30      0.9662  0 us    37138.3 /s
# 50      0.98582 0 us    25705 /s
# 100     0.99645 0 us    15565.8 /s
# 200     0.99928 0 us    8594.92 /s
# 300     0.99967 0 us    6052.05 /s
# 400     0.99981 0 us    4647.03 /s
# 500     0.99986 0 us    3727.9 /s
# 600     0.99988 0 us    2991.33 /s
# 700     0.99988 0 us    2559.9 /s
