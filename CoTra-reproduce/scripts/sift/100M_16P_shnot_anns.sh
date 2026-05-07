# ../scripts/sift/100M_16P_shnot_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/bigann
million=100
base_file=${dataset_path}/base.${million}M.u8bin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/bigann-${million}M
query_path=/data/share/users/xyzhi/data/bigann/query.public.10K.u8bin
num_threads=1
index_save_path=/data/share/users/xyzhi/data/bigann

# remember to change machine num
index_save_dir=${index_save_path}/share_nothing/test_${million}M_16P
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

10      0.90105 0 us    46407.4 /s
20      0.96123 0 us    40866.2 /s
30      0.98057 0 us    33074.9 /s
50      0.99282 0 us    23506.9 /s
100     0.9986  0 us    13790.8 /s
200     0.99969 0 us    7870.4 /s
300     0.99986 0 us    5547.69 /s
400     0.99987 0 us    4284.34 /s
500     0.99988 0 us    3511.11 /s
600     0.99988 0 us    2904.53 /s
700     0.99988 0 us    2464.86 /s
