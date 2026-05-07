# ../scripts/deep/100M_16P_Kshnot_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/deep-${million}M
query_path=/data/share/users/xyzhi/data/deep/query.public.10K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/deep

R=48
index_save_dir=${index_save_path}/K_share_nothing/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type b2kmeansbatch \
    --graph_type Kshared_nothing \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 --scalagraph_v2 -s ${million} -t ${num_threads}
done

# select
# 10      0.68171 0 us    69570.1 /s
# 15      0.76573 0 us    53213.3 /s
# 20      0.7966  0 us    51544.5 /s
# 30      0.86161 0 us    38690.3 /s
# 40      0.88778 0 us    32497.7 /s
# 50      0.90481 0 us    27858.4 /s
# 70      0.93023 0 us    21594.6 /s
# 100     0.94667 0 us    16228.3 /s
# 150     0.96046 0 us    11522.9 /s
# 200     0.96668 0 us    8981.61 /s
# 250     0.96971 0 us    7385.67 /s
# 300     0.97091 0 us    6316.24 /s
# 400     0.97207 0 us    4888.59 /s
# 500     0.97338 0 us    3990.13 /s
# 600     0.97433 0 us    3387.4 /s
# 700     0.97462 0 us    2925.38 /s

# all
# 10      0.72292 0 us    26957.4 /s
# 15      0.79495 0 us    22347   /s
# 20      0.83936 0 us    19057.5 /s
# 30      0.89112 0 us    15263.9 /s
# 40      0.92022 0 us    12767.9 /s
# 50      0.93918 0 us    11118.2 /s
# 70      0.96131 0 us    8843.15 /s
# 100     0.97678 0 us    6938.13 /s
# 150     0.98829 0 us    5174.98 /s
# 200     0.99296 0 us    4117.21 /s
# 250     0.9952  0 us    3477.43 /s
# 300     0.99628 0 us    2997.24 /s
# 400     0.99769 0 us    2362.27 /s
# 500     0.99827 0 us    1962.66 /s
# 600     0.99852 0 us    1677.62 /s
# 700     0.99874 0 us    1457.93 /s

