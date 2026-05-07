# numactl --interleave=all ../scripts/deep/100M_16P_single_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w1043.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/deep
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/deep-${million}M
query_path=/data/share/users/xyzhi/data/deep/query.public.10K.fbin
num_threads=1
index_save_path=/data/share/users/xyzhi/data/deep

# this scripts is curently used for 
R=96 # degree limit
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P_${R}D
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type single \
    --graph_type vamana \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done

# 8t
# 10      0.6272  45.6711 us      21895.7 /s
# 20      0.75434 65.3355 us      15305.6 /s
# 30      0.81852 87.7651 us      11394.1 /s
# 50      0.88879 132.395 us      7553.16 /s
# 100     0.9495  236.623 us      4226.13 /s
# 200     0.97935 431.452 us      2317.75 /s
# 300     0.98712 629.502 us      1588.56 /s
# 400     0.99034 818.442 us      1221.83 /s
# 500     0.99174 1000.94 us      999.059 /s
# 600     0.99375 1177.7 us       849.116 /s
# 700     0.995   1357.08 us      736.874 /s

# 64t
# 10      0.6272  68.0605 us      14692.8 /s
# 20      0.75434 28.1075 us      35577.7 /s
# 30      0.81852 26.3681 us      37924.6 /s
# 50      0.88879 38.1467 us      26214.6 /s
# 100     0.9495  66.5545 us      15025.3 /s
# 200     0.97935 121.555 us      8226.7 /s
# 300     0.98712 175.458 us      5699.35 /s
# 400     0.99034 184.552 us      5418.52 /s
# 500     0.99174 221.039 us      4524.1 /s
# 600     0.99375 269.547 us      3709.92 /s
# 700     0.995   317.094 us      3153.64 /s

# 128t
# 10      0.6272  15.0377 us      66499.5 /s
# 20      0.75434 16.2953 us      61367.4 /s
# 30      0.81852 21.7071 us      46067.9 /s
# 50      0.88879 33.9146 us      29485.8 /s
# 100     0.9495  64.6338 us      15471.8 /s
# 200     0.97935 124.746 us      8016.27 /s
# 300     0.98712 190.995 us      5235.75 /s
# 400     0.99034 269.081 us      3716.35 /s