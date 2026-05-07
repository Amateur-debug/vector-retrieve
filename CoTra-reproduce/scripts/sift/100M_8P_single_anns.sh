# ../scripts/sift/100M_8P_single_anns.sh

source ../scripts/restart_memcache.sh
# CONF_FILE="../scripts/ip_w33-40.conf"
# for w38 test(8m)
# CONF_FILE="../scripts/ip_w38_8m.conf"
# for w32 test(8m)
CONF_FILE="../scripts/ip_w32_8m.conf"

clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/bigann
million=100
base_file=${dataset_path}/base.${million}M.u8bin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/bigann-${million}M
query_path=/data/share/users/xyzhi/data/bigann/query.public.10K.u8bin
num_threads=1
index_save_path=/data/share/users/xyzhi/data/bigann

R=48
index_save_dir=${index_save_path}/scalagraph/test_${million}M_8P


for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type single \
    --graph_type vamana \
    --data_type uint8 --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done


# profiling
# Computation Num Tocal : 8299602 Avg: 829
# 20      0.75753 166.934 us      5990.38 /s
# Computation Num Tocal : 11761265 Avg: 1176
# 30      0.82641 222.165 us      4501.15 /s
# Computation Num Tocal : 14966364 Avg: 1496
# 50      0.89616 323.687 us      3089.4 /s
# Computation Num Tocal : 21165372 Avg: 2116
# 100     0.9562  564.531 us      1771.38 /s
# Computation Num Tocal : 35938131 Avg: 3593
# 200     0.98376 1020.66 us      979.755 /s
# Computation Num Tocal : 63446835 Avg: 6344
# 300     0.99122 1474.56 us      678.169 /s
# Computation Num Tocal : 89209126 Avg: 8920
# 400     0.99449 1922.01 us      520.289 /s
# Computation Num Tocal : 113771249 Avg: 11377
# 500     0.99633 2367.18 us      422.444 /s
# Computation Num Tocal : 137460867 Avg: 13746
# 600     0.99713 2851.43 us      350.701 /s
# Computation Num Tocal : 160418600 Avg: 16041


# 8t

# 64t
# 10      0.77465 29.368 us       34050.7 /s
# 20      0.88411 18.4428 us      54221.7 /s
# 30      0.92878 21.2279 us      47107.8 /s
# 50      0.96637 31.7232 us      31522.7 /s
# 100     0.99074 53.9926 us      18521.1 /s
# 200     0.99803 98.68 us        10133.8 /s
# 300     0.99911 141.021 us      7091.12 /s
# 400     0.99948 182.748 us      5472.02 /s
# 500     0.99965 222.386 us      4496.68 /s
# 600     0.99971 264.314 us      3783.37 /s
# 700     0.99978 299.762 us      3335.98 /s


# 128t
# (0.88411,51820.7),
# (0.92878,42730.5),
# (0.96637,30303.6),
# (0.99074,16888.8),
# (0.99803,9323.87),
# (0.99911,6605.73),
# (0.99948,5095.6 ),
# (0.99965,4187.67),
# (0.99971,3559.52),
# (0.99978,3117.77),