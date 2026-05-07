# ../scripts/laion/100M_16P_single_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w1043.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/laion
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/laion/gt.10k.bin
query_path=/data/share/users/xyzhi/data/laion/query.10k.fbin
num_threads=4
index_save_path=/data/share/users/xyzhi/data/laion

# this scripts is curently used for 
R=48 # degree limit
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
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
# 10      0.74013 98.2701 us      10176 /s
# 20      0.82492 129.717 us      7709.07 /s
# 30      0.86145 164.441 us      6081.22 /s
# 50      0.89448 232.342 us      4304.01 /s
# 100     0.92105 392.853 us      2545.48 /s
# 200     0.93757 690.478 us      1448.27 /s
# 300     0.94452 972.452 us      1028.33 /s
# 400     0.94901 1234.99 us      809.726 /s
# 500     0.95228 1527.86 us      654.51 /s
# 600     0.95474 1787.13 us      559.558 /s
# 700     0.95742 2018.41 us      495.44 /s

# 48t
# 30      0.86145 44.6051 us      22419 /s
# 50      0.89448 63.4339 us      15764.4 /s
# 100     0.92105 103.776 us      9636.17 /s
# 200     0.93757 179.74 us       5563.59 /s
# 300     0.94452 252.865 us      3954.68 /s
# 400     0.94901 320.622 us      3118.94 /s
# 500     0.95228 400.421 us      2497.37 /s
# 600     0.95474 453.623 us      2204.47 /s
# 700     0.95742 513.658 us      1946.82 /s




# 128t
# 10      0.74013 40.6428 us      24604.6 /s
# 20      0.82492 162.815 us      6141.93 /s
# 30      0.86145 193.682 us      5163.1 /s
# 50      0.89448 283.944 us      3521.83 /s
# 100     0.92105 496.506 us      2014.07 /s
# 200     0.93757 1030.03 us      970.842 /s
# 300     0.94452 1814.88 us      551.001 /s
# 400     0.94901 2591.33 us      385.902 /s
# 500     0.95228 3093.27 us      323.282 /s
# 600     0.95474 4048.57 us      247.001 /s
# 700     0.95742 4786.55 us      208.919 /s