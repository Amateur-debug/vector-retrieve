# ../scripts/text2image/100M_8P_single_anns.sh
# numactl --interleave=all ../scripts/text2image/100M_8P_single_anns.sh
# numactl --localalloc ../scripts/text2image/100M_8P_single_anns.sh

# run in single server
source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w1043_8m.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/text2image
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/text2image-${million}M
query_path=/data/share/users/xyzhi/data/text2image/query.public.100K.fbin
num_threads=192
index_save_path=/home/xyzhi/data/text2image

# this scripts is curently used for 
index_save_dir=${index_save_path}


for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type single \
    --graph_type vamana \
    --data_type float --dist_fn mips \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done

# 8t
# 10      0.4194  38.479 us       25988.2 /s
# 20      0.55439 59.0565 us      16932.9 /s
# 30      0.63181 77.2419 us      12946.3 /s
# 50      0.71862 108.362 us      9228.32 /s
# 100     0.8161  181.868 us      5498.5 /s
# 200     0.88883 317.371 us      3150.89 /s
# 300     0.92031 451.864 us      2213.06 /s
# 400     0.9377  581.414 us      1719.94 /s
# 500     0.94851 708.283 us      1411.86 /s
# 600     0.95629 835.705 us      1196.6 /s
# 700     0.96234 961.573 us      1039.96 /s

# 48t
# 50      0.71862 51.3525 us      19473.2 /s
# 100     0.8161  86.2315 us      11596.7 /s
# 200     0.88883 149.651 us      6682.2 /s
# 300     0.92031 208.8 us        4789.28 /s
# 400     0.9377  265.692 us      3763.76 /s
# 500     0.94851 320.038 us      3124.63 /s
# 600     0.95629 375.478 us      2663.27 /s
# 700     0.96234 428.566 us      2333.36 /s

# 64t
# 10      0.4194  33.8838 us      29512.6 /s
# 20      0.55439 34.988 us       28581.2 /s
# 30      0.63181 48.0399 us      20816 /s
# 50      0.71862 70.356 us       14213.4 /s
# 100     0.8161  120.318 us      8311.27 /s
# 200     0.88883 249.029 us      4015.6 /s
# 300     0.92031 379.576 us      2634.52 /s
# 400     0.9377  445.759 us      2243.36 /s
# 500     0.94851 553.875 us      1805.46 /s
# 600     0.95629 678.782 us      1473.23 /s
# 700     0.96234 638.65 us       1565.8 /s

# 128t
# 50      0.71862 519.453 us      14997.4 
# 60      0.74534 588.275 us      13088.79
# 70      0.76865 644.514 us      11649.68
# 80      0.78812 757.358 us      10514.8 
# 90      0.80325 712.959 us      9546.5  
# 100     0.8161  777.893 us      8687.17 
# 150     0.86242 850.106 us      6285.75  
# 200     0.88883 1129.19 us      4976.7  
# 250     0.90584 1604.66 us      4062.89 
# 300     0.92031 2043.89 us      3485.52 
# 400     0.9377  2491.54 us      2730.6  
# 450     0.94371 421.875         2370.37
# 500     0.94851 458.094         2182.96


# 192t


