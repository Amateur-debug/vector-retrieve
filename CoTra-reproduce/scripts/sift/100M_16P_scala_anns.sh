# ../scripts/sift/100M_16P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/bigann
million=100
base_file=${dataset_path}/base.${million}M.u8bin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/bigann-${million}M
query_path=/data/share/users/xyzhi/data/bigann/query.public.10K.u8bin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/bigann

# this scripts is curently used for 
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type scala_v3 \
    --graph_type scalagraph_v3 \
    --data_type uint8 --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 96 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done

10       0.85605         0.00    68451.891
20       0.92404         0.00    50744.422
30       0.95123         0.00    55227.262
50       0.97612         0.00    46025.684
100      0.99320         0.00    31944.391
200      0.99823         0.00    22918.486
300      0.99914         0.00    16829.492
400      0.99953         0.00    14400.590
