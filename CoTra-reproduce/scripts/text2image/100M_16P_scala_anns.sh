# ../scripts/text2image/100M_16P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/text2image
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/GT_all/GT_${million}M/text2image-${million}M
query_path=/data/share/users/xyzhi/data/text2image/query.public.100K.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/text2image

# this scripts is curently used for 
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type scala_v3 \
    --graph_type scalagraph_v3 \
    --data_type float --dist_fn mips \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R 48 -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done

10       0.53763         0.00    77639.148
20       0.65389         0.00    76959.781
30       0.71798         0.00    54662.434
50       0.78717         0.00    41197.699
100      0.86515         0.00    29601.914
200      0.92296         0.00    17824.740
300      0.94605         0.00    12382.843
400      0.95847         0.00    9606.176
500      0.96717         0.00    7930.598