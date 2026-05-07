num_threads=8
topk=10
# index_save_path=/data/share/users/xyzhi/data/bigann
# index_save_dir=${index_save_path}/sift
# evaluation_save_dir=/home/v-chenmeng/projects/ann/experiments/merge_quality_test/sift
# experiment_save_dir=/home/v-chenmeng/projects/ann/experiments/remote_access/sift
# mkdir -p ${evaluation_save_dir}

dataset_path=/data/share/users/xyzhi/data/bigann
query_file=${dataset_path}/query.public.10K.u8bin

index_save_path=/data/share/users/xyzhi/rdma-anns/build
index_save_dir=${index_save_path}/tmp

ln -s ${dataset_path}/base.1M.u8bin ${index_save_dir}/merged_index_mem.index.data

./third_party/diskann/apps/search_memory_index  \
--data_type uint8 --dist_fn l2 \
--index_path_prefix ${index_save_dir}/merged_index_mem.index \
--query_file ${query_file}  \
--gt_file ${dataset_path}/vamana/gt.base.1M.query.10K.bin \
-K ${topk} -L 50 100 200 300 400 450 \
-T ${num_threads} --result_path /data/share/users/xyzhi/rdma-anns/build/tmp


# ./apps/search_memory_index  \
# --data_type uint8 --dist_fn l2 \
# --index_path_prefix /data/share/users/xyzhi/data/bigann/vamana \
# --query_file /data/share/users/xyzhi/data/bigann/query.public.10K.u8bin  \
# --gt_file /data/share/users/xyzhi/data/bigann/vamana/gt.query.10K.base.100M.bin \
# -K ${topk} -L 50 100 200 300 350 400 450 \
# -T ${num_threads} --result_path /data/share/users/xyzhi/data/bigann/vamana/results/100M
# --evaluation_save_path ${evaluation_save_dir}/index_sift_100M_one_shot_R64_L500_A1.2_T40_k_${topk}_searchT_${num_threads}.csv


# ./apps/search_memory_index  \
#     --data_type uint8 --dist_fn l2 \
#     --index_path_prefix /raid/chenmeng/ann_index/sift/meta_merged_index_sift_100k_R32_L500_A1.2_B_10_M_3_T32_kbase_1/merged_index_mem.index_tempFiles_meta_index_mem.index \
#     --query_file /raid/chenmeng/ann_dataset/sift/query.public.10K.u8bin \
#     --gt_file /raid/chenmeng/ann_index/sift/meta_merged_index_sift_100k_R32_L500_A1.2_B_10_M_3_T32_kbase_1/meta_index_test.gt \
#     -K 10 -L 10 20 30 40 50 60 70 80 90 100 110 120 150 200 250 300 350 \
#     -T ${num_threads} --result_path /dev/null 

# ./apps/search_memory_index  \
#     --data_type uint8 --dist_fn l2 \
#     --index_path_prefix /raid/chenmeng/ann_index/sift/meta_merged_index_sift_100k_R32_L500_A1.2_B_10_M_3_T32_kbase_1/merged_index_mem.index \
#     --query_file /raid/chenmeng/ann_dataset/sift/query.public.10K.u8bin \
#     --gt_file /raid/chenmeng/ann_dataset/sift/gt.query.10K.base.100K.bin \
#     -K 10 -L 10 20 30 40 50 60 70 80 90 100 110 120 150 200 250 300 350 \
#     -T ${num_threads} --result_path /dev/null 
