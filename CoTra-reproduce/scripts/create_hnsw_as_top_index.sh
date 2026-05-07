# Note: Please change the path to run the following command

make -j && \
./tests/index/create_hnsw_top \
--data_path /data/share/users/xyzhi/data/bigann/base.100M.u8bin \
    --data_type uint8 \
    --dist_type l2 \
    --index_path /data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.01_top_idnex.index \
    --sample_percent 0.01 \
    -M 8 \
    -e 200 \
    -p 4 \
    --num_threads 24

# make -j && \
# ./tests/index/create_hnsw_top \
# --data_path /data/share/users/xyzhi/data/bigann/base.1M.u8bin \
#     --data_type uint8 \
#     --dist_type l2 \
#     --index_path ./test_hnsw_top.index \
#     --sample_percent 0.01 \
#     -M 8 \
#     -e 50 \
#     --num_threads 4


# make -j && \
# ./tests/index/create_hnsw_top \
# --data_path /data/share/users/xyzhi/data/bigann/base.100M.u8bin \
#     --data_type uint8 \
#     --dist_type l2 \
#     --index_path /data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.001_top_idnex.index \
#     --sample_percent 0.001 \
#     -M 8 \
#     -e 150 \
#     -p 4 \
#     --num_threads 4

# make -j && \
# ./tests/index/create_hnsw_top \
# --data_path /data/share/users/xyzhi/data/bigann/base.100M.u8bin \
#     --data_type uint8 \
#     --dist_type l2 \
#     --index_path /data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.0001_top_idnex.index \
#     --sample_percent 0.0001 \
#     -M 8 \
#     -e 100 \
#     -p 4 \
#     --num_threads 4


# make -j && \
# ./tests/index/create_hnsw_top \
# --data_path /data/share/users/xyzhi/data/bigann/base.100M.u8bin \
#     --data_type uint8 \
#     --dist_type l2 \
#     --index_path /data/share/users/xyzhi/data/bigann/hnsw_top_index/base_100M_hnsw_0.1_top_idnex.index \
#     --sample_percent 0.1 \
#     -M 24 \
#     -e 500 \
#     -p 4 \
#     --num_threads 32
