# ../scripts/laion/100M_16P_scala_anns.sh

source ../scripts/restart_memcache.sh
CONF_FILE="../scripts/ip_w33-48.conf"
clear_memcache "$CONF_FILE"

dataset_path=/data/share/users/xyzhi/data/laion
million=100
base_file=${dataset_path}/base.${million}M.fbin
gt_file=/data/share/users/xyzhi/data/laion/gt.10k.bin
query_path=/data/share/users/xyzhi/data/laion/query.10k.fbin
num_threads=8
index_save_path=/data/share/users/xyzhi/data/laion

R=48 # degree limit
index_save_dir=${index_save_path}/scalagraph/test_${million}M_16P
for m in 1
do
    mkdir -p ${index_save_dir}
    ./tests/scala_anns \
    --config_file ${CONF_FILE} \
    --app_type scala_v3 \
    --graph_type scalagraph_v3 \
    --data_type float --dist_fn l2 \
    --data_path ${base_file} \
    --query_path ${query_path} \
    --gt_path ${gt_file} \
    --index_path_prefix ${index_save_dir}/merged_index \
    -R ${R} -L 500 -B 100 -M 120 -T ${num_threads} \
    --scala_v3 -s ${million} -t ${num_threads}
done



# 10       0.82112         0.00    43940.207
# 20       0.87954         0.00    49100.719
# 30       0.90343         0.00    32968.590
# 50       0.92867         0.00    28029.105
# 100      0.94974         0.00    17897.861
# 200      0.96066         0.00    11151.679
# 300      0.96515         0.00    7878.693
# 400      0.96768         0.00    6219.080
# 500      0.96934         0.00    5147.939
# 600      0.97034         0.00    3864.687

\begin{table*}[!t]
  \small
  \centering
  \caption{The throughput (QPS) at 0.95 recall and speedup over the single machine (8 threads) for each dataset and solution.}
  \label{tab:new exp-jct-util}
  
  \begin{minipage}{0.48\textwidth}
    \centering
    \scalebox{0.85}{
      \begin{tabular}{lcccc}
        \toprule
        & \multicolumn{4}{c}{\textbf{8 Machine}} \\
        \cmidrule(r){2-5}
        System & SIFT & DEEP  & Text2Image & LAION\\
        \midrule
        \textbf{Single} & 30.5K (3.4$\times$) & 15.0K (3.7$\times$) &  1.8K (1.5$\times$) & 2.7K(1.7$\times$)\\
        \midrule
        \textbf{Milvus} & 3.0K (0.3$\times$) & 3.7K (0.9$\times$) & 0.7K (0.6$\times$) & --\\
        \midrule
        \textbf{Global} & 6.3K (0.7$\times$) & 5.9K(1.4$\times$) & 1.4K (1.2$\times$) & 1.0K(0.6$\times$)\\
        \midrule
        \textbf{Shard} & 35.1K (3.9$\times$) & 21.1K (5.1$\times$) &  3.3K (2.9$\times$) & 4.5K(2.8$\times$)\\
        \midrule
        \textbf{CoTra} & 68.9K (7.6$\times$) & 32.1K (7.6$\times$) &  7.2K (6.2$\times$) & 9.9K(6.2$\times$)\\
        \bottomrule
      \end{tabular}
    }
  \end{minipage}
  \hfill
  \begin{minipage}{0.48\textwidth}
    \centering
    \scalebox{0.85}{
      \begin{tabular}{lcccc}
        \toprule
        & \multicolumn{4}{c}{\textbf{16 Machine}} \\
        \cmidrule(r){2-5}
        System & SIFT & DEEP & Text2Image & LAION \\
        \midrule
        \textbf{Single} & 30.3K (3.4$\times$) & 15.5K (3.8$\times$) &  2.1K (2.1$\times$) & 3.1K(1.9$\times$)\\
        \midrule
        \textbf{Milvus} & 3.5K (0.4$\times$) & 3.6K (0.9$\times$) & 0.9K (0.9$\times$) & --\\
        \midrule
        \textbf{Global} & 9.7K (1.1$\times$) & 7.4K (1.8$\times$) & 1.9K (1.9$\times$) & 1.3K(0.8$\times$)\\
        \midrule
        \textbf{Shard} & 40.8K (4.5$\times$) & 26.0K (6.5$\times$) & 4.3K (4.4$\times$) & 5.0K(3.1$\times$)\\
        \midrule
        \textbf{CoTra} & 116.6K(12.7$\times$) & 55.3K (13.4$\times$) & 9.6K (9.8$\times$) & 17.9K (11.2$\times$)\\
        \bottomrule
      \end{tabular}
    }
  \end{minipage}
\end{table*}
