#include "coromem/include/utils.h"
#include "index/kmeans.h"

int main(int argc, char **argv) {
  commandLine cmd(
      argc, argv,
      "usage: -m <million_num> -p <part_num> -d <path_data> -o <output_path>");

  auto million_num = cmd.getOptionIntValue("-m", 10);
  int vec_size = million_num * 1e6;  // 10M/100M/1B
  int part_num = cmd.getOptionIntValue("-p", 2);
  auto path_data_str = cmd.getOptionValue("-d", "none");
  auto path_data = path_data_str.c_str();
  auto output_path_str = cmd.getOptionValue("-o", "none");
  auto output_path = output_path_str.c_str();
  auto pos_label_path_str = cmd.getOptionValue("-k", "none");
  auto pos_label_path = pos_label_path_str.c_str();
  auto label_pos_path_str = cmd.getOptionValue("-l", "none");
  auto label_pos_path = label_pos_path_str.c_str();

  int *part = (int *)malloc(vec_size * 2 * 4);
  memset(part, -1, vec_size * 2 * 4);
  /*
  part: <fisrt_part_id, second_part_id(if have)> [vec_size]
  part[2*x]: the first part id of vector x.

  label_pos_map[x]: the vector x(label) new position.
  pos_label_map[x]: the label vector in postition x has.
  */
  int *label_pos_map = (int *)malloc(vec_size * 4);
  memset(label_pos_map, -1, vec_size * 4);
  int *pos_label_map = (int *)malloc(vec_size * 4);
  memset(pos_label_map, -1, vec_size * 4);
  int *part_offset = (int *)malloc(vec_size * 4);
  memset(part_offset, -1, vec_size * 4);
  int *part_size = (int *)malloc(part_num * 4);
  memset(part_size, -1, part_num * 4);

  char partition_path[part_num][1024];
  for (int i = 0; i < part_num; i++) {
    // snprintf(
    //     partition_path[i], sizeof(partition_path[i]),
    //     "/data/share/users/xyzhi/data/bigann/%dM_kmeans/merged_%d_uint32/"
    //     "merged_%d-%d_uint32.bin",
    //     million_num, part_num, i, part_num);

    // balance kmeans 4-parts
    // snprintf(
    //     partition_path[i], sizeof(partition_path[i]),
    //     "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans/"
    //     "merged_index_mem.index_tempFiles_meta_index_subshard-%d_ids_uint32."
    //     "bin",
    //     i);

    // TODO: cmd
    // balance kmeans 8-parts
    snprintf(
        partition_path[i], sizeof(partition_path[i]),
        "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_8_part/"
        "merged_index_mem.index_tempFiles_meta_index_subshard-%d_ids_uint32."
        "bin",
        i);
  }

  //   read_partition(
  //       partition_path, part, label_pos_map, pos_label_map, part_offset,
  //       part_size, million_num, vec_size, part_num);

  //   write_map(
  //       pos_label_map, label_pos_map, part_size, vec_size, part_num,
  //       pos_label_path, label_pos_path);

  //   kmeans_reorder(
  //       part, label_pos_map, pos_label_map, part_offset, part_size,
  //       million_num, vec_size, part_num, path_data, output_path);

  int *tmp_part_size, *tmp_pos_label;
  int tmp_vec_size, tmp_part_num;
  get_pos_label_map(
      pos_label_path, tmp_vec_size, tmp_part_num, tmp_part_size, tmp_pos_label);

  verify_reorder_data(path_data, output_path, tmp_pos_label, vec_size);
}
