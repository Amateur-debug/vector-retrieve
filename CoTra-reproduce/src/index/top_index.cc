

#include "index/top_index.h"

#include <omp.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "third_party/hnswlib/hnswlib/hnswalg.h"
#include "third_party/hnswlib/hnswlib/hnswlib.h"

bool exists_test(const char *path) {
  if (std::filesystem::exists(path)) {
    return true;
  } else {
    return false;
  }
}

// To avoid redifinition, copy from kmeans file
void get_l2p_map(
    const char *label_pos_path, size_t &vec_size, size_t &part_num, int *&part_size,
    int *&label_pos_map) {
  if (exists_test(label_pos_path)) {
    std::ifstream map_in(label_pos_path, std::ios::binary);
    std::streampos position;

    uint32_t read_vec_num, read_part_num;
    map_in.read((char *)&read_vec_num, sizeof(uint32_t));
    vec_size = read_vec_num;
    printf("vec_size %llu\n", vec_size);
    map_in.read((char *)&read_part_num, sizeof(uint32_t));
    part_num = read_part_num;
    printf("part_num %llu\n", part_num);
    // assert(vec_size == 1e8 && part_num == 4);

    part_size = (int *)malloc(sizeof(uint32_t) * part_num);
    map_in.read((char *)part_size, sizeof(uint32_t) * part_num);
    for (int i = 0; i < part_num; i++) {
      printf("%d ", part_size[i]);
    }
    printf("\n");
    label_pos_map = (int *)malloc(sizeof(uint32_t) * vec_size);
    map_in.read((char *)label_pos_map, sizeof(uint32_t) * vec_size);

    map_in.close();
    return;
  } else {
    printf("Label Pos Map file does not exist !!!\n");
    exit(0);
    return;
  }
}

size_t dataTypeSize(const std::string &dtype) {
  if (dtype == "float") {
    return sizeof(float);
  } else if (dtype == "int8") {
    return sizeof(int8_t);
  } else if (dtype == "uint8") {
    return sizeof(uint8_t);
  } else {
    throw std::invalid_argument("Unknown data type: " + dtype);
  }
}

void readHeader(std::ifstream &ifs, uint32_t &numVectors, uint32_t &dimension) {
  // Read the number of vectors
  if (!ifs.read(reinterpret_cast<char *>(&numVectors), sizeof(uint32_t))) {
    throw std::runtime_error("Failed to read numVectors from file header");
  }
  // Read the dimension
  if (!ifs.read(reinterpret_cast<char *>(&dimension), sizeof(uint32_t))) {
    throw std::runtime_error("Failed to read dimension from file header");
  }
}

//----------------------------------------------------------------------
// Worker function for sampling vectors in a given range
//----------------------------------------------------------------------
template <typename data_t>
void sampleWorker(
    const std::string &dataPath, const std::string &dtype, float samplePercent,
    uint32_t dimension, uint32_t start_idx, uint32_t end_idx,
    std::vector<std::vector<data_t>> &local_samples,
    std::vector<size_t> &local_vec_ids, std::mutex &local_mutex) {
  size_t typeSize = dataTypeSize(dtype);
  std::ifstream ifs(dataPath, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open data file: " + dataPath);
  }

  // Calculate the byte offset where vector data starts
  // Header is 8 bytes (2 uint32_t)
  size_t header_size = sizeof(uint32_t) * 2;
  size_t vector_size = dimension * typeSize;
  size_t offset = header_size + static_cast<size_t>(start_idx) * vector_size;

  // Seek to the start of this thread's assigned vectors
  ifs.seekg(offset, std::ios::beg);
  if (!ifs) {
    throw std::runtime_error(
        "Failed to seek to position " + std::to_string(offset));
  }

  // Prepare random number generator
  std::random_device rd;
  std::mt19937 gen(
      rd() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Buffer to read a chunk of vectors
  const size_t chunkSize = 100000;  // Adjust as needed
  size_t total_vectors = end_idx - start_idx;
  size_t vectors_read = 0;
  size_t cur_id = start_idx;

  while (vectors_read < total_vectors) {
    size_t currentChunkSize = std::min(chunkSize, total_vectors - vectors_read);
    size_t buffer_size = currentChunkSize * dimension * typeSize;
    std::vector<char> buffer(buffer_size);

    if (!ifs.read(buffer.data(), buffer_size)) {
      throw std::runtime_error("Failed while reading data vectors in thread");
    }

    for (size_t i = 0; i < currentChunkSize; ++i) {
      if (dist(gen) < samplePercent) {
        std::vector<data_t> data_vec(dimension);

        size_t offset_vec = i * dimension * typeSize;

        // Copy the vector data to data_vec

        memcpy(
            data_vec.data(), buffer.data() + offset_vec, dimension * typeSize);

        // Safely add the sampled vector to the shared container
        {
          std::lock_guard<std::mutex> lock(local_mutex);
          local_samples.emplace_back(std::move(data_vec));
          local_vec_ids.push_back(cur_id);
        }
      }
      cur_id++;
    }

    vectors_read += currentChunkSize;
  }

  ifs.close();
}

//----------------------------------------------------------------------
// sampleVectorsFromHugeFile
//    - Reads the header (n, d).
//    - Divides the vectors among multiple threads.
//    - Each thread samples vectors in its assigned range.
//    - Returns the sampled set of vectors in a 2D vector
//----------------------------------------------------------------------
template <typename data_t>
std::tuple<std::vector<std::vector<data_t>>, std::vector<size_t>>
sampleVectorsFromHugeFile(
    const std::string &dataPath, float samplePercent, const std::string &dtype,
    int num_threads) {
  // Open the file to read header
  std::ifstream ifs(dataPath, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open data file: " + dataPath);
  }

  // Read header to get (n, dimension)
  uint32_t numVectors = 0;
  uint32_t dimension = 0;
  readHeader(ifs, numVectors, dimension);
  ifs.close();  // We'll reopen it in threads

  std::cout << "[INFO] File header: numVectors=" << numVectors
            << ", dimension=" << dimension << std::endl;

  if (num_threads < 1) {
    num_threads = 1;
  }
  if (static_cast<size_t>(num_threads) > numVectors) {
    num_threads = numVectors;
  }

  // Calculate the range of vectors for each thread
  std::vector<std::pair<uint32_t, uint32_t>> thread_ranges;
  uint32_t vectors_per_thread = numVectors / num_threads;
  uint32_t remainder = numVectors % num_threads;
  uint32_t current_start = 0;

  for (int i = 0; i < num_threads; ++i) {
    uint32_t current_end =
        current_start + vectors_per_thread + (i < remainder ? 1 : 0);
    thread_ranges.emplace_back(std::make_pair(current_start, current_end));
    current_start = current_end;

    std::cout << "[INFO] Thread " << i << " assigned vectors "
              << thread_ranges.back().first << " to "
              << thread_ranges.back().second << std::endl;
  }

  // Shared container for sampled vectors
  std::vector<std::vector<data_t>> sampledVectors;
  sampledVectors.reserve(
      static_cast<size_t>(numVectors * samplePercent * 1.1));  // Estimate

  std::vector<size_t> sampledVectorIds;
  sampledVectorIds.reserve(sampledVectors.capacity());

  // Mutex for synchronizing access to sampledVectors
  std::mutex sampled_mutex;

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(
        sampleWorker<data_t>, std::ref(dataPath), std::ref(dtype),
        samplePercent, dimension, thread_ranges[i].first,
        thread_ranges[i].second, std::ref(sampledVectors),
        std::ref(sampledVectorIds), std::ref(sampled_mutex));
  }

  // Wait for all threads to finish
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  std::cout << "[INFO] Finished sampling. Total " << sampledVectors.size()
            << " vectors kept (out of " << numVectors << ")." << std::endl;
  return std::make_tuple(sampledVectors, sampledVectorIds);
}

//----------------------------------------------------------------------
// Build HNSW index on the sampled vectors and save it
//----------------------------------------------------------------------
template <typename dist_t, typename data_t>
void buildAndSaveHnswIndex(
    const std::vector<std::vector<data_t>> &sampledVectors,
    const std::vector<size_t> &sampledVectorIds, size_t dimension,
    const std::string &indexPath, int M, int efConstruction,
    hnswlib::SpaceInterface<dist_t> *space, const std::string label_trans_file,
    std::string distType, std::string dataType) {

  // transform label to position
  std::cout << "Read Label Pos map file: "<< label_trans_file << "\n";
  int *kmeans_part_size;
  int *kmeans_label_pos_map, *kmeans_pos_label_map;
  size_t kmeans_vec_size, kmeans_part_num;
  get_l2p_map(
      label_trans_file.c_str(), kmeans_vec_size, kmeans_part_num,
      kmeans_part_size, kmeans_label_pos_map);
  
  // If you are using L2 distance, create L2Space

  // hnswlib::L2Space l2space(dimension);

  // The maximum number of elements = size of sampledVectors
  size_t maxElements = sampledVectors.size();
  if (maxElements == 0) {
    throw std::runtime_error("No vectors to build index on!");
  }

  std::cout << "[INFO] Initializing HNSW index with maxElements=" << maxElements
            << ", M=" << M << ", efConstruction=" << efConstruction
            << std::endl;

  // Create the index
  hnswlib::HierarchicalNSW<dist_t> hnswIndex(
      space, maxElements, M, efConstruction, 107);

  // Optional: set ef (construction time search parameter)
  hnswIndex.setEf(efConstruction);

  // Add points
  std::cout << "[INFO] Adding points to the HNSW index..." << std::endl;

  for (size_t i = 0; i < 1; ++i) {
    hnswIndex.addPoint((void *)sampledVectors[i].data(), i);
  }
  // size_t idx = 0;
#pragma omp parallel for
  for (size_t i = 1; i < maxElements; i++) {
    hnswIndex.addPoint((void *)sampledVectors[i].data(), i);
    // if (i % 100000 == 0 && i != 0) {
    //     std::cout << "[INFO] Added " << i << " points..." << std::endl;
    // }
  }

  // Save index in rdma anns top index format

  std::ofstream output(indexPath, std::ios::binary);
  // std::ofstream output_id_map(indexPath + ".id_map", std::ios::binary);

  if (!output.is_open()) {
    throw std::runtime_error("Failed to open output file: " + indexPath);
  }

  // if (!output_id_map.is_open()) {
  //     throw std::runtime_error("Failed to open output file: " + indexPath +
  //     ".id_map");
  // }

  // output_id_map.write((char*)sampledVectorIds.data(),
  // sampledVectorIds.size() * sizeof(size_t));

  if (sampledVectorIds.size() != maxElements) {
    throw std::runtime_error("Sampled vector IDs size mismatch!");
  }

  std::cout << "[INFO] Saved id map to: " << indexPath + ".id_map" << std::endl;

  if (hnswIndex.cur_element_count != maxElements) {
    throw std::runtime_error("Element count mismatch!");
  }

  std::unordered_map<int, int> level_ele_count;
  for (int i = 0; i < maxElements; ++i) {
    level_ele_count[hnswIndex.element_levels_[i]] += 1;
    if (hnswIndex.element_levels_[i] == hnswIndex.maxlevel_ + 1) {
      throw std::runtime_error(
          "element " + std::to_string(i) + " level " +
          std::to_string(hnswIndex.element_levels_[i]) + " is out of range!");
    }
  }
  int level_accum = 0;
  int expect_level_0_count = 0;

  uint32_t level_num = hnswIndex.maxlevel_ + 1;

  std::cout << "Enter point: " << hnswIndex.enterpoint_node_ << std::endl;
  for (int i = 0; i < level_num; i++) {
    level_accum += level_ele_count[i];

    if (i > 0) {
      expect_level_0_count += level_ele_count[i];
    }
    std::cout << "level " << i << " : " << level_ele_count[i]
              << " accumulated count: " << level_accum << std::endl;
  }
  std::cout << "Upper layer points count: " << expect_level_0_count
            << std::endl;

  // num vecs
  output.write(reinterpret_cast<const char *>(&maxElements), sizeof(uint32_t));
  // num levels
  output.write(reinterpret_cast<const char *>(&level_num), sizeof(uint32_t));

  for (int i = hnswIndex.maxlevel_; i >= 0; --i) {
    output.write(
        reinterpret_cast<const char *>(&level_ele_count[i]), sizeof(uint32_t));
  }

  output.write(
      reinterpret_cast<const char *>(&hnswIndex.size_data_per_element_),
      sizeof(size_t));

  output.write(
      reinterpret_cast<const char *>(&hnswIndex.maxM0_), sizeof(size_t));

  output.write(
      reinterpret_cast<const char *>(&hnswIndex.label_offset_), sizeof(size_t));

  std::cout << "maxElements = " << maxElements << std::endl;
  std::cout << "maxlevel_ = " << hnswIndex.maxlevel_ << std::endl;
  std::cout << "size_data_per_element_ = " << hnswIndex.size_data_per_element_
            << std::endl;
  std::cout << "maxM0_ = " << hnswIndex.maxM0_ << std::endl;
  std::cout << "label_offset_ = " << hnswIndex.label_offset_ << std::endl;

  if (hnswIndex.label_offset_ !=
      (hnswIndex.maxM0_ + 1) * 4 + dimension * dataTypeSize(dataType)) {
    size_t expec_ofs = (hnswIndex.maxM0_ + 1) * 4 + dimension * dataTypeSize(dataType);
    size_t data_size = dataTypeSize(dataType);
    std::cout<<"expect offset " <<expec_ofs <<"data_size "<<data_size<<"dimension "<<dimension<< "\n";
    throw std::runtime_error("label offset mismatch!");
  }

  // write all data in level 0
  for (int i = 0; i < maxElements; ++i) {
    char *element_block =
        hnswIndex.data_level0_memory_ + i * hnswIndex.size_data_per_element_;
    std::vector<char> buffer(hnswIndex.size_data_per_element_);

    memcpy(buffer.data(), element_block, hnswIndex.size_data_per_element_);

    char *ptr = buffer.data();
    uint64_t *label_ptr = (uint64_t *)(ptr + hnswIndex.label_offset_);

    uint64_t label = kmeans_label_pos_map[sampledVectorIds[*label_ptr]];
    memcpy(label_ptr, &label, 8);
    // std::cout << "element " << i << " label " << sampledVectorIds[i] << " ,
    // kmean reorder label: " << kmeans_label_pos_map[sampledVectorIds[i]] <<
    // std::endl;

    output.write(buffer.data(), hnswIndex.size_data_per_element_);
  }

  uint32_t size_data_per_element = hnswIndex.size_data_per_element_;

  uint32_t index_size = 4 + 4 + level_num * 4 + sizeof(size_t) * 3 +
                        (maxElements * size_data_per_element);

  std::cout << "Expected index size = " << index_size
            << ", written index size = " << output.tellp() << std::endl;
  output.close();

  std::cout << "[INFO] HNSW index built and saved to: " << indexPath
            << ", total " << hnswIndex.cur_element_count
            << " points, expected: " << maxElements << std::endl;
}

template <typename data_t>
void run_sample_and_build(
    std::string dataPath, std::string dataType, std::string distType,
    std::string indexPath, std::string string_samplePercent,
    std::string label_trans_file, int M, int efConstruction, int num_threads,
    int part_num) {

  float samplePercent = atof(string_samplePercent.c_str());
  std::cout << "       Part num: " << part_num << std::endl;

  std::cout << "[INFO] label trans file: " << label_trans_file << std::endl;
  std::cout << "[INFO] output index file: " << indexPath << std::endl;

  auto sample_res =
      sampleVectorsFromHugeFile<data_t>(dataPath, samplePercent, dataType, 16);

  std::vector<std::vector<data_t>> sampledVectors = std::get<0>(sample_res);
  std::vector<size_t> sampledVectorIds = std::get<1>(sample_res);

  // verify
  std::ifstream ifs(dataPath, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open data file: " + dataPath);
  }
  uint32_t npts, dim;
  readHeader(ifs, npts, dim);
  for (int i = 0; i < 10; ++i) {
    std::cout << "Verifying sampled vector " << i
              << " id: " << sampledVectorIds[i] << std::endl;
    size_t offset = 8 + sampledVectorIds[i] * dim * dataTypeSize(dataType);
    ifs.seekg(offset, std::ios::beg);
    std::vector<data_t> vec(dim);
    ifs.read(
        reinterpret_cast<char *>(vec.data()), dim * dataTypeSize(dataType));
    if (memcmp(
            vec.data(), sampledVectors[i].data(),
            dim * dataTypeSize(dataType)) != 0) {
      throw std::runtime_error("Sampled vector mismatch!");
    }
  }
  ifs.close();

  // If the file header says dimension = D, we can retrieve from the first
  // vector
  //   (assuming we sampled at least one).
  //   If no vectors are sampled, buildAndSaveHnswIndex() will throw.
  size_t dimension = sampledVectors.empty() ? 0 : sampledVectors[0].size();

  // 3. Build and save the HNSW index (single-threaded)
  std::cout << "Build save index with "<<dataType<<"\n ";
  if (dataType == "float") {
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dimension);
    buildAndSaveHnswIndex<float, data_t>(
        sampledVectors, sampledVectorIds, dimension, indexPath, M,
        efConstruction, space, label_trans_file, distType, dataType);
  } else if (dataType == "int8") {
    hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(dimension);
    buildAndSaveHnswIndex<int, data_t>(
        sampledVectors, sampledVectorIds, dimension, indexPath, M,
        efConstruction, space, label_trans_file, distType, dataType);
  } else if (dataType == "uint8") {
    hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(dimension);
    buildAndSaveHnswIndex<int, data_t>(
        sampledVectors, sampledVectorIds, dimension, indexPath, M,
        efConstruction, space, label_trans_file, distType, dataType);
  } else {
    throw std::invalid_argument("Unknown data type: " + dataType);
  }
}

void HNSWTopIndex::build_hnsw_top_index(commandLine &cmd) {
  std::string dataPath = cmd.getOptionValue("--data_path", "data.bin");
  std::string dataType = cmd.getOptionValue(
      "--data_type", "uint8");  // e.g. "float", "int8", or "uint8"
  std::string distType = cmd.getOptionValue("--dist_type", "l2");
  std::string indexPath = cmd.getOptionValue("--index_path", "hnsw_index.bin");
  std::string string_samplePercent =
      cmd.getOptionValue("--sample_percent", "0.01");  // e.g. 1%
  int M = cmd.getOptionIntValue("--topindex_deg", 16);
  int efConstruction = cmd.getOptionIntValue("-e", 200);
  int num_threads = cmd.getOptionIntValue("--num_threads", 1);
  int part_num = cmd.getOptionIntValue("-p", 1);
  std::string label_trans_file =
      "/data/share/users/xyzhi/data/bigann/vamana/balance_kmeans_" +
      std::to_string(part_num) + "_part/100M_" + std::to_string(part_num) +
      "_labelpos_map.bin";

  float samplePercent = atof(string_samplePercent.c_str());
  std::cout << "[INFO] Starting HNSW Top index sampling build with the "
               "following parameters:"
            << std::endl;
  std::cout << "       Data Path: " << dataPath << std::endl;
  std::cout << "       Data Type: " << dataType << std::endl;
  std::cout << "       Dist Type: " << distType << std::endl;
  std::cout << "       Index Path: " << indexPath << std::endl;
  std::cout << "       Sample Percent: " << samplePercent * 100 << "%"
            << std::endl;
  std::cout << "       M: " << M << std::endl;
  std::cout << "       efConstruction: " << efConstruction << std::endl;
  std::cout << "       Num Threads: " << num_threads << std::endl;

  omp_set_num_threads(num_threads);

  if (dataType == "float") {
    run_sample_and_build<float>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else if (dataType == "int8") {
    run_sample_and_build<int8_t>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else if (dataType == "uint8") {
    run_sample_and_build<uint8_t>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else {
    throw std::invalid_argument("Unknown data type: " + dataType);
  }
}

void HNSWTopIndex::build_hnsw_top_index(
    IndexParameter &index_param, commandLine &cmd) {
  std::cout << "Start build hnsw top index ..." << std::endl;
  std::string dataPath = index_param.data_path;
  std::string dataType =
      index_param.data_type;  // e.g. "float", "int8", or "uint8"
  printf("data type : %s \n", dataType.c_str());
  printf("fsz: %llu datasz %llu\n", sizeof(float), dataTypeSize(dataType));
  std::string distType = index_param.dist_fn;
  std::string indexPath = index_param.top_index_path;
  std::string string_samplePercent = index_param.top_sample_percent;
      // cmd.getOptionValue("--sample_percent", "0.01");  // e.g. 1%
  int M = index_param.topindex_deg;
  //cmd.getOptionIntValue("--topindex_deg", 16);
  int efConstruction = 200; //cmd.getOptionIntValue("-e", 200);
  int num_threads = index_param.thread_num;//cmd.getOptionIntValue("-t", 1);
  int part_num = index_param.num_parts;
  std::string label_trans_file = index_param.temp_label2pos_map_file;

  float samplePercent = atof(string_samplePercent.c_str());
  std::cout << "[INFO] Starting HNSW Top index sampling build with the "
               "following parameters:"
            << std::endl;
  std::cout << "       Data Path: " << dataPath << std::endl;
  std::cout << "       Data Type: " << index_param.data_type << std::endl;
  std::cout << "       Dist Type: " << index_param.data_type << std::endl;
  std::cout << "       Index Path: " << indexPath << std::endl;
  std::cout << "       Sample Percent: " << samplePercent * 100 << "%"
            << std::endl;
  std::cout << "       M: " << M << std::endl;
  std::cout << "       efConstruction: " << efConstruction << std::endl;
  std::cout << "       Num Threads: " << num_threads << std::endl;

  omp_set_num_threads(num_threads);

  if (index_param.data_type == "float") {
    run_sample_and_build<float>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else if (index_param.data_type == "int8") {
    run_sample_and_build<int8_t>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else if (index_param.data_type == "uint8") {
    run_sample_and_build<uint8_t>(
        dataPath, dataType, distType, indexPath, string_samplePercent,
        label_trans_file, M, efConstruction, num_threads, part_num);
  } else {
    throw std::invalid_argument("Unknown data type: " + index_param.data_type);
  }
}