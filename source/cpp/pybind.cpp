#include <pybind11/numpy.h> // 提供 NumPy ndarray 与 C++ 数组的互转支持
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 提供 STL 容器（vector、map 等）与 Python 类型的自动转换支持

#include "cluster.h"
#include "hnsw.h"
#include "payload.h"
#include "sdk.h"
#include "transport.h"

namespace py = pybind11;

PYBIND11_MODULE(vector_retrieve, m)
{
    m.doc() = "Pybind11 bindings for Cluster and related classes";

    // 绑定通信配置类，用于配置集群节点间的传输模式（TCP/RDMA）
    py::class_<TransportConfig>(m, "TransportConfig")
        .def(py::init<>())
        .def_readwrite("comm_mode", &TransportConfig::comm_mode)
        .def_readwrite("ib_dev", &TransportConfig::ib_dev)
        .def_readwrite("ib_port", &TransportConfig::ib_port)
        .def_readwrite("gid_index", &TransportConfig::gid_index);

    py::class_<Payload<float>>(m, "PayloadFloat")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<float>::size_bits)
        .def_readwrite("require_id", &Payload<float>::require_id)
        .def_readwrite("continue_search", &Payload<float>::continue_search)
        .def_readwrite("origin_node_id", &Payload<float>::origin_node_id)
        .def_readwrite("remote_transfer_cnt", &Payload<float>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_data", &Payload<float>::remote_transfer_data)
        .def_readwrite("send_timestamp", &Payload<float>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<float>::distance_cnt)
        .def_readwrite("results", &Payload<float>::results)
        .def("topk_ids", &Payload<float>::topk_ids, py::arg("k"));

    // 绑定 Payload 类（double 版本），用于在 Python 侧管理搜索中间状态
    py::class_<Payload<double>>(m, "PayloadDouble")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<double>::size_bits)
        .def_readwrite("require_id", &Payload<double>::require_id)
        .def_readwrite("continue_search", &Payload<double>::continue_search)
        .def_readwrite("origin_node_id", &Payload<double>::origin_node_id)
        .def_readwrite("remote_transfer_cnt", &Payload<double>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_data", &Payload<double>::remote_transfer_data)
        .def_readwrite("send_timestamp", &Payload<double>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<double>::distance_cnt)
        .def_readwrite("results", &Payload<double>::results)
        .def("topk_ids", &Payload<double>::topk_ids, py::arg("k"));

    // 绑定 Payload 类（int32_t 版本）
    py::class_<Payload<int32_t>>(m, "PayloadInt32")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<int32_t>::size_bits)
        .def_readwrite("require_id", &Payload<int32_t>::require_id)
        .def_readwrite("continue_search", &Payload<int32_t>::continue_search)
        .def_readwrite("origin_node_id", &Payload<int32_t>::origin_node_id)
        .def_readwrite("remote_transfer_cnt", &Payload<int32_t>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_data", &Payload<int32_t>::remote_transfer_data)
        .def_readwrite("send_timestamp", &Payload<int32_t>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<int32_t>::distance_cnt)
        .def_readwrite("results", &Payload<int32_t>::results)
        .def("topk_ids", &Payload<int32_t>::topk_ids, py::arg("k"));

    // 绑定 Payload 类（uint32_t 版本）
    py::class_<Payload<uint32_t>>(m, "PayloadUInt32")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<uint32_t>::size_bits)
        .def_readwrite("require_id", &Payload<uint32_t>::require_id)
        .def_readwrite("continue_search", &Payload<uint32_t>::continue_search)
        .def_readwrite("origin_node_id", &Payload<uint32_t>::origin_node_id)
        .def_readwrite("remote_transfer_cnt", &Payload<uint32_t>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_data", &Payload<uint32_t>::remote_transfer_data)
        .def_readwrite("send_timestamp", &Payload<uint32_t>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<uint32_t>::distance_cnt)
        .def_readwrite("results", &Payload<uint32_t>::results)
        .def("topk_ids", &Payload<uint32_t>::topk_ids, py::arg("k"));

    // 绑定 HNSWGraph 类（float 版本），提供建图、查询、序列化等核心操作的 Python 接口
    py::class_<HNSWGraph<float>>(m, "HNSWGraphFloat")
        .def(py::init<int, int, int, int, int, size_t, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("max_elements") = 100000, py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<float>::dim)
        .def_readwrite("m", &HNSWGraph<float>::m)
        .def_readwrite("ef_construction", &HNSWGraph<float>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<float>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<float>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<float>::max_level)
        .def_readwrite("seed", &HNSWGraph<float>::seed)
        .def_readwrite("entry_point_id", &HNSWGraph<float>::entry_point_id)
        .def("get_vector", &HNSWGraph<float>::get_vector, py::arg("vid"))
        .def("get_neighbors", &HNSWGraph<float>::get_neighbors, py::arg("vid"), py::arg("level") = 0)
        .def("get_all_vectors", &HNSWGraph<float>::get_all_vectors)
        .def("get_all_neighbors", &HNSWGraph<float>::get_all_neighbors)
        .def("get_cur_element_count", &HNSWGraph<float>::get_cur_element_count)
        .def("add_vector", &HNSWGraph<float>::add_vector, py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())  // 释放 Python GIL 锁，允许 C++ 层真正的多线程并行执行
        .def("query", &HNSWGraph<float>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<float>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<float>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 HNSWGraph 类（double 版本），接口与 float 版本完全一致
    py::class_<HNSWGraph<double>>(m, "HNSWGraphDouble")
        .def(py::init<int, int, int, int, int, size_t, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("max_elements") = 100000, py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<double>::dim)
        .def_readwrite("m", &HNSWGraph<double>::m)
        .def_readwrite("ef_construction", &HNSWGraph<double>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<double>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<double>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<double>::max_level)
        .def_readwrite("seed", &HNSWGraph<double>::seed)
        .def_readwrite("entry_point_id", &HNSWGraph<double>::entry_point_id)
        .def("get_vector", &HNSWGraph<double>::get_vector, py::arg("vid"))
        .def("get_neighbors", &HNSWGraph<double>::get_neighbors, py::arg("vid"), py::arg("level") = 0)
        .def("get_all_vectors", &HNSWGraph<double>::get_all_vectors)
        .def("get_all_neighbors", &HNSWGraph<double>::get_all_neighbors)
        .def("get_cur_element_count", &HNSWGraph<double>::get_cur_element_count)
        .def("add_vector", &HNSWGraph<double>::add_vector, py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())  // 释放 Python GIL 锁，允许 C++ 层真正的多线程并行执行
        .def("query", &HNSWGraph<double>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<double>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<double>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 HNSWGraph 类（int32_t 版本）
    py::class_<HNSWGraph<int32_t>>(m, "HNSWGraphInt32")
        .def(py::init<int, int, int, int, int, size_t, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("max_elements") = 100000, py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<int32_t>::dim)
        .def_readwrite("m", &HNSWGraph<int32_t>::m)
        .def_readwrite("ef_construction", &HNSWGraph<int32_t>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<int32_t>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<int32_t>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<int32_t>::max_level)
        .def_readwrite("seed", &HNSWGraph<int32_t>::seed)
        .def_readwrite("entry_point_id", &HNSWGraph<int32_t>::entry_point_id)
        .def("get_vector", &HNSWGraph<int32_t>::get_vector, py::arg("vid"))
        .def("get_neighbors", &HNSWGraph<int32_t>::get_neighbors, py::arg("vid"), py::arg("level") = 0)
        .def("get_all_vectors", &HNSWGraph<int32_t>::get_all_vectors)
        .def("get_all_neighbors", &HNSWGraph<int32_t>::get_all_neighbors)
        .def("get_cur_element_count", &HNSWGraph<int32_t>::get_cur_element_count)
        .def("add_vector", &HNSWGraph<int32_t>::add_vector, py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &HNSWGraph<int32_t>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<int32_t>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<int32_t>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 HNSWGraph 类（uint32_t 版本）
    py::class_<HNSWGraph<uint32_t>>(m, "HNSWGraphUInt32")
        .def(py::init<int, int, int, int, int, size_t, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("max_elements") = 100000, py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<uint32_t>::dim)
        .def_readwrite("m", &HNSWGraph<uint32_t>::m)
        .def_readwrite("ef_construction", &HNSWGraph<uint32_t>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<uint32_t>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<uint32_t>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<uint32_t>::max_level)
        .def_readwrite("seed", &HNSWGraph<uint32_t>::seed)
        .def_readwrite("entry_point_id", &HNSWGraph<uint32_t>::entry_point_id)
        .def("get_vector", &HNSWGraph<uint32_t>::get_vector, py::arg("vid"))
        .def("get_neighbors", &HNSWGraph<uint32_t>::get_neighbors, py::arg("vid"), py::arg("level") = 0)
        .def("get_all_vectors", &HNSWGraph<uint32_t>::get_all_vectors)
        .def("get_all_neighbors", &HNSWGraph<uint32_t>::get_all_neighbors)
        .def("get_cur_element_count", &HNSWGraph<uint32_t>::get_cur_element_count)
        .def("add_vector", &HNSWGraph<uint32_t>::add_vector, py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &HNSWGraph<uint32_t>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<uint32_t>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<uint32_t>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 Node 类（float 版本），布式集群中的单个节点，支持启动服务器、查询和添加向量
    py::class_<Node<float>>(m, "NodeFloat")
        .def(py::init<int, Cluster<float> *, HNSWGraph<float> *, bool, int>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false, py::arg("msg_pool_size") = 32)
        .def_readwrite("node_id", &Node<float>::node_id)
        .def_readwrite("num_queries", &Node<float>::num_queries)
        .def_readwrite("is_remote_node", &Node<float>::is_remote_node)
        .def_readwrite("ip_addr", &Node<float>::ip_addr)
        .def_readwrite("port", &Node<float>::port)
        .def_readwrite("local_hnsw", &Node<float>::local_hnsw)
        .def_readwrite("msg_pool_size", &Node<float>::msg_pool_size_)
        .def("start_server", &Node<float>::start_server)
        .def("reset", &Node<float>::reset)
        .def("query", &Node<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<float>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 Node 类（double 版本），接口与 float 版本完全一致
    py::class_<Node<double>>(m, "NodeDouble")
        .def(py::init<int, Cluster<double> *, HNSWGraph<double> *, bool, int>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false, py::arg("msg_pool_size") = 32)
        .def_readwrite("node_id", &Node<double>::node_id)
        .def_readwrite("num_queries", &Node<double>::num_queries)
        .def_readwrite("is_remote_node", &Node<double>::is_remote_node)
        .def_readwrite("ip_addr", &Node<double>::ip_addr)
        .def_readwrite("port", &Node<double>::port)
        .def_readwrite("local_hnsw", &Node<double>::local_hnsw)
        .def_readwrite("msg_pool_size", &Node<double>::msg_pool_size_)
        .def("start_server", &Node<double>::start_server)
        .def("reset", &Node<double>::reset)
        .def("query", &Node<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<double>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 Node 类（int32_t 版本）
    py::class_<Node<int32_t>>(m, "NodeInt32")
        .def(py::init<int, Cluster<int32_t> *, HNSWGraph<int32_t> *, bool, int>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false, py::arg("msg_pool_size") = 32)
        .def_readwrite("node_id", &Node<int32_t>::node_id)
        .def_readwrite("num_queries", &Node<int32_t>::num_queries)
        .def_readwrite("is_remote_node", &Node<int32_t>::is_remote_node)
        .def_readwrite("ip_addr", &Node<int32_t>::ip_addr)
        .def_readwrite("port", &Node<int32_t>::port)
        .def_readwrite("local_hnsw", &Node<int32_t>::local_hnsw)
        .def_readwrite("msg_pool_size", &Node<int32_t>::msg_pool_size_)
        .def("start_server", &Node<int32_t>::start_server)
        .def("reset", &Node<int32_t>::reset)
        .def("query", &Node<int32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<int32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 Node 类（uint32_t 版本）
    py::class_<Node<uint32_t>>(m, "NodeUInt32")
        .def(py::init<int, Cluster<uint32_t> *, HNSWGraph<uint32_t> *, bool, int>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false, py::arg("msg_pool_size") = 32)
        .def_readwrite("node_id", &Node<uint32_t>::node_id)
        .def_readwrite("num_queries", &Node<uint32_t>::num_queries)
        .def_readwrite("is_remote_node", &Node<uint32_t>::is_remote_node)
        .def_readwrite("ip_addr", &Node<uint32_t>::ip_addr)
        .def_readwrite("port", &Node<uint32_t>::port)
        .def_readwrite("local_hnsw", &Node<uint32_t>::local_hnsw)
        .def_readwrite("msg_pool_size", &Node<uint32_t>::msg_pool_size_)
        .def("start_server", &Node<uint32_t>::start_server)
        .def("reset", &Node<uint32_t>::reset)
        .def("query", &Node<uint32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<uint32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 ReduceQuerySession 类（float 版本）
    // 持久连接会话，避免 reduce 模式下每次查询都 std::async + acquire/release 连接
    py::class_<ReduceQuerySession<float>>(m, "ReduceQuerySessionFloat")
        .def(py::init<Cluster<float> *>(), py::arg("cluster"))
        .def("query", &ReduceQuerySession<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &ReduceQuerySession<float>::close);

    // 绑定 ReduceQuerySession 类（double 版本）
    py::class_<ReduceQuerySession<double>>(m, "ReduceQuerySessionDouble")
        .def(py::init<Cluster<double> *>(), py::arg("cluster"))
        .def("query", &ReduceQuerySession<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &ReduceQuerySession<double>::close);

    // 绑定 ReduceQuerySession 类（int32_t 版本）
    py::class_<ReduceQuerySession<int32_t>>(m, "ReduceQuerySessionInt32")
        .def(py::init<Cluster<int32_t> *>(), py::arg("cluster"))
        .def("query", &ReduceQuerySession<int32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &ReduceQuerySession<int32_t>::close);

    // 绑定 ReduceQuerySession 类（uint32_t 版本）
    py::class_<ReduceQuerySession<uint32_t>>(m, "ReduceQuerySessionUInt32")
        .def(py::init<Cluster<uint32_t> *>(), py::arg("cluster"))
        .def("query", &ReduceQuerySession<uint32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("close", &ReduceQuerySession<uint32_t>::close);

    // 绑定 Cluster 类（float 版本），提供集群级别的建图、查询、分区、序列化等操作
    py::class_<Cluster<float>, std::unique_ptr<Cluster<float>>>(m, "ClusterFloat")
        .def(py::init([](HNSWGraph<float> *hnsw, int node_num, const std::string &comm_mode,
                         const std::string &ib_dev, int ib_port, int gid_index) {
                TransportConfig config;
                config.comm_mode = comm_mode;
                config.ib_dev = ib_dev;
                config.ib_port = ib_port;
                config.gid_index = gid_index;
                return new Cluster<float>(hnsw, node_num, config);
             }),
             py::arg("hnsw"), py::arg("node_num") = 4,
             py::arg("comm_mode") = "tcp",
             py::arg("ib_dev") = "",
             py::arg("ib_port") = 1,
             py::arg("gid_index") = -1)
        .def_property_readonly("nodes", [](Cluster<float> &c)
                               {
            std::vector<Node<float> *> ptrs;
            ptrs.reserve(c.nodes.size());
            for (auto &n : c.nodes)
                ptrs.push_back(&n);
            return ptrs; })
        .def_readwrite("node_num", &Cluster<float>::node_num)
        .def_readwrite("bandwidth", &Cluster<float>::bandwidth)
        .def_readwrite("partition_mode", &Cluster<float>::partition_mode)
        .def_readwrite("layer_config", &Cluster<float>::layer_config)
        .def_readwrite("outer_ring_max_iterations", &Cluster<float>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<float>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<float>::redundancy_num)
        .def_readwrite("is_distributed", &Cluster<float>::is_distributed)
        .def_readwrite("is_reduce", &Cluster<float>::is_reduce)
        .def("reset", &Cluster<float>::reset)
        .def("add_vector", &Cluster<float>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &Cluster<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("kmeans", &Cluster<float>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123, py::arg("num_threads") = 0)
        .def("outer_ring", &Cluster<float>::outer_ring)
        .def("partition", &Cluster<float>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{}, py::arg("num_threads") = 0)
        .def("set_node_info", &Cluster<float>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<float>::start_node_server,
             py::arg("node_id"), py::arg("msg_pool_size") = 32)
        .def("join_node_server", &Cluster<float>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<float>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<float>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<float>>(Cluster<float>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 Cluster 类（double 版本），接口与 float 版本完全一致
    py::class_<Cluster<double>, std::unique_ptr<Cluster<double>>>(m, "ClusterDouble")
        .def(py::init([](HNSWGraph<double> *hnsw, int node_num, const std::string &comm_mode,
                         const std::string &ib_dev, int ib_port, int gid_index) {
                TransportConfig config;
                config.comm_mode = comm_mode;
                config.ib_dev = ib_dev;
                config.ib_port = ib_port;
                config.gid_index = gid_index;
                return new Cluster<double>(hnsw, node_num, config);
             }),
             py::arg("hnsw"), py::arg("node_num") = 4,
             py::arg("comm_mode") = "tcp",
             py::arg("ib_dev") = "",
             py::arg("ib_port") = 1,
             py::arg("gid_index") = -1)
        .def_property_readonly("nodes", [](Cluster<double> &c)
                               {
            std::vector<Node<double> *> ptrs;
            ptrs.reserve(c.nodes.size());
            for (auto &n : c.nodes)
                ptrs.push_back(&n);
            return ptrs; })
        .def_readwrite("node_num", &Cluster<double>::node_num)
        .def_readwrite("bandwidth", &Cluster<double>::bandwidth)
        .def_readwrite("partition_mode", &Cluster<double>::partition_mode)
        .def_readwrite("layer_config", &Cluster<double>::layer_config)
        .def_readwrite("outer_ring_max_iterations", &Cluster<double>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<double>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<double>::redundancy_num)
        .def_readwrite("is_distributed", &Cluster<double>::is_distributed)
        .def_readwrite("is_reduce", &Cluster<double>::is_reduce)
        .def("reset", &Cluster<double>::reset)
        .def("add_vector", &Cluster<double>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &Cluster<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("kmeans", &Cluster<double>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123, py::arg("num_threads") = 0)
        .def("outer_ring", &Cluster<double>::outer_ring)
        .def("partition", &Cluster<double>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{}, py::arg("num_threads") = 0)
        .def("set_node_info", &Cluster<double>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<double>::start_node_server,
             py::arg("node_id"), py::arg("msg_pool_size") = 32)
        .def("join_node_server", &Cluster<double>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<double>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<double>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<double>>(Cluster<double>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 Cluster 类（int32_t 版本）
    py::class_<Cluster<int32_t>, std::unique_ptr<Cluster<int32_t>>>(m, "ClusterInt32")
        .def(py::init([](HNSWGraph<int32_t> *hnsw, int node_num, const std::string &comm_mode,
                         const std::string &ib_dev, int ib_port, int gid_index) {
                TransportConfig config;
                config.comm_mode = comm_mode;
                config.ib_dev = ib_dev;
                config.ib_port = ib_port;
                config.gid_index = gid_index;
                return new Cluster<int32_t>(hnsw, node_num, config);
             }),
             py::arg("hnsw"), py::arg("node_num") = 4,
             py::arg("comm_mode") = "tcp",
             py::arg("ib_dev") = "",
             py::arg("ib_port") = 1,
             py::arg("gid_index") = -1)
        .def_property_readonly("nodes", [](Cluster<int32_t> &c)
                               {
            std::vector<Node<int32_t> *> ptrs;
            ptrs.reserve(c.nodes.size());
            for (auto &n : c.nodes)
                ptrs.push_back(&n);
            return ptrs; })
        .def_readwrite("node_num", &Cluster<int32_t>::node_num)
        .def_readwrite("bandwidth", &Cluster<int32_t>::bandwidth)
        .def_readwrite("partition_mode", &Cluster<int32_t>::partition_mode)
        .def_readwrite("layer_config", &Cluster<int32_t>::layer_config)
        .def_readwrite("outer_ring_max_iterations", &Cluster<int32_t>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<int32_t>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<int32_t>::redundancy_num)
        .def_readwrite("is_distributed", &Cluster<int32_t>::is_distributed)
        .def_readwrite("is_reduce", &Cluster<int32_t>::is_reduce)
        .def("reset", &Cluster<int32_t>::reset)
        .def("add_vector", &Cluster<int32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &Cluster<int32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("kmeans", &Cluster<int32_t>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123, py::arg("num_threads") = 0)
        .def("outer_ring", &Cluster<int32_t>::outer_ring)
        .def("partition", &Cluster<int32_t>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{}, py::arg("num_threads") = 0)
        .def("set_node_info", &Cluster<int32_t>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<int32_t>::start_node_server,
             py::arg("node_id"), py::arg("msg_pool_size") = 32)
        .def("join_node_server", &Cluster<int32_t>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<int32_t>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<int32_t>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<int32_t>>(Cluster<int32_t>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 Cluster 类（uint32_t 版本）
    py::class_<Cluster<uint32_t>, std::unique_ptr<Cluster<uint32_t>>>(m, "ClusterUInt32")
        .def(py::init([](HNSWGraph<uint32_t> *hnsw, int node_num, const std::string &comm_mode,
                         const std::string &ib_dev, int ib_port, int gid_index) {
                TransportConfig config;
                config.comm_mode = comm_mode;
                config.ib_dev = ib_dev;
                config.ib_port = ib_port;
                config.gid_index = gid_index;
                return new Cluster<uint32_t>(hnsw, node_num, config);
             }),
             py::arg("hnsw"), py::arg("node_num") = 4,
             py::arg("comm_mode") = "tcp",
             py::arg("ib_dev") = "",
             py::arg("ib_port") = 1,
             py::arg("gid_index") = -1)
        .def_property_readonly("nodes", [](Cluster<uint32_t> &c)
                               {
            std::vector<Node<uint32_t> *> ptrs;
            ptrs.reserve(c.nodes.size());
            for (auto &n : c.nodes)
                ptrs.push_back(&n);
            return ptrs; })
        .def_readwrite("node_num", &Cluster<uint32_t>::node_num)
        .def_readwrite("bandwidth", &Cluster<uint32_t>::bandwidth)
        .def_readwrite("partition_mode", &Cluster<uint32_t>::partition_mode)
        .def_readwrite("layer_config", &Cluster<uint32_t>::layer_config)
        .def_readwrite("outer_ring_max_iterations", &Cluster<uint32_t>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<uint32_t>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<uint32_t>::redundancy_num)
        .def_readwrite("is_distributed", &Cluster<uint32_t>::is_distributed)
        .def_readwrite("is_reduce", &Cluster<uint32_t>::is_reduce)
        .def("reset", &Cluster<uint32_t>::reset)
        .def("add_vector", &Cluster<uint32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", &Cluster<uint32_t>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("kmeans", &Cluster<uint32_t>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123, py::arg("num_threads") = 0)
        .def("outer_ring", &Cluster<uint32_t>::outer_ring)
        .def("partition", &Cluster<uint32_t>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{}, py::arg("num_threads") = 0)
        .def("set_node_info", &Cluster<uint32_t>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<uint32_t>::start_node_server,
             py::arg("node_id"), py::arg("msg_pool_size") = 32)
        .def("join_node_server", &Cluster<uint32_t>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<uint32_t>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<uint32_t>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<uint32_t>>(Cluster<uint32_t>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 ClusterSDKClient 类（float 版本），提供远程集群的客户端操作接口
    py::class_<ClusterSDKClient<float>>(m, "ClusterSDKClientFloat")
        .def(py::init<const std::string &, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000,
             py::arg("comm_mode") = "tcp")
        .def("connect", &ClusterSDKClient<float>::connect,
             py::arg("ip") = "", py::arg("port") = -1)
        .def("init_routes", &ClusterSDKClient<float>::init_routes)
        .def("disconnect", &ClusterSDKClient<float>::disconnect)
        .def("is_connected", &ClusterSDKClient<float>::is_connected)
        .def("set_query_endpoints", &ClusterSDKClient<float>::set_query_endpoints,
             py::arg("ips"), py::arg("ports"))
        .def("add_vector", &ClusterSDKClient<float>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query_send", &ClusterSDKClient<float>::query_send,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("query_recv", &ClusterSDKClient<float>::query_recv,
             py::call_guard<py::gil_scoped_release>())
        .def("query_batch", &ClusterSDKClient<float>::query_batch,
             py::arg("query_ids"), py::arg("query_vectors"), py::arg("payloads"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<float>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
             py::arg("num_threads") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("set_hnsw_params", &ClusterSDKClient<float>::set_hnsw_params,
             py::arg("ef_search") = -1,
             py::arg("ef_construction") = -1,
             py::arg("limited_level") = -2,
             py::call_guard<py::gil_scoped_release>())
        .def("load_cluster", &ClusterSDKClient<float>::load_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_cluster", &ClusterSDKClient<float>::save_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_hnsw", &ClusterSDKClient<float>::load_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_hnsw", &ClusterSDKClient<float>::save_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_information", [](ClusterSDKClient<float> &client) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(client.get_information());
        });

    // 绑定 ClusterSDKClient 类（double 版本），接口与 float 版本完全一致
    py::class_<ClusterSDKClient<double>>(m, "ClusterSDKClientDouble")
        .def(py::init<const std::string &, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000,
             py::arg("comm_mode") = "tcp")
        .def("connect", &ClusterSDKClient<double>::connect,
             py::arg("ip") = "", py::arg("port") = -1)
        .def("init_routes", &ClusterSDKClient<double>::init_routes)
        .def("disconnect", &ClusterSDKClient<double>::disconnect)
        .def("is_connected", &ClusterSDKClient<double>::is_connected)
        .def("set_query_endpoints", &ClusterSDKClient<double>::set_query_endpoints,
             py::arg("ips"), py::arg("ports"))
        .def("add_vector", &ClusterSDKClient<double>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query_send", &ClusterSDKClient<double>::query_send,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("query_recv", &ClusterSDKClient<double>::query_recv,
             py::call_guard<py::gil_scoped_release>())
        .def("query_batch", &ClusterSDKClient<double>::query_batch,
             py::arg("query_ids"), py::arg("query_vectors"), py::arg("payloads"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<double>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
             py::arg("num_threads") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("set_hnsw_params", &ClusterSDKClient<double>::set_hnsw_params,
             py::arg("ef_search") = -1,
             py::arg("ef_construction") = -1,
             py::arg("limited_level") = -2,
             py::call_guard<py::gil_scoped_release>())
        .def("load_cluster", &ClusterSDKClient<double>::load_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_cluster", &ClusterSDKClient<double>::save_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_hnsw", &ClusterSDKClient<double>::load_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_hnsw", &ClusterSDKClient<double>::save_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_information", [](ClusterSDKClient<double> &client) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(client.get_information());
        });

    // 绑定 ClusterSDKClient 类（int32_t 版本）
    py::class_<ClusterSDKClient<int32_t>>(m, "ClusterSDKClientInt32")
        .def(py::init<const std::string &, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000,
             py::arg("comm_mode") = "tcp")
        .def("connect", &ClusterSDKClient<int32_t>::connect,
             py::arg("ip") = "", py::arg("port") = -1)
        .def("init_routes", &ClusterSDKClient<int32_t>::init_routes)
        .def("disconnect", &ClusterSDKClient<int32_t>::disconnect)
        .def("is_connected", &ClusterSDKClient<int32_t>::is_connected)
        .def("set_query_endpoints", &ClusterSDKClient<int32_t>::set_query_endpoints,
             py::arg("ips"), py::arg("ports"))
        .def("add_vector", &ClusterSDKClient<int32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query_send", &ClusterSDKClient<int32_t>::query_send,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("query_recv", &ClusterSDKClient<int32_t>::query_recv,
             py::call_guard<py::gil_scoped_release>())
        .def("query_batch", &ClusterSDKClient<int32_t>::query_batch,
             py::arg("query_ids"), py::arg("query_vectors"), py::arg("payloads"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<int32_t>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
             py::arg("num_threads") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("set_hnsw_params", &ClusterSDKClient<int32_t>::set_hnsw_params,
             py::arg("ef_search") = -1,
             py::arg("ef_construction") = -1,
             py::arg("limited_level") = -2,
             py::call_guard<py::gil_scoped_release>())
        .def("load_cluster", &ClusterSDKClient<int32_t>::load_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_cluster", &ClusterSDKClient<int32_t>::save_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_hnsw", &ClusterSDKClient<int32_t>::load_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_hnsw", &ClusterSDKClient<int32_t>::save_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_information", [](ClusterSDKClient<int32_t> &client) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(client.get_information());
        });

    // 绑定 ClusterSDKClient 类（uint32_t 版本）
    py::class_<ClusterSDKClient<uint32_t>>(m, "ClusterSDKClientUInt32")
        .def(py::init<const std::string &, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000,
             py::arg("comm_mode") = "tcp")
        .def("connect", &ClusterSDKClient<uint32_t>::connect,
             py::arg("ip") = "", py::arg("port") = -1)
        .def("init_routes", &ClusterSDKClient<uint32_t>::init_routes)
        .def("disconnect", &ClusterSDKClient<uint32_t>::disconnect)
        .def("is_connected", &ClusterSDKClient<uint32_t>::is_connected)
        .def("set_query_endpoints", &ClusterSDKClient<uint32_t>::set_query_endpoints,
             py::arg("ips"), py::arg("ports"))
        .def("add_vector", &ClusterSDKClient<uint32_t>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>())
        .def("query_send", &ClusterSDKClient<uint32_t>::query_send,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("query_recv", &ClusterSDKClient<uint32_t>::query_recv,
             py::call_guard<py::gil_scoped_release>())
        .def("query_batch", &ClusterSDKClient<uint32_t>::query_batch,
             py::arg("query_ids"), py::arg("query_vectors"), py::arg("payloads"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<uint32_t>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
             py::arg("num_threads") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("set_hnsw_params", &ClusterSDKClient<uint32_t>::set_hnsw_params,
             py::arg("ef_search") = -1,
             py::arg("ef_construction") = -1,
             py::arg("limited_level") = -2,
             py::call_guard<py::gil_scoped_release>())
        .def("load_cluster", &ClusterSDKClient<uint32_t>::load_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_cluster", &ClusterSDKClient<uint32_t>::save_cluster,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_hnsw", &ClusterSDKClient<uint32_t>::load_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_hnsw", &ClusterSDKClient<uint32_t>::save_hnsw,
             py::arg("filepath"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_information", [](ClusterSDKClient<uint32_t> &client) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(client.get_information());
        });
}