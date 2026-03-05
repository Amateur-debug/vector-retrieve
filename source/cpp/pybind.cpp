#include <pybind11/numpy.h> // 支持 numpy 数组
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 支持 STL 类型绑定

#include "cluster.h"
#include "hnsw.h"
#include "payload.h"
#include "sdk.h"
#include "transport.h"

namespace py = pybind11;

PYBIND11_MODULE(vector_retrieve, m)
{
    m.doc() = "Pybind11 bindings for Cluster and related classes";

    // 绑定 TransportConfig 配置类
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
        .def_readwrite("remote_transfer_latency", &Payload<float>::remote_transfer_latency)
        .def_readwrite("send_timestamp", &Payload<float>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<float>::distance_cnt)
        .def_readwrite("results", &Payload<float>::results)
        .def("topk_ids", &Payload<float>::topk_ids, py::arg("k"));

    // 绑定 Payload 类（double 版本）
    py::class_<Payload<double>>(m, "PayloadDouble")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<double>::size_bits)
        .def_readwrite("require_id", &Payload<double>::require_id)
        .def_readwrite("continue_search", &Payload<double>::continue_search)
        .def_readwrite("origin_node_id", &Payload<double>::origin_node_id)
        .def_readwrite("remote_transfer_cnt", &Payload<double>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_latency", &Payload<double>::remote_transfer_latency)
        .def_readwrite("send_timestamp", &Payload<double>::send_timestamp)
        .def_readwrite("distance_cnt", &Payload<double>::distance_cnt)
        .def_readwrite("results", &Payload<double>::results)
        .def("topk_ids", &Payload<double>::topk_ids, py::arg("k"));

    // 绑定 HNSWGraph 类（float 版本）
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
             py::call_guard<py::gil_scoped_release>())  // 释放GIL，允许真正的多线程并行
        .def("query", &HNSWGraph<float>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<float>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<float>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 HNSWGraph 类（double 版本）
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
             py::call_guard<py::gil_scoped_release>())  // 释放GIL，允许真正的多线程并行
        .def("query", &HNSWGraph<double>::query, py::call_guard<py::gil_scoped_release>())
        .def("save", &HNSWGraph<double>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<double>::load, py::arg("filepath"), py::return_value_policy::take_ownership);

    // 绑定 Node 类（float 版本）
    py::class_<Node<float>>(m, "NodeFloat")
        .def(py::init<int, Cluster<float> *, HNSWGraph<float> *, bool>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false)
        .def_readwrite("node_id", &Node<float>::node_id)
        .def_readwrite("num_queries", &Node<float>::num_queries)
        .def_readwrite("is_remote_node", &Node<float>::is_remote_node)
        .def_readwrite("ip_addr", &Node<float>::ip_addr)
        .def_readwrite("port", &Node<float>::port)
        .def_readwrite("local_hnsw", &Node<float>::local_hnsw)
        .def("start_server", &Node<float>::start_server)
        .def("reset", &Node<float>::reset)
        .def("query", &Node<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<float>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 Node 类（double 版本）
    py::class_<Node<double>>(m, "NodeDouble")
        .def(py::init<int, Cluster<double> *, HNSWGraph<double> *, bool>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"), py::arg("is_remote_node") = false)
        .def_readwrite("node_id", &Node<double>::node_id)
        .def_readwrite("num_queries", &Node<double>::num_queries)
        .def_readwrite("is_remote_node", &Node<double>::is_remote_node)
        .def_readwrite("ip_addr", &Node<double>::ip_addr)
        .def_readwrite("port", &Node<double>::port)
        .def_readwrite("local_hnsw", &Node<double>::local_hnsw)
        .def("start_server", &Node<double>::start_server)
        .def("reset", &Node<double>::reset)
        .def("query", &Node<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("add_vector", &Node<double>::add_vector,
             py::arg("vid"), py::arg("vec"),
             py::call_guard<py::gil_scoped_release>());

    // 绑定 Cluster 类（float 版本）
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
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123)
        .def("outer_ring", &Cluster<float>::outer_ring)
        .def("partition", &Cluster<float>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{})
        .def("set_node_info", &Cluster<float>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<float>::start_node_server,
             py::arg("node_id"))
        .def("join_node_server", &Cluster<float>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<float>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<float>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<float>>(Cluster<float>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 Cluster 类（double 版本）
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
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123)
        .def("outer_ring", &Cluster<double>::outer_ring)
        .def("partition", &Cluster<double>::partition,
             py::arg("mode"), py::arg("redundancy") = 1, py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{}, py::arg("layer_partition_mode") = "round_robin", py::arg("layer_replicate_flags") = std::vector<bool>{})
        .def("set_node_info", &Cluster<double>::set_node_info,
             py::arg("node_id"), py::arg("ip"), py::arg("port"), py::arg("is_remote"))
        .def("start_node_server", &Cluster<double>::start_node_server,
             py::arg("node_id"))
        .def("join_node_server", &Cluster<double>::join_node_server,
             py::arg("node_id"))
        .def("transfer_to_remote", &Cluster<double>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"))
        .def("save", &Cluster<double>::save, py::arg("filepath"))
        .def_static("load", [](const std::string &filepath) {
            return std::unique_ptr<Cluster<double>>(Cluster<double>::load(filepath));
        }, py::arg("filepath"));

    // 绑定 ClusterSDKClient 类（float 版本）
    py::class_<ClusterSDKClient<float>>(m, "ClusterSDKClientFloat")
        .def(py::init<const std::string &, int, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000, py::arg("timeout") = 5,
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
        .def("query", &ClusterSDKClient<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<float>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
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

    // 绑定 ClusterSDKClient 类（double 版本）
    py::class_<ClusterSDKClient<double>>(m, "ClusterSDKClientDouble")
        .def(py::init<const std::string &, int, int, const std::string &>(),
             py::arg("ip") = "127.0.0.1", py::arg("port") = 19000, py::arg("timeout") = 5,
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
        .def("query", &ClusterSDKClient<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10,
             py::call_guard<py::gil_scoped_release>())
        .def("partition", &ClusterSDKClient<double>::partition,
             py::arg("mode"), py::arg("redundancy") = 1,
             py::arg("layer_config") = std::unordered_map<int, std::vector<int>>{},
             py::arg("layer_partition_mode") = "round_robin",
             py::arg("layer_replicate_flags") = std::vector<bool>{},
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
}