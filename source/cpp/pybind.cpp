#include <pybind11/numpy.h> // 支持 numpy 数组
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 支持 STL 类型绑定

#include "cluster.h"
#include "hnsw.h"
#include "payload.h"

namespace py = pybind11;

PYBIND11_MODULE(vector_retrieve, m)
{
    m.doc() = "Pybind11 bindings for Cluster and related classes";

    py::class_<Payload<float>>(m, "PayloadFloat")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<float>::size_bits)
        .def_readwrite("require_id", &Payload<float>::require_id)
        .def_readwrite("continue_search", &Payload<float>::continue_search)
        .def_readwrite("results", &Payload<float>::results)
        .def("topk_ids", &Payload<float>::topk_ids, py::arg("k"));

    // 绑定 Payload 类（double 版本）
    py::class_<Payload<double>>(m, "PayloadDouble")
        .def(py::init<>())
        .def_property_readonly("size_bits", &Payload<double>::size_bits)
        .def_readwrite("require_id", &Payload<double>::require_id)
        .def_readwrite("continue_search", &Payload<double>::continue_search)
        .def_readwrite("results", &Payload<double>::results)
        .def("topk_ids", &Payload<double>::topk_ids, py::arg("k"));

    // 绑定 HNSWGraph 类（float 版本）
    py::class_<HNSWGraph<float>>(m, "HNSWGraphFloat")
        .def(py::init<int, int, int, int, int, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<float>::dim)
        .def_readwrite("m", &HNSWGraph<float>::m)
        .def_readwrite("ef_construction", &HNSWGraph<float>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<float>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<float>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<float>::max_level)
        .def_readwrite("seed", &HNSWGraph<float>::seed)
        .def_readwrite("vectors", &HNSWGraph<float>::vectors)
        .def_readwrite("neighbors", &HNSWGraph<float>::neighbors)
        .def_readwrite("entry_point_id", &HNSWGraph<float>::entry_point_id)
        .def("add_vector", &HNSWGraph<float>::add_vector, py::arg("vid"), py::arg("vec"))
        .def("query", &HNSWGraph<float>::query)
        .def("save", &HNSWGraph<float>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<float>::load, py::arg("filepath"));

    // 绑定 HNSWGraph 类（double 版本）
    py::class_<HNSWGraph<double>>(m, "HNSWGraphDouble")
        .def(py::init<int, int, int, int, int, unsigned int>(),
             py::arg("dim"), py::arg("m"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("limited_level"), py::arg("seed") = 123)
        .def_readwrite("dim", &HNSWGraph<double>::dim)
        .def_readwrite("m", &HNSWGraph<double>::m)
        .def_readwrite("ef_construction", &HNSWGraph<double>::ef_construction)
        .def_readwrite("ef_search", &HNSWGraph<double>::ef_search)
        .def_readwrite("limited_level", &HNSWGraph<double>::limited_level)
        .def_readwrite("max_level", &HNSWGraph<double>::max_level)
        .def_readwrite("seed", &HNSWGraph<double>::seed)
        .def_readwrite("vectors", &HNSWGraph<double>::vectors)
        .def_readwrite("neighbors", &HNSWGraph<double>::neighbors)
        .def_readwrite("entry_point_id", &HNSWGraph<double>::entry_point_id)
        .def("add_vector", &HNSWGraph<double>::add_vector, py::arg("vid"), py::arg("vec"))
        .def("query", &HNSWGraph<double>::query)
        .def("save", &HNSWGraph<double>::save, py::arg("filepath"))
        .def_static("load", &HNSWGraph<double>::load, py::arg("filepath"));

    // 绑定 Node 类（float 版本）
    py::class_<Node<float>>(m, "NodeFloat")
        .def(py::init<int, Cluster<float> *, HNSWGraph<float> *>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"))
        .def_readwrite("node_id", &Node<float>::node_id)
        .def_readwrite("num_queries", &Node<float>::num_queries)
        .def("reset", &Node<float>::reset)
        .def("query", &Node<float>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10);

    // 绑定 Node 类（double 版本）
    py::class_<Node<double>>(m, "NodeDouble")
        .def(py::init<int, Cluster<double> *, HNSWGraph<double> *>(),
             py::arg("node_id"), py::arg("cluster"), py::arg("local_hnsw"))
        .def_readwrite("node_id", &Node<double>::node_id)
        .def_readwrite("num_queries", &Node<double>::num_queries)
        .def("reset", &Node<double>::reset)
        .def("query", &Node<double>::query,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k") = 10);

    // 绑定 Cluster 类（float 版本）
    py::class_<Cluster<float>>(m, "ClusterFloat")
        .def(py::init<HNSWGraph<float> *, int>(),
             py::arg("hnsw"), py::arg("node_num") = 4)
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
        .def_readwrite("outer_ring_max_iterations", &Cluster<float>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<float>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<float>::redundancy_num)
        .def_readwrite("remote_transfer_cnt", &Cluster<float>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_latency", &Cluster<float>::remote_transfer_latency)
        .def("reset", &Cluster<float>::reset)
        .def_static("kmeans", &Cluster<float>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123)
        .def("outer_ring", &Cluster<float>::outer_ring)
        .def("partition", &Cluster<float>::partition,
             py::arg("mode"), py::arg("redundancy") = 1)
        .def("transfer_to_remote", &Cluster<float>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"));

    // 绑定 Cluster 类（double 版本）
    py::class_<Cluster<double>>(m, "ClusterDouble")
        .def(py::init<HNSWGraph<double> *, int>(),
             py::arg("hnsw"), py::arg("node_num") = 4)
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
        .def_readwrite("outer_ring_max_iterations", &Cluster<double>::outer_ring_max_iterations)
        .def_readwrite("redundancy_level", &Cluster<double>::redundancy_level)
        .def_readwrite("redundancy_num", &Cluster<double>::redundancy_num)
        .def_readwrite("remote_transfer_cnt", &Cluster<double>::remote_transfer_cnt)
        .def_readwrite("remote_transfer_latency", &Cluster<double>::remote_transfer_latency)
        .def("reset", &Cluster<double>::reset)
        .def_static("kmeans", &Cluster<double>::kmeans,
                    py::arg("data"), py::arg("k"), py::arg("max_iters") = 100, py::arg("seed") = 123)
        .def("outer_ring", &Cluster<double>::outer_ring)
        .def("partition", &Cluster<double>::partition,
             py::arg("mode"), py::arg("redundancy") = 1)
        .def("transfer_to_remote", &Cluster<double>::transfer_to_remote,
             py::arg("query_id"), py::arg("query_vector"), py::arg("payload"), py::arg("k"));
}