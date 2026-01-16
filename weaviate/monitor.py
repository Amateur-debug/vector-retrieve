import requests

class WeaviateMonitor:
    def __init__(self, nodes_ip=None):
        if nodes_ip:
            self.nodes_ip = nodes_ip
        else:
            self.nodes_ip = {
                "weaviate_node0": "192.168.200.204",
                "weaviate_node1": "192.168.200.205",
                "weaviate_node2": "192.168.200.206",
                "weaviate_node3": "192.168.200.207",
                "weaviate_node4": "192.168.200.208",
                "weaviate_node5": "192.168.200.209",
            }

    def _fetch_metric(self, metric_prefix):
        """
        从所有节点获取指定前缀的指标。
        每次调用都会触发对所有节点的 HTTP 请求。
        返回: {node_name: aggregated_value}
        """
        metrics = {}
        for name, ip in self.nodes_ip.items():
            try:
                # 假设 Prometheus 指标暴露在 2112 端口
                url = f"http://{ip}:2112/metrics"
                resp = requests.get(url, timeout=0.5)
                if resp.status_code == 200:
                    total_val = 0.0
                    # 简单的流式解析，累加所有匹配前缀的行
                    for line in resp.text.splitlines():
                        if line.startswith(metric_prefix):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    total_val += float(parts[-1])
                                except ValueError:
                                    pass
                    metrics[name] = total_val
                else:
                    metrics[name] = -1.0
            except Exception:
                metrics[name] = -1.0
        return metrics

    # --- Batch Metrics ---

    @property
    def batch_durations_ms(self):
        return self._fetch_metric("batch_durations_ms")

    @property
    def batch_delete_durations_ms(self):
        return self._fetch_metric("batch_delete_durations_ms")

    @property
    def batch_size_bytes(self):
        return self._fetch_metric("batch_size_bytes")

    @property
    def batch_size_objects(self):
        return self._fetch_metric("batch_size_objects")

    @property
    def batch_objects_processed_total(self):
        return self._fetch_metric("batch_objects_processed_total")

    @property
    def batch_objects_processed_bytes(self):
        return self._fetch_metric("batch_objects_processed_bytes")

    # --- Query Metrics ---

    @property
    def concurrent_queries_count(self):
        return self._fetch_metric("concurrent_queries_count")

    @property
    def queries_durations_ms_sum(self):
        return self._fetch_metric("queries_durations_ms_sum")

    @property
    def queries_durations_ms_count(self):
        return self._fetch_metric("queries_durations_ms_count")

    @property
    def queries_filtered_vector_durations_ms(self):
        return self._fetch_metric("queries_filtered_vector_durations_ms")

    @property
    def concurrent_goroutines(self):
        return self._fetch_metric("concurrent_goroutines")

    @property
    def requests_total(self):
        return self._fetch_metric("requests_total")

    @property
    def query_dimensions_total(self):
        return self._fetch_metric("query_dimensions_total")

    @property
    def query_dimensions_combined_total(self):
        return self._fetch_metric("query_dimensions_combined_total")