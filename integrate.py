import prometheus_client
from prometheus_client import start_http_server, Counter

class Metrics:
    def __init__(self):
        self.data_processed = Counter('pe_data_processed', 'Total processed data points')
        self.processing_time = prometheus_client.Gauge('pe_processing_seconds', 'Data processing latency')
        start_http_server(9090)  # Metrics server for Prometheus

    def track_processing(self, fn):
        """Decorator for metric collection"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            self.data_processed.inc()
            self.processing_time.set(time.time() - start)
            return result
        return wrapper
