#!/usr/bin/env python3
import os
import sys
import psutil
import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple

class SystemLauncher:
    def __init__(self):
        self.base_dir = Path("/opt/kaleidoscope")
        self.config_dir = Path("/etc/kaleidoscope")
        self.log_dir = Path("/var/log/kaleidoscope")
        self.setup_logging()
        self.load_config()
        self.check_system_requirements()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_config(self) -> Dict:
        config_files = {
            'neural': self.config_dir / 'neural/config.yml',
            'quantum': self.config_dir / 'quantum/config.yml',
            'pipeline': self.config_dir / 'pipeline/config.yml'
        }
        
        self.config = {}
        for key, path in config_files.items():
            with open(path) as f:
                self.config[key] = yaml.safe_load(f)

    def check_system_requirements(self):
        requirements = {
            'cpu_cores': 4,
            'ram_gb': 16,
            'gpu_memory_gb': 8
        }

        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < requirements['cpu_cores']:
            raise SystemError(f"Insufficient CPU cores: {cpu_count}/{requirements['cpu_cores']}")

        # Check RAM
        total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        if total_ram < requirements['ram_gb']:
            raise SystemError(f"Insufficient RAM: {total_ram:.1f}GB/{requirements['ram_gb']}GB")

        # Check GPU
        if not torch.cuda.is_available():
            raise SystemError("No CUDA-capable GPU found")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        if gpu_memory < requirements['gpu_memory_gb']:
            raise SystemError(f"Insufficient GPU memory: {gpu_memory:.1f}GB/{requirements['gpu_memory_gb']}GB")

    def start_components(self):
        from quantum_core.quantum_processor import QuantumProcessor
        from graph_processor.spectral_optimizer import SpectralOptimizer
        from neural_engine.quantum_neural_net import QuantumNeuralNetwork

        components = {
            'quantum_processor': QuantumProcessor(
                n_qubits=self.config['quantum']['n_qubits'],
                depth=self.config['quantum']['depth']
            ),
            'graph_optimizer': SpectralOptimizer(
                n_clusters=self.config['pipeline']['n_clusters']
            ),
            'neural_network': QuantumNeuralNetwork(
                n_qubits=self.config['quantum']['n_qubits'],
                n_layers=self.config['neural']['n_layers']
            ).cuda()
        }

        # Initialize CUDA kernels
        self.initialize_cuda_kernels()

        return components

    def initialize_cuda_kernels(self):
        cuda_lib_path = self.base_dir / "cuda_kernels/libquantum_evolution.so"
        if not cuda_lib_path.exists():
            raise FileNotFoundError(f"CUDA kernel library not found at {cuda_lib_path}")
        
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # Set CUDA device properties
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def monitor_system_health(self):
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                # RAM usage
                memory = psutil.virtual_memory()
                # GPU usage
                gpu_stats = torch.cuda.memory_stats()
                
                metrics = {
                    'cpu_usage': cpu_percent,
                    'ram_usage': memory.percent,
                    'gpu_allocated': gpu_stats['allocated_bytes.all.current'] / (1024**3),
                    'gpu_reserved': gpu_stats['reserved_bytes.all.current'] / (1024**3)
                }
                
                logging.info(f"System Metrics: {metrics}")
                
                # Check thresholds
                if any([
                    cpu_percent > 90,
                    memory.percent > 90,
                    gpu_stats['allocated_bytes.all.current'] / gpu_stats['reserved_bytes.all.current'] > 0.9
                ]):
                    logging.warning("System resources critical")
                    
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
            
            time.sleep(5)

def main():
    launcher = SystemLauncher()
    
    try:
        components = launcher.start_components()
        logging.info("System components initialized successfully")
        
        # Start monitoring in separate thread
        import threading
        monitor_thread = threading.Thread(target=launcher.monitor_system_health)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Keep main thread alive
        monitor_thread.join()
        
    except Exception as e:
        logging.error(f"System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
