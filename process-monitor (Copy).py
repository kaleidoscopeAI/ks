#!/usr/bin/env python3
"""
Process monitoring and optimization for CPU-based quantum processing.
"""

import os
import sys
import time
import psutil
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import signal
from multiprocessing import shared_memory
import numpy as np

class ProcessOptimizer:
    def __init__(self, config_path: str = "config/quantum_config.yml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.process_control = self._load_process_control()
        self.monitored_pids = set()
        self.shared_mem = None
        self._setup_signal_handlers()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('ProcessOptimizer')

    def _load_process_control(self) -> Dict:
        """Load process control data."""
        control_file = Path(self.config['paths']['temp']) / 'process_control.json'
        try:
            with open(control_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load process control file: {e}")
            return {}

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self.cleanup()
        sys.exit(0)

    def setup_shared_memory(self):
        """Set up shared memory for inter-process communication."""
        try:
            # Create shared memory for process stats
            mem_size = 1024  # Size in bytes
            self.shared_mem = shared_memory.SharedMemory(
                name='quantum_process_stats',
                create=True,
                size=mem_size
            )
            self.logger.info("Shared memory initialized")
        except Exception as e:
            self.logger.error(f"Failed to set up shared memory: {e}")

    def get_process_info(self, pid: int) -> Dict:
        """Get detailed information about a process."""
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                cpu_times = process.cpu_times()
                memory_info = process.memory_info()
                
                return {
                    'pid': pid,
                    'name': process.name(),
                    'status': process.status(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'user_time': cpu_times.user,
                    'system_time': cpu_times.system,
                    'rss': memory_info.rss / (1024 * 1024),  # Convert to MB
                    'vms': memory_info.vms / (1024 * 1024),  # Convert to MB
                    'num_threads': process.num_threads(),
                    'io_counters': process.io_counters()._asdict() if hasattr(process, 'io_counters') else None,
                    'nice': process.nice()
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}

    def optimize_process(self, pid: int):
        """Optimize a process based on its resource usage."""
        process_info = self.get_process_info(pid)
        if not process_info:
            return

        try:
            process = psutil.Process(pid)
            
            # CPU optimization
            if process_info['cpu_percent'] > 80:
                # Reduce priority if using too much CPU
                current_nice = process.nice()
                if current_nice < 10:
                    process.nice(current_nice + 1)
                    self.logger.info(f"Reduced priority for process {pid}")

            # Memory optimization
            if process_info['memory_percent'] > 80:
                # Try to free memory
                if hasattr(process, 'memory_maps'):
                    process.memory_maps()
                self.logger.warning(f"High memory usage for process {pid}")

            # Thread optimization
            if process_info['num_threads'] > 20:  # Arbitrary threshold
                self.logger.warning(f"High thread count for process {pid}")

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def analyze_cpu_affinity(self, pid: int):
        """Analyze and optimize CPU affinity for a process."""
        try:
            process = psutil.Process(pid)
            current_affinity = process.cpu_affinity()
            cpu_percent = psutil.cpu_percent(percpu=True)
            
            # Find least loaded CPUs
            cpu_loads = list(enumerate(cpu_percent))
            cpu_loads.sort(key=lambda x: x[1])  # Sort by CPU load
            optimal_cpus = [cpu[0] for cpu in cpu_loads[:len(current_affinity)]]
            
            if set(optimal_cpus) != set(current_affinity):
                process.cpu_affinity(optimal_cpus)
                self.logger.info(f"Optimized CPU affinity for process {pid}")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def monitor_memory_usage(self):
        """Monitor and manage system memory usage."""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.logger.warning("Critical memory usage detected")
            
            # Find processes using most memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuch