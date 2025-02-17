#!/usr/bin/env python3
"""
CPU-optimized system monitoring and maintenance utility.
"""

import os
import sys
import time
import psutil
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class SystemMonitor:
    def __init__(self, config_path: str = "config/quantum_config.yml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.stats_history = {
            'cpu': [],
            'memory': [],
            'disk': []
        }
        
        # Track CPU cores separately for better analysis
        self.cpu_cores = psutil.cpu_count()
        self.per_cpu_stats = [[] for _ in range(self.cpu_cores)]

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
        return logging.getLogger('SystemMonitor')

    def collect_system_stats(self) -> Dict:
        """Collect current system statistics."""
        # Get per-CPU usage
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        for i, usage in enumerate(per_cpu):
            self.per_cpu_stats[i].append(usage)
            
        # Get memory details
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'overall': psutil.cpu_percent(interval=1),
                'per_cpu': per_cpu,
                'frequency': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None
            },
            'memory': {
                'used_percent': memory.percent,
                'available': memory.available / (1024 * 1024 * 1024),  # Convert to GB
                'swap_used': swap.percent
            },
            'disk': {
                'usage': psutil.disk_usage('/').percent,
                'io_counters': psutil.disk_io_counters()._asdict() if hasattr(psutil, 'disk_io_counters') else None
            },
            'processes': len(psutil.pids())
        }
        
        # Update history
        self.stats_history['cpu'].append(stats['cpu']['overall'])
        self.stats_history['memory'].append(stats['memory']['used_percent'])
        self.stats_history['disk'].append(stats['disk']['usage'])
        
        return stats

    def check_system_health(self) -> Dict:
        """Check system health against thresholds."""
        stats = self.collect_system_stats()
        health_status = {
            'status': 'healthy',
            'warnings': [],
            'critical': []
        }

        # CPU check
        if stats['cpu']['overall'] > 90:
            health_status['critical'].append('CPU usage critical')
            health_status['status'] = 'critical'
        elif stats['cpu']['overall'] > 80:
            health_status['warnings'].append('CPU usage high')
            health_status['status'] = 'warning'

        # Look for CPU core imbalance
        cpu_usage_std = np.std(stats['cpu']['per_cpu'])
        if cpu_usage_std > 20:  # High standard deviation indicates imbalance
            health_status['warnings'].append('CPU core usage imbalance detected')

        # Memory check
        if stats['memory']['used_percent'] > 90:
            health_status['critical'].append('Memory usage critical')
            health_status['status'] = 'critical'
        elif stats['memory']['used_percent'] > 80:
            health_status['warnings'].append('Memory usage high')
            health_status['status'] = 'warning'

        # Swap check
        if stats['memory']['swap_used'] > 60:
            health_status['warnings'].append('High swap usage')

        # Process count check
        if stats['processes'] > 500:  # Arbitrary threshold, adjust as needed
            health_status['warnings'].append('High process count')

        return health_status

    def plot_system_stats(self, save_path: Optional[str] = None):
        """Plot system statistics over time."""
        plt.figure(figsize=(12, 8))
        timestamps = range(len(self.stats_history['cpu']))

        # CPU and Memory Plot
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, self.stats_history['cpu'], label='CPU Usage (%)', color='blue')
        plt.plot(timestamps, self.stats_history['memory'], label='Memory Usage (%)', color='red')
        plt.title('System Resource Usage Over Time')
        plt.xlabel('Time (samples)')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)

        # Per-CPU Core Plot
        plt.subplot(2, 1, 2)
        for i, core_stats in enumerate(self.per_cpu_stats):
            plt.plot(timestamps[-len(core_stats):], core_stats, 
                    label=f'Core {i}', alpha=0.5)
        plt.title('CPU Core Usage Over Time')
        plt.xlabel('Time (samples)')
        plt.ylabel('Usage (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def monitor_system(self, interval: int = 5, duration: Optional[int] = None):
        """Monitor system continuously."""
        try:
            start_time = time.time()
            while True:
                stats = self.collect_system_stats()
                health = self.check_system_health()
                
                # Log status
                self.logger.info(f"System Stats: {stats}")
                if health['status'] != 'healthy':
                    for warning in health['warnings']:
                        self.logger.warning(warning)
                    for critical in health['critical']:
                        self.logger.error(critical)

                # Generate plots periodically
                if len(self.stats_history['cpu']) % 60 == 0:  # Every 60 samples
                    plot_path = f"logs/system_stats_{int(time.time())}.png"
                    self.plot_system_stats(save_path=plot_path)
                    self.logger.info(f"Generated system stats plot: {plot_path}")

                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                    
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        finally:
            # Save final plot
            plot_path = f"logs/system_stats_final_{int(time.time())}.png"
            self.plot_system_stats(save_path=plot_path)
            self.logger.info(f"Final system stats plot saved: {plot_path}")

    def get_process_info(self) -> List[Dict]:
        """Get information about running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                proc_info['cpu_percent'] = proc.cpu_percent(interval=0.1)
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        return sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="System monitoring utility")
    parser.add_argument('--interval', type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument('--duration', type=int, help="Monitoring duration in seconds")
    parser.add_argument('--top-processes', action='store_true', help="Show top CPU-consuming processes")
    args = parser.parse_args()

    monitor = SystemMonitor()

    if args.top_processes:
        processes = monitor.get_process_info()
        print("\nTop CPU-Consuming Processes:")
        print("-" * 50)
        for proc in processes[:10]:  # Show top 10
            print(f"PID: {proc['pid']:>6} | CPU: {proc['cpu_percent']:>5.1f}% | "
                  f"Memory: {proc['memory_percent']:>5.1f}% | Name: {proc['name']}")
        return

    monitor.monitor_system(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
