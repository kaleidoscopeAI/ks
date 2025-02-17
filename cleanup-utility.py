#!/usr/bin/env python3
"""
Cleanup utility for the quantum system.
Handles cleanup of temporary files, old logs, and system maintenance.
"""

import os
import sys
import shutil
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set
import argparse

class SystemCleanup:
    def __init__(self, config_path: str = "config/quantum_config.yml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()

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
        return logging.getLogger('SystemCleanup')

    def cleanup_logs(self, max_age_days: int = 7) -> int:
        """Clean up old log files."""
        log_dir = Path(self.config['paths']['logs'])
        count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in log_dir.glob('*.log'):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                log_file.unlink()
                count += 1
                self.logger.info(f"Removed old log file: {log_file}")
        
        return count

    def cleanup_temp_files(self) -> int:
        """Clean up temporary files."""
        temp_dir = Path(self.config['paths']['temp'])
        count = 0
        
        for temp_file in temp_dir.glob('*'):
            temp_file.unlink()
            count += 1
            self.logger.info(f"Removed temporary file: {temp_file}")
        
        return count

    def cleanup_model_checkpoints(self, keep_latest: int = 3) -> int:
        """Clean up old model checkpoints, keeping the N most recent."""
        model_dir = Path(self.config['paths']['models'])
        count = 0
        
        # Get all checkpoint files
        checkpoints = list(model_dir.glob('*.pkl'))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_latest:]:
            checkpoint.unlink()
            count += 1
            self.logger.info(f"Removed old model checkpoint: {checkpoint}")
        
        return count

    def vacuum_database(self) -> bool:
        """Vacuum the system database if exists."""
        try:
            import sqlite3
            db_path = Path(self.config['paths']['data']) / 'system.db'
            
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")
                conn.close()
                self.logger.info("Database vacuumed successfully")
                return True
        except Exception as e:
            self.logger.error(f"Database vacuum failed: {e}")
            return False
        
        return False

    def cleanup_gpu_cache(self) -> bool:
        """Clean up GPU cache if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
                return True
        except Exception as e:
            self.logger.error(f"GPU cache cleanup failed: {e}")
            return False
        
        return False

    def find_orphaned_files(self) -> Set[Path]:
        """Find orphaned files that don't belong to any known process."""
        orphaned = set()
        temp_dir = Path(self.config['paths']['temp'])
        
        # Get list of running processes
        import psutil
        active_pids = {p.pid for p in psutil.process_iter()}
        
        # Check temp files
        for temp_file in temp_dir.glob('*.tmp'):
            try:
                pid = int(temp_file.stem.split('_')[0])
                if pid not in active_pids:
                    orphaned.add(temp_file)
            except (ValueError, IndexError):
                continue
        
        return orphaned

    def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned files."""
        orphaned = self.find_orphaned_files()
        count = 0
        
        for file_path in orphaned:
            try:
                file_path.unlink()
                count += 1
                self.logger.info(f"Removed orphaned file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to remove orphaned file {file_path}: {e}")
        
        return count

    def optimize_storage(self):
        """Optimize storage by compressing old files."""
        import tarfile
        from datetime import datetime
        
        log_dir = Path(self.config['paths']['logs'])
        archive_name = f"logs_archive_{datetime.now().strftime('%Y%m%d')}.tar.gz"
        
        with tarfile.open(log_dir / archive_name, "w:gz") as tar:
            for log_file in log_dir.glob('*.log'):
                if (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days > 30:
                    tar.add(log_file, arcname=log_file.name)
                    log_file.unlink()
                    self.logger.info(f"Archived and removed: {log_file}")

    def run_full_cleanup(self, max_age_days: int = 7, keep_checkpoints: int = 3):
        """Run all cleanup operations."""
        results = {
            'logs_removed': self.cleanup_logs(max_age_days),
            'temp_files_removed': self.cleanup_temp_files(),
            'checkpoints_removed': self.cleanup_model_checkpoints(keep_checkpoints),
            'orphaned_files_removed': self.cleanup_orphaned_files(),
            'database_vacuumed': self.vacuum_database(),
            'gpu_cache_cleared': self.cleanup_gpu_cache()
        }
        
        # Generate report
        report = [
            "Cleanup Report",
            "=" * 50,
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            f"Log files removed: {results['logs_removed']}",
            f"Temporary files removed: {results['temp_files_removed']}",
            f"Model checkpoints removed: {results['checkpoints_removed']}",
            f"Orphaned files removed: {results['orphaned_files_removed']}",
            f"Database vacuumed: {'Yes' if results['database_vacuumed'] else 'No'}",
            f"GPU cache cleared: {'Yes' if results['gpu_cache_cleared'] else 'No'}"
        ]
        
        # Save report
        report_path = Path(self.config['paths']['logs']) / f"cleanup_report_{int(datetime.now().timestamp())}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return results

def main():
    parser = argparse.ArgumentParser(description="System cleanup utility")
    parser.add_argument('--max-age', type=int, default=7, help="Maximum age of files to keep (days)")
    parser.add_argument('--keep-checkpoints', type=int, default=3, help="Number of recent checkpoints to keep")
    parser.add_argument('--optimize-storage', action='store_true', help="Optimize storage by compressing old files")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without doing it")
    args = parser.parse_args()

    cleanup = SystemCleanup()
    
    if args.dry_run:
        print("Dry run - showing what would be cleaned up:")
        orphaned = cleanup.find_orphaned_files()
        print(f"Would remove {len(orphaned)} orphaned files")
        return

    if args.optimize_storage:
        cleanup.optimize_storage()
    
    results = cleanup.run_full_cleanup(
        max_age_days=args.max_age,
        keep_checkpoints=args.keep_checkpoints
    )
    
    print("\nCleanup completed successfully!")
    print(f"Removed {results['logs_removed']} log files")
    print(f"Removed {results['temp_files_removed']} temporary files")
    print(f"Removed {results['checkpoints_removed']} old checkpoints")
    print(f"Removed {results['orphaned_files_removed']} orphaned files")

if __name__ == "__main__":
    main()