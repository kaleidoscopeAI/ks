#!/usr/bin/env python3
"""
Deployment script for the quantum system.
"""

import os
import sys
import shutil
import subprocess
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class SystemDeployer:
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
        return logging.getLogger('SystemDeployer')

    def check_environment(self) -> bool:
        """Verify deployment environment."""
        required_vars = [
            'QUANTUM_HOME',
            'PYTHONPATH',
            'CUDA_VISIBLE_DEVICES'
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in os.environ:
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return False
            
        return True

    def backup_existing(self):
        """Backup existing installation."""
        paths = self.config['paths']
        backup_dir = Path('backups') / f"backup_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for path_name, path in paths.items():
            src_path = Path(path)
            if src_path.exists():
                dst_path = backup_dir / path_name
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                self.logger.info(f"Backed up {path_name} to {dst_path}")

    def deploy_files(self):
        """Deploy system files to appropriate locations."""
        install_path = Path(os.environ['QUANTUM_HOME'])
        
        # Copy core files
        shutil.copytree('src', install_path / 'src', dirs_exist_ok=True)
        shutil.copytree('config', install_path / 'config', dirs_exist_ok=True)
        
        # Create required directories
        for path in self.config['paths'].values():
            (install_path / path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Deployed files to {install_path}")

    def compile_cuda_kernels(self) -> bool:
        """Compile CUDA kernels if CUDA is enabled."""
        if not self.config['cuda']['enabled']:
            self.logger.info("CUDA disabled, skipping kernel compilation")
            return True
            
        try:
            kernel_dir = Path('cuda_kernels')
            for kernel_file in kernel_dir.glob('*.cu'):
                output_file = kernel_file.with_suffix('.so')
                cmd = [
                    'nvcc',
                    '-shared',
                    '-Xcompiler',
                    '-fPIC',
                    '-o',
                    str(output_file),
                    str(kernel_file)
                ]
                subprocess.run(cmd, check=True)
                self.logger.info(f"Compiled CUDA kernel: {kernel_file}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"CUDA compilation failed: {e}")
            return False

    def run_tests(self) -> bool:
        """Run system tests."""
        try:
            result = subprocess.run(
                [sys.executable, 'tests/test_quantum_system.py'],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("All tests passed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Tests failed:\n{e.output}")
            return False

    def set_permissions(self):
        """Set appropriate permissions for deployed files."""
        install_path = Path(os.environ['QUANTUM_HOME'])
        
        # Set executable permissions for scripts
        for script in (install_path / 'scripts').glob('*.py'):
            script.chmod(0o755)
        
        # Set directory permissions
        for path in self.config['paths'].values():
            dir_path = install_path / path
            dir_path.chmod(0o775)

    def create_activation_script(self):
        """Create activation script for environment setup."""
        activate_script = Path('scripts/activate.sh')
        with open(activate_script, 'w') as f:
            f.write("""#!/bin/bash
export QUANTUM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="$QUANTUM_HOME/src:$PYTHONPATH"
export PATH="$QUANTUM_HOME/scripts:$PATH"
export CUDA_VISIBLE_DEVICES=0

echo "Quantum environment activated."
""")
        activate_script.chmod(0o755)
        self.logger.info("Created activation script")

    def deploy(self, run_tests: bool = True, backup: bool = True) -> bool:
        """Run full deployment process."""
        steps = [
            (self.check_environment, "Checking environment"),
            (self.compile_cuda_kernels, "Compiling CUDA kernels"),
            (lambda: not run_tests or self.run_tests(), "Running tests"),
            (lambda: not backup or self.backup_existing(), "Creating backup"),
            (self.deploy_files, "Deploying files"),
            (self.set_permissions, "Setting permissions"),
            (self.create_activation_script, "Creating activation script")
        ]

        for step_func, step_name in steps:
            self.logger.info(f"Starting: {step_name}")
            if not step_func():
                self.logger.error(f"Deployment failed at: {step_name}")
                return False
            self.logger.info(f"Completed: {step_name}")

        self.logger.info("Deployment completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Deploy quantum system")
    parser.add_argument('--no-tests', action='store_true', help="Skip running tests")
    parser.add_argument('--no-backup', action='store_true', help="Skip backup")
    args = parser.parse_args()

    deployer = SystemDeployer()
    success = deployer.deploy(
        run_tests=not args.no_tests,
        backup=not args.no_backup
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
