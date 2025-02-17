#!/usr/bin/env python3
import os
import sys
import asyncio
import subprocess
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('domain_automation.log')
        ]
    )
    return logging.getLogger('DomainAutomation')

def check_dependencies():
    required = [
        'aiohttp',
        'aiodns',
        'boto3',
        'cryptography',
        'numpy',
        'pandas',
        'scipy'
    ]
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([
                sys.executable, 
                '-m', 
                'pip', 
                'install', 
                '--user',
                pkg
            ])

async def main():
    logger = setup_logging()
    current_dir = Path(__file__).parent.absolute()
    
    # Ensure all required files exist
    required_files = [
        'secure_domain_manager.py',
        'ssl_certificate_manager.py',
        'domain_dns_manager.py',
        'monitoring_system.py',
        'metrics_processor.py',
        'quantum_cancer_analyzer.py'
    ]
    
    for file in required_files:
        if not (current_dir / file).exists():
            logger.error(f"Missing required file: {file}")
            sys.exit(1)
    
    # Add current directory to path
    sys.path.insert(0, str(current_dir))
    
    # Install dependencies
    check_dependencies()
    
    # Import core modules
    from secure_domain_manager import SecureDomainManager
    from ssl_certificate_manager import SSLCertificateManager
    from domain_dns_manager import DomainDNSManager
    from monitoring_system import DomainHealthMonitor, MonitoringConfig
    from metrics_processor import DomainMetricsProcessor
    
    # Domain configuration
    config = {
        'domain': 'artificialthinker.com',
        'alb_dns': 'kaleidoscope-prod-alb.us-east-2.elb.amazonaws.com',
        'region': 'us-east-2',
        'monitoring': {
            'check_interval': 30,
            'threshold_response_time': 0.5,
            'threshold_availability': 99.9,
            'ssl_expiry_warning': 30,
            'regions_to_check': [
                'us-east-1',
                'us-west-2',
                'eu-west-1',
                'ap-southeast-1',
                'ap-northeast-1'
            ]
        }
    }
    
    try:
        # Initialize managers
        domain_manager = SecureDomainManager(config['domain'])
        
        monitor = DomainHealthMonitor(MonitoringConfig(
            domain=config['domain'],
            check_interval=config['monitoring']['check_interval'],
            threshold_response_time=config['monitoring']['threshold_response_time'],
            threshold_availability=config['monitoring']['threshold_availability'],
            ssl_expiry_warning=config['monitoring']['ssl_expiry_warning'],
            regions_to_check=config['monitoring']['regions_to_check']
        ))
        
        metrics = DomainMetricsProcessor(config['domain'])
        
        # Execute domain setup and monitoring
        await asyncio.gather(
            domain_manager.configure_domain(config['alb_dns']),
            monitor.start_monitoring(),
            metrics.process_metrics(days=30)
        )
        
    except Exception as e:
        logger.error(f"Domain automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)