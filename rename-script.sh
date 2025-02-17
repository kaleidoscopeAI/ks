#!/bin/bash
set -e

# Core modules
mv quantum-avogadro-enhanced.py modules/quantum_processor.py
mv quantum-system-part2.py modules/quantum_system.py
mv quantum-analysis.py modules/quantum_analyzer.py
mv quantum-system.py modules/quantum_core.py
mv quantum-cpu.py modules/quantum_cpu.py
mv avogadro-integration.py modules/avogadro_interface.py

# Domain automation
mv ssl-certificate-manager.txt modules/ssl_certificate_manager.py
mv cancer_cell_target_simulation.py modules/cancer_simulator.py
mv quantum-drug-discovery-gui.txt modules/quantum_gui.py
mv godaddy-credentials.py modules/domain_credentials.py

# Create directory structure
mkdir -p modules/core
mkdir -p modules/interfaces
mkdir -p modules/automation
mkdir -p configs
mkdir -p logs

# Move files to appropriate directories
mv modules/quantum_*.py modules/core/
mv modules/avogadro_interface.py modules/interfaces/
mv modules/ssl_certificate_manager.py modules/automation/
mv modules/domain_credentials.py modules/automation/

# Create symlinks for main executables
ln -sf modules/automation/ssl_certificate_manager.py ssl_cert_manager
ln -sf modules/core/quantum_processor.py quantum_processor
ln -sf modules/interfaces/avogadro_interface.py avogadro_interface

# Set permissions
chmod +x ssl_cert_manager quantum_processor avogadro_interface
find modules/ -name "*.py" -exec chmod 644 {} \;

# Create __init__.py files
touch modules/__init__.py
touch modules/core/__init__.py
touch modules/interfaces/__init__.py
touch modules/automation/__init__.py

echo "File structure reorganization complete"