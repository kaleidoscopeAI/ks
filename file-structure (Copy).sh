#!/bin/bash
# setup.sh - Creates the complete file system structure

# Create main project directory
mkdir -p quantum_molecular_system/{src,data,logs,results,tests,config,examples}

# Create source code directory structure
mkdir -p quantum_molecular_system/src/{core,utils,interfaces}

# Create data subdirectories
mkdir -p quantum_molecular_system/data/{molecules,temp}

# Create config directory
mkdir -p quantum_molecular_system/config

# Create source files
cat > quantum_molecular_system/src/core/__init__.py << 'EOF'
# Core module initialization
EOF

cat > quantum_molecular_system/src/core/quantum_processor.py << 'EOF'
"""
Core quantum processing functionality.
Import primary QuantumSystem class from here.
"""
from quantum_molecular_system.src.core.system import QuantumSystem

__all__ = ['QuantumSystem']
EOF

# Create configuration file
cat > quantum_molecular_system/config/system_config.yml << 'EOF'
system:
  version: "1.0.0"
  name: "Quantum Molecular System"

paths:
  data: "data/"
  logs: "logs/"
  results: "results/"
  molecules: "data/molecules/"
  temp: "data/temp/"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/system.log"

molecular:
  force_field: "MMFF94"
  optimization_steps: 1000
  convergence_threshold: 1e-6
  energy_unit: "kJ/mol"

output:
  save_format: "json"
  include_timestamps: true
  compress_old_results: true
EOF

# Create example molecule files
cat > quantum_molecular_system/examples/ethanol.smi << 'EOF'
CCO ethanol
EOF

# Create README
cat > quantum_molecular_system/README.md << 'EOF'
# Quantum Molecular System

A system for molecular analysis using Avogadro2 and quantum processing.

## Directory Structure

```
quantum_molecular_system/
├── src/
│   ├── core/           # Core processing logic
│   ├── utils/          # Utility functions
│   └── interfaces/     # External interfaces
├── data/
│   ├── molecules/      # Molecular input files
│   └── temp/          # Temporary files
├── logs/              # System logs
├── results/           # Analysis results
├── tests/             # Test suite
├── config/            # Configuration files
└── examples/          # Example files
```

## Usage

1. Place molecular files in `data/molecules/`
2. Run analysis:
   ```
   python src/core/quantum_processor.py "CCO"
   ```
3. Find results in `results/`
4. Check logs in `logs/`

## Configuration

Edit `config/system_config.yml` to modify system settings.
EOF

# Create requirements file
cat > quantum_molecular_system/requirements.txt << 'EOF'
numpy>=1.21.0
avogadro>=1.97.0
rdkit>=2022.9.1
pyyaml>=6.0.0
EOF

# Set up Python package structure
touch quantum_molecular_system/src/core/__init__.py
touch quantum_molecular_system/src/utils/__init__.py
touch quantum_molecular_system/src/interfaces/__init__.py

# Create empty log file
touch quantum_molecular_system/logs/system.log

# Create empty results directory
touch quantum_molecular_system/results/.gitkeep

# Create example test file
cat > quantum_molecular_system/tests/test_system.py << 'EOF'
import unittest
from quantum_molecular_system.src.core import QuantumSystem

class TestQuantumSystem(unittest.TestCase):
    def setUp(self):
        self.system = QuantumSystem()

    def test_molecule_loading(self):
        """Test molecule loading from SMILES."""
        result = self.system.load_molecule("CCO")
        self.assertTrue(result)
        self.assertIsNotNone(self.system.molecule)

if __name__ == '__main__':
    unittest.main()
EOF

# Make directories readable/writable
chmod -R 755 quantum_molecular_system
chmod -R 777 quantum_molecular_system/{logs,data/temp,results}

echo "Quantum Molecular System file structure created successfully."
