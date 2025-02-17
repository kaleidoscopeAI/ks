#!/usr/bin/env python3
"""
Testing and validation script for the quantum system.
"""

import unittest
import numpy as np
import torch
import logging
from pathlib import Path
import yaml
from typing import Dict, List
from enhanced_quantum_system import EnhancedQuantumSystem
from quantum_core import OptimizedQuantumEngine

class QuantumSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load configuration
        with open('config/quantum_config.yml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Initialize system
        cls.system = EnhancedQuantumSystem(
            n_qubits=cls.config['quantum_engine']['n_qubits'],
            n_workers=cls.config['quantum_engine']['n_workers']
        )
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger('QuantumSystemTests')

    def test_molecule_loading(self):
        """Test molecule loading functionality."""
        # Test SMILES loading
        self.assertTrue(self.system.load_molecule("CCO"))
        self.assertIsNotNone(self.system.molecule)
        self.assertEqual(self.system.molecule.atomCount(), 9)  # Ethanol with hydrogens
        
        # Test invalid SMILES
        self.assertFalse(self.system.load_molecule("invalid_smiles"))

    def test_quantum_state(self):
        """Test quantum state initialization and evolution."""
        self.system.load_molecule("CCO")
        
        # Test state initialization
        self.assertIsNotNone(self.system.quantum_state)
        self.assertEqual(len(self.system.quantum_state.wavefunction), 2**self.system.n_qubits)
        
        # Test normalization
        wf_norm = np.linalg.norm(self.system.quantum_state.wavefunction)
        self.assertAlmostEqual(wf_norm, 1.0, places=6)
        
        # Test coherence
        self.assertGreaterEqual(self.system.quantum_state.coherence, 0.0)
        self.assertLessEqual(self.system.quantum_state.coherence, 1.0)

    def test_molecular_analysis(self):
        """Test molecular analysis capabilities."""
        self.system.load_molecule("CCO")
        analysis = self.system.analyze_molecule()
        
        # Check analysis structure
        required_keys = [
            "basic_properties",
            "quantum_analysis",
            "graph_analysis",
            "neural_predictions",
            "electronic_structure",
            "molecular_descriptors"
        ]
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Test specific properties
        props = analysis["basic_properties"]
        self.assertGreater(props["molecular_mass"], 0)
        self.assertEqual(props["atom_count"], 9)  # Ethanol with hydrogens

    def test_geometry_optimization(self):
        """Test geometry optimization."""
        self.system.load_molecule("CCO")
        result = self.system.optimize_geometry(max_steps=100)
        
        # Check optimization results
        self.assertIn("final_energy", result)
        self.assertIn("coordinates", result)
        self.assertIn("convergence_steps", result)
        self.assertLessEqual(result["convergence_steps"], 100)
        
        # Check energy decrease
        energy_history = result["energy_history"]
        self.assertLess(energy_history[-1], energy_history[0])

    def test_graph_analysis(self):
        """Test graph analysis features."""
        self.system.load_molecule("CCO")
        analysis = self.system.analyze_molecule()
        graph_analysis = analysis["graph_analysis"]
        
        # Check graph metrics
        basic_metrics = graph_analysis["basic_metrics"]
        self.assertEqual(basic_metrics["n_nodes"], 9)  # Ethanol with hydrogens
        self.assertGreater(basic_metrics["n_edges"], 0)
        self.assertGreater(basic_metrics["avg_degree"], 0)
        
        # Check spectral metrics
        spectral_metrics = graph_analysis["spectral_metrics"]
        self.assertGreater(spectral_metrics["spectral_radius"], 0)
        self.assertGreaterEqual(spectral_metrics["spectral_gap"], 0)

    def test_neural_predictions(self):
        """Test neural network predictions."""
        self.system.load_molecule("CCO")
        analysis = self.system.analyze_molecule()
        predictions = analysis["neural_predictions"]
        
        # Check prediction structure
        self.assertIn("quantum_state_prediction", predictions)
        self.assertIn("prediction_norm", predictions)
        self.assertIn("prediction_entropy", predictions)
        
        # Test prediction values
        pred_norm = predictions["prediction_norm"]
        self.assertGreater(pred_norm, 0)
        self.assertLess(pred_norm, 10)  # Reasonable range

    def test_electronic_structure(self):
        """Test electronic structure analysis."""
        self.system.load_molecule("CCO")
        analysis = self.system.analyze_molecule()
        electronic = analysis["electronic_structure"]
        
        # Check electronic configuration
        config = electronic["electronic_configuration"]
        self.assertGreater(len(config), 0)
        self.assertEqual(sum(config.values()), electronic["total_electrons"])
        
        # Check molecular orbitals
        orbitals = electronic["molecular_orbitals"]
        self.assertEqual(len(orbitals["energy_levels"]), orbitals["n_orbitals"])
        self.assertLess(orbitals["homo_energy"], orbitals["lumo_energy"])

    def test_cuda_support(self):
        """Test CUDA functionality if available."""
        if torch.cuda.is_available():
            self.assertTrue(self.system.cuda_enabled)
            self.assertIsNotNone(self.system.evolution_kernel)
        else:
            self.assertFalse(self.system.cuda_enabled)

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test invalid molecule
        with self.assertRaises(ValueError):
            self.system.analyze_molecule()  # No molecule loaded
        
        # Test invalid optimization
        with self.assertRaises(ValueError):
            self.system.optimize_geometry(max_steps=-1)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.system.shutdown()

def run_tests():
    """Run all tests and generate report."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(QuantumSystemTests)
    
    # Set up test result collection
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    report_path = Path('logs/test_report.txt')
    with open(report_path, 'w') as f:
        f.write("Quantum System Test Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tests Run: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write("\nDetailed Results:\n")
        f.write("-" * 50 + "\n")
        
        if result.failures:
            f.write("\nFailures:\n")
            for test, traceback in result.failures:
                f.write(f"\n{test}\n")
                f.write(f"{traceback}\n")
                
        if result.errors:
            f.write("\nErrors:\n")
            for test, traceback in result.errors:
                f.write(f"\n{test}\n")
                f.write(f"{traceback}\n")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)Tests Run: {result.testsRun}\n")
        f.write(f"