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
