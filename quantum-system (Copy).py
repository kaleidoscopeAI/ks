# src/quantum_processor.py
import os
import numpy as np
import logging
import avogadro
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Optional

class QuantumProcessor:
    """Main quantum processing system with Avogadro2 integration."""
    
    def __init__(self):
        self.avo = avogadro.core()
        self.molecule = None
        self._setup_logging()

    def _setup_logging(self):
        """Initialize logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('QuantumProcessor')

    def load_molecule(self, identifier: str) -> bool:
        """Load molecule from SMILES or file."""
        try:
            if self._is_smiles(identifier):
                mol = Chem.MolFromSmiles(identifier)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                self.molecule = self._convert_rdkit_to_avogadro(mol)
            else:
                self.molecule = self.avo.io.FileFormatManager.readFile(identifier)
            
            self.logger.info(f"Loaded molecule with {self.molecule.atomCount()} atoms")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load molecule: {e}")
            return False

    def analyze_molecule(self) -> Dict:
        """Analyze loaded molecule."""
        if not self.molecule:
            raise ValueError("No molecule loaded")

        try:
            properties = self._calculate_properties()
            optimization = self.optimize_geometry()
            
            analysis = {
                "properties": properties,
                "optimization": optimization,
                "atom_count": self.molecule.atomCount(),
                "bond_count": self.molecule.bondCount()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def optimize_geometry(self, steps: int = 1000) -> Dict:
        """Optimize molecular geometry."""
        if not self.molecule:
            raise ValueError("No molecule loaded")
            
        try:
            ff = self.avo.ForceField()
            ff.setup(self.molecule)
            ff.optimize(steps=steps)
            
            return {
                "energy": ff.energyCalculation(),
                "coordinates": self._get_coordinates(),
                "converged": True
            }
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

    def _calculate_properties(self) -> Dict:
        """Calculate molecular properties."""
        properties = {}
        
        # Calculate mass
        mass = 0.0
        for i in range(self.molecule.atomCount()):
            atom = self.molecule.atom(i)
            mass += atom.mass()
        properties["molecular_mass"] = mass
        
        # Calculate center of mass
        com = self.molecule.centerOfMass()
        properties["center_of_mass"] = (com.x(), com.y(), com.z())
        
        # Calculate bond lengths
        bond_lengths = []
        for i in range(self.molecule.bondCount()):
            bond = self.molecule.bond(i)
            bond_lengths.append(bond.length())
        
        properties["average_bond_length"] = np.mean(bond_lengths)
        properties["max_bond_length"] = np.max(bond_lengths)
        
        return properties

    def _convert_rdkit_to_avogadro(self, rdkit_mol) -> avogadro.core.Molecule:
        """Convert RDKit molecule to Avogadro format."""
        avo_mol = self.avo.core.Molecule()
        
        # Transfer atoms
        conf = rdkit_mol.GetConformer()
        for i in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            avo_atom = avo_mol.addAtom(atom.GetAtomicNum())
            avo_atom.setPosition3d(pos.x, pos.y, pos.z)
            
        # Transfer bonds
        for bond in rdkit_mol.GetBonds():
            avo_mol.addBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondTypeAsDouble()
            )
            
        return avo_mol

    def _get_coordinates(self) -> List[tuple]:
        """Get atomic coordinates."""
        coords = []
        for i in range(self.molecule.atomCount()):
            atom = self.molecule.atom(i)
            pos = atom.position3d()
            coords.append((pos.x(), pos.y(), pos.z()))
        return coords

    @staticmethod
    def _is_smiles(identifier: str) -> bool:
        """Check if string is SMILES."""
        return all(c in 'CNOPSFIBrClHc[]()=#-+' for c in identifier)

# Example usage
if __name__ == "__main__":
    processor = QuantumProcessor()
    
    # Load and analyze ethanol
    if processor.load_molecule("CCO"):
        analysis = processor.analyze_molecule()
        print("Analysis results:", analysis)
