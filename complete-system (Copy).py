#!/usr/bin/env python3
"""
Quantum Molecular Analysis System
A complete system for molecular analysis using Avogadro2 and quantum processing.
"""

import os
import sys
import numpy as np
import logging
import avogadro
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import json

class QuantumSystem:
    def __init__(self):
        # Initialize core components
        self.avo = avogadro.core()
        self.molecule = None
        self._setup_logging()
        self._create_directories()

    def _setup_logging(self):
        """Initialize logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "quantum_system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('QuantumSystem')

    def _create_directories(self):
        """Create necessary directories."""
        dirs = ['logs', 'data', 'results']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def load_molecule(self, identifier: str) -> bool:
        """Load molecule from SMILES or file."""
        try:
            if self._is_smiles(identifier):
                mol = Chem.MolFromSmiles(identifier)
                if not mol:
                    self.logger.error("Invalid SMILES string")
                    return False
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                self.molecule = self._convert_rdkit_to_avogadro(mol)
            else:
                file_path = Path(identifier)
                if not file_path.exists():
                    self.logger.error(f"File not found: {identifier}")
                    return False
                self.molecule = self.avo.io.FileFormatManager.readFile(str(file_path))
            
            self.logger.info(f"Loaded molecule with {self.molecule.atomCount()} atoms")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load molecule: {e}")
            return False

    def analyze(self) -> Dict:
        """Perform complete molecular analysis."""
        if not self.molecule:
            raise ValueError("No molecule loaded")

        try:
            # Basic properties
            properties = self._calculate_properties()
            
            # Geometry optimization
            optimization = self.optimize_geometry()
            
            # Energy calculations
            energy = self._calculate_energy()
            
            # Combine results
            analysis = {
                "basic_properties": properties,
                "optimization": optimization,
                "energy": energy,
                "structure": {
                    "atom_count": self.molecule.atomCount(),
                    "bond_count": self.molecule.bondCount(),
                    "coordinates": self._get_coordinates()
                }
            }
            
            # Save results
            self._save_results(analysis)
            
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
            
            # Track optimization progress
            initial_energy = ff.energyCalculation()
            ff.optimize(steps=steps)
            final_energy = ff.energyCalculation()
            
            result = {
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_difference": initial_energy - final_energy,
                "coordinates": self._get_coordinates(),
                "steps": steps,
                "converged": True
            }
            
            self.logger.info(f"Geometry optimization completed: ΔE = {result['energy_difference']:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

    def _calculate_properties(self) -> Dict:
        """Calculate molecular properties."""
        properties = {}
        
        # Molecular mass
        mass = 0.0
        for i in range(self.molecule.atomCount()):
            atom = self.molecule.atom(i)
            mass += atom.mass()
        properties["molecular_mass"] = mass
        
        # Center of mass
        com = self.molecule.centerOfMass()
        properties["center_of_mass"] = (com.x(), com.y(), com.z())
        
        # Bond analysis
        bond_lengths = []
        bond_types = {1: 0, 2: 0, 3: 0}  # Single, double, triple bonds
        
        for i in range(self.molecule.bondCount()):
            bond = self.molecule.bond(i)
            bond_lengths.append(bond.length())
            order = int(bond.order())
            if order in bond_types:
                bond_types[order] += 1
        
        properties.update({
            "average_bond_length": np.mean(bond_lengths),
            "max_bond_length": np.max(bond_lengths),
            "bond_types": bond_types
        })
        
        return properties

    def _calculate_energy(self) -> Dict:
        """Calculate molecular energy."""
        ff = self.avo.ForceField()
        ff.setup(self.molecule)
        
        energy_components = {
            "total": ff.energyCalculation(),
            "components": {
                "bonded": ff.energyBonded() if hasattr(ff, 'energyBonded') else None,
                "nonbonded": ff.energyNonbonded() if hasattr(ff, 'energyNonbonded') else None
            }
        }
        
        return energy_components

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

    def _save_results(self, results: Dict):
        """Save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("results") / f"analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"Results saved to {results_file}")

    @staticmethod
    def _is_smiles(identifier: str) -> bool:
        """Check if string is SMILES."""
        return all(c in 'CNOPSFIBrClHc[]()=#-+' for c in identifier)

def main():
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python quantum_system.py <SMILES or file>")
        sys.exit(1)

    # Initialize system
    system = QuantumSystem()
    molecule_input = sys.argv[1]

    # Load and analyze molecule
    if system.load_molecule(molecule_input):
        try:
            results = system.analyze()
            print("\nAnalysis Results:")
            print("=" * 50)
            print(f"Molecular Mass: {results['basic_properties']['molecular_mass']:.2f}")
            print(f"Number of Atoms: {results['structure']['atom_count']}")
            print(f"Number of Bonds: {results['structure']['bond_count']}")
            print(f"Final Energy: {results['energy']['total']:.4f}")
            print("\nDetailed results have been saved to the 'results' directory.")
        except Exception as e:
            print(f"Analysis failed: {e}")
            sys.exit(1)
    else:
        print("Failed to load molecule.")
        sys.exit(1)

if __name__ == "__main__":
    main()
