import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from typing import Dict, List, Tuple, Optional
import json
import websockets
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import avogadro

class AvogadroChatbot:
    def __init__(self):
        # Initialize Avogadro API connection
        self.avo = avogadro.core()
        self.molecule = None
        
        # Initialize intent classification model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        
        # Define command mappings
        self.commands = {
            'load_molecule': self._load_molecule,
            'optimize_geometry': self._optimize_geometry,
            'calculate_energy': self._calculate_energy,
            'visualize': self._visualize,
            'analyze_bonds': self._analyze_bonds
        }

    async def process_message(self, message: str) -> Dict:
        """Process incoming chat messages and route to appropriate handlers"""
        intent = self._classify_intent(message)
        
        if intent not in self.commands:
            return {"error": "Unknown command. Please try a supported operation."}
            
        try:
            result = await self.commands[intent](message)
            return {"success": True, "data": result}
        except Exception as e:
            return {"error": str(e)}

    def _classify_intent(self, message: str) -> str:
        """Classify user intent using BERT"""
        inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        predicted_intent = outputs.logits.argmax().item()
        return self._map_intent_to_command(predicted_intent)

    async def _load_molecule(self, message: str) -> Dict:
        """Load molecule from SMILES or file"""
        # Extract molecule identifier from message
        molecule_id = self._extract_molecule_id(message)
        
        try:
            if self._is_smiles(molecule_id):
                mol = Chem.MolFromSmiles(molecule_id)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                self.molecule = self._convert_rdkit_to_avogadro(mol)
            else:
                self.molecule = self.avo.io.FileFormatManager.readFile(molecule_id)
            
            return {"molecule_loaded": True, "atoms": self.molecule.atomCount()}
        except Exception as e:
            return {"error": f"Failed to load molecule: {str(e)}"}

    async def _optimize_geometry(self, message: str) -> Dict:
        """Optimize molecular geometry using force field"""
        if self.molecule is None:
            return {"error": "No molecule loaded"}
            
        try:
            ff = self.avo.ForceField()
            ff.setup(self.molecule)
            ff.optimize(steps=1000)
            
            energy = ff.energyCalculation()
            return {
                "optimization_complete": True,
                "final_energy": energy,
                "coordinates": self._get_coordinates()
            }
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}

    async def _calculate_energy(self, message: str) -> Dict:
        """Calculate molecular energy"""
        if self.molecule is None:
            return {"error": "No molecule loaded"}
            
        try:
            ff = self.avo.ForceField()
            ff.setup(self.molecule)
            energy = ff.energyCalculation()
            
            components = ff.energyComponents()
            return {
                "total_energy": energy,
                "components": components
            }
        except Exception as e:
            return {"error": f"Energy calculation failed: {str(e)}"}

    async def _visualize(self, message: str) -> Dict:
        """Generate 3D visualization"""
        if self.molecule is None:
            return {"error": "No molecule loaded"}
            
        try:
            view = py3Dmol.view(width=800, height=600)
            
            # Convert Avogadro molecule to format py3Dmol can handle
            xyz = self._get_xyz_string()
            view.addModel(xyz, "xyz")
            
            # Apply visualization settings
            view.setStyle({'stick':{}})
            view.zoomTo()
            
            return {
                "visualization": view.toHTML(),
                "coordinates": self._get_coordinates()
            }
        except Exception as e:
            return {"error": f"Visualization failed: {str(e)}"}

    async def _analyze_bonds(self, message: str) -> Dict:
        """Analyze molecular bonds"""
        if self.molecule is None:
            return {"error": "No molecule loaded"}
            
        try:
            bonds = []
            for i in range(self.molecule.bondCount()):
                bond = self.molecule.bond(i)
                bonds.append({
                    "atom1": bond.atom1().atomicNumber(),
                    "atom2": bond.atom2().atomicNumber(),
                    "length": bond.length(),
                    "order": bond.order()
                })
            
            return {
                "bonds": bonds,
                "total_bonds": len(bonds)
            }
        except Exception as e:
            return {"error": f"Bond analysis failed: {str(e)}"}

    def _convert_rdkit_to_avogadro(self, rdkit_mol) -> avogadro.core.Molecule:
        """Convert RDKit molecule to Avogadro format"""
        avo_mol = self.avo.core.Molecule()
        
        conf = rdkit_mol.GetConformer()
        for i in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            avo_atom = avo_mol.addAtom(atom.GetAtomicNum())
            avo_atom.setPosition3d(pos.x, pos.y, pos.z)
            
        for bond in rdkit_mol.GetBonds():
            avo_mol.addBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondTypeAsDouble()
            )
            
        return avo_mol

    def _get_coordinates(self) -> List[Tuple[float, float, float]]:
        """Get atomic coordinates"""
        coords = []
        for i in range(self.molecule.atomCount()):
            atom = self.molecule.atom(i)
            pos = atom.position3d()
            coords.append((pos.x(), pos.y(), pos.z()))
        return coords

    def _get_xyz_string(self) -> str:
        """Convert molecule to XYZ format"""
        xyz = f"{self.molecule.atomCount()}\n\n"
        for i in range(self.molecule.atomCount()):
            atom = self.molecule.atom(i)
            pos = atom.position3d()
            xyz += f"{atom.atomicNumber()} {pos.x()} {pos.y()} {pos.z()}\n"
        return xyz

    @staticmethod
    def _is_smiles(molecule_id: str) -> bool:
        """Check if string is likely a SMILES string"""
        return all(c in 'CNOPSFIBrClHc[]()=#-+' for c in molecule_id)

    @staticmethod
    def _extract_molecule_id(message: str) -> str:
        """Extract molecule identifier from message"""
        # This is a simple implementation - could be enhanced with NLP
        words = message.split()
        for word in words:
            if AvogadroChatbot._is_smiles(word) or word.endswith(('.mol', '.xyz', '.pdb')):
                return word
        return ""

    @staticmethod
    def _map_intent_to_command(intent_id: int) -> str:
        """Map numerical intent to command string"""
        intent_map = {
            0: 'load_molecule',
            1: 'optimize_geometry',
            2: 'calculate_energy',
            3: 'visualize',
            4: 'analyze_bonds'
        }
        return intent_map.get(intent_id, 'unknown')

async def main():
    # Initialize chatbot
    chatbot = AvogadroChatbot()
    
    # Set up websocket server
    async def handler(websocket, path):
        async for message in websocket:
            response = await chatbot.process_message(message)
            await websocket.send(json.dumps(response))
    
    # Start server
    server = await websockets.serve(handler, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
