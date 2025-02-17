#!/usr/bin/env python3
import os
import re
from pathlib import Path

def fix_imports(file_path: Path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Update relative imports
    replacements = {
        'import quantum_': 'from modules.core import quantum_',
        'from quantum_': 'from modules.core.quantum_',
        'import avogadro_': 'from modules.interfaces import avogadro_',
        'from avogadro_': 'from modules.interfaces.avogadro_',
        'import ssl_certificate_': 'from modules.automation import ssl_certificate_',
        'from ssl_certificate_': 'from modules.automation.ssl_certificate_',
        'import domain_credentials': 'from modules.automation import domain_credentials',
        'from domain_credentials': 'from modules.automation.domain_credentials'
    }

    for old, new in replacements.items():
        content = re.sub(f'^{old}', new, content, flags=re.MULTILINE)

    with open(file_path, 'w') as f:
        f.write(content)

def main():
    root_dir = Path('modules')
    for file in root_dir.rglob('*.py'):
        if file.name != '__init__.py':
            fix_imports(file)

if __name__ == '__main__':
    main()