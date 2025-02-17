#!/bin/bash
set -e

# System configuration
INSTALL_DIR="/opt/quantum-chatbot"
VENV_DIR="${INSTALL_DIR}/venv"
CONFIG_DIR="/etc/quantum-chatbot"
LOG_DIR="/var/log/quantum-chatbot"
DATA_DIR="${INSTALL_DIR}/data"
MODEL_DIR="${INSTALL_DIR}/models"

# Create directory structure
echo "Creating directory structure..."
sudo mkdir -p ${INSTALL_DIR}/{src,models,data,config}
sudo mkdir -p ${LOG_DIR}
sudo mkdir -p ${CONFIG_DIR}

# Create Python virtual environment
echo "Setting up Python environment..."
python3 -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

# Install optimized dependencies
pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    joblib \
    nltk \
    beautifulsoup4 \
    scrapy \
    requests \
    pandas \
    networkx \
    pytest \
    pyyaml \
    tqdm

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create configuration file
cat > ${CONFIG_DIR}/config.yml << EOF
quantum_engine:
  model_type: "RandomForest"
  n_estimators: 100
  test_size: 0.2
  random_state: 42
  use_multiprocessing: true
  n_jobs: -1  # Use all available cores

scraping:
  allowed_sites:
    - "pubchem.ncbi.nlm.nih.gov"
    - "www.ebi.ac.uk/chembldb/"
    - "www.drugbank.com"
    - "www.genome.gov"
    - "clinicaltrials.gov"
    - "www.fda.gov"
    - "en.wikipedia.org"
  timeout: 10
  max_retries: 3

processing:
  batch_size: 1000
  chunk_size: 100
  cache_size: 1000
  parallel_jobs: -1

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file: ${LOG_DIR}/quantum-chatbot.log
EOF

# Create systemd service
cat > /etc/systemd/system/quantum-chatbot.service << EOF
[Unit]
Description=Quantum-Inspired Chatbot Service
After=network.target

[Service]
Type=simple
User=quantum-chatbot
Group=quantum-chatbot
Environment=PYTHONPATH=${INSTALL_DIR}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/src/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Create system user
sudo useradd -r -s /bin/false quantum-chatbot
sudo chown -R quantum-chatbot:quantum-chatbot ${INSTALL_DIR} ${LOG_DIR} ${CONFIG_DIR}

# Setup file permissions
sudo chmod -R 755 ${INSTALL_DIR}
sudo chmod -R 644 ${CONFIG_DIR}
sudo chmod -R 755 ${LOG_DIR}

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable quantum-chatbot
sudo systemctl start quantum-chatbot

echo "Installation complete. Check status with: sudo systemctl status quantum-chatbot"
