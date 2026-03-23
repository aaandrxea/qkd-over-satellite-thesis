#!/bin/bash

echo "📁 Creazione struttura progetto..."

mkdir -p satellite-qkd-simulation/{config,data,notebooks,src/{geometry,channel,detection,qkd,simulation,utils},tests,results/{raw,processed,plots},docs}

touch satellite-qkd-simulation/README.md
touch satellite-qkd-simulation/requirements.txt

# CONFIG
touch satellite-qkd-simulation/config/{scenario.yaml,satellite.yaml,atmosphere.yaml,detector.yaml}

# NOTEBOOKS
touch satellite-qkd-simulation/notebooks/{01_link_budget.ipynb,02_qber_analysis.ipynb,03_skr_vs_elevation.ipynb}

# SRC - GEOMETRY
touch satellite-qkd-simulation/src/geometry/{tle_loader.py,orbit_propagation.py,link_geometry.py}

# SRC - CHANNEL
touch satellite-qkd-simulation/src/channel/{atmospheric.py,turbulence.py,pointing.py,link_budget.py}

# SRC - DETECTION
touch satellite-qkd-simulation/src/detection/{photon_model.py,noise.py,click_model.py}

# SRC - QKD
touch satellite-qkd-simulation/src/qkd/{qber.py,key_rate.py,decoy_state.py}

# SRC - SIMULATION
touch satellite-qkd-simulation/src/simulation/{runner.py,monte_carlo.py,parameter_sweep.py}

# SRC - UTILS
touch satellite-qkd-simulation/src/utils/{constants.py,units.py,plotting.py}

# TESTS
touch satellite-qkd-simulation/tests/{test_channel.py,test_qber.py,test_geometry.py}

echo "✅ Struttura creata!"
