# ResearchSeminarMARL

**TU Ilmenau | Research Seminar on Multi-Agent Reinforcement Learning (MARL)**  
This project explores and compares the performance of Independent Q-Learning (IQL) and Mean Field Q-Learning (MFQ) agents in cooperative multi-agent environments.

---

# Overview

This repository includes:
- Implementations of IQL and MFQ agents
- Scripts for running experiments
- Training reward visualizations
- Setup documentation and results tracking

The experiments are designed to evaluate learning behavior, cooperation dynamics, and training stability across different parameter configurations.

---

# Agents Implemented

- **IQ**L – Independent Q-Learning  
  Agents treat each other as part of the environment, with no coordination.
  
- **MFQ** – Mean Field Q-Learning  
  Agents approximate the influence of others using a mean-field approach, promoting scalable learning.

---

# Project Structure
├── algo/ # Core learning algorithms
 ├── iql.py
 ├── mfq.py


├── scripts/ # Experiment execution scripts
 ├── run_iql.py
 ├──run_mfq.py
 ├── docs/ # Optional: Reports, setup notes, and results


├── plots/ # Reward graphs and performance visualizations



├── README.md # Project overview and usage guide

├── requirements.txt # Python dependencies

├── .gitignore # Ignored files and folders

👨‍💻 Authors

Sumit Shrivastava
Graduate Student, M.Sc. Research in Computer System and Engineering
TU Ilmenau (2025)
