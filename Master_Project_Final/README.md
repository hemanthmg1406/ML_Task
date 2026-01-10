# Master's Project: Robust ML Pipeline

## 📌 Overview
This project implements a production-ready Machine Learning pipeline using a Champion XGBoost model (Score: 0.77). It includes a modular architecture with automated data cleaning, feature selection, and a rigorous "Crash Test" audit suite.

## 🛠 Prerequisites
* **Python:** 3.10+
* **OS:** Windows / macOS / Linux

## 🚀 Installation
1.  **Create a Virtual Environment:**
    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    \`\`\`
2.  **Install Dependencies:**
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

## ▶️ How to Run
To execute the full pipeline (Data Loading -> Training -> Auditing -> Production Save), run:

\`\`\`bash
python main.py
\`\`\`

## 📂 Project Structure
* \`src/\`: Contains the core logic modules.
    * \`audit.py\`: The robust diagnostic suite (Noise, Drift, Segmentation tests).
    * \`model.py\`: The XGBoost Champion implementation.
* \`config.py\`: Central configuration for features and hyperparameters.
* \`data/\`: Contains the dataset.
