# AES-10 Round Side-Channel Attack Simulator

## Overview
This project demonstrates **Correlation Power Analysis (CPA)** and **CNN-based Deep Learning** attacks on an AES-10 round implementation. It simulates power traces using the Hamming Weight leakage model and provides an interactive web interface to visualize the attack in real-time.

## Features
- Simulates AES-10 round encryption
- CPA attack on the first round S-Box output
- CNN-based machine learning attack
- One-click toggle for Masking countermeasure
- Live correlation plots
- Interactive Streamlit web interface

## Results

| Model | Unprotected Accuracy | Protected Accuracy |
|-------|---------------------|-------------------|
| **Random Forest** | 100% | 0.01% |
| **CNN (Deep Learning)** | 100% | 0.30% |

Both models achieve **100% accuracy** on unprotected traces but **fail completely** when masking is applied—proving the effectiveness of countermeasures.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/aes-side-channel-attack-simulator.git
cd aes-side-channel-attack-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Technologies Used
- Python, Streamlit
- TensorFlow / Keras (CNN)
- Scikit-learn (Random Forest)
- NumPy, Matplotlib

## Project Structure
```
aes-side-channel-attack-simulator/
│
├── app.py                              # Streamlit web interface
├── side_channel_simulation.ipynb       # Jupyter Notebook (research)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore file
└──side_channel_attack_simulator.pptx   # Presentation slides
```

## How It Works

### CPA Attack
The Correlation Power Analysis attack targets the first round S-Box output of AES:
```
Intermediate = S-BOX(Plaintext ⊕ Key)
```
For each key guess (0-255), the attack predicts power using Hamming Weight and calculates correlation with actual traces. The correct key shows the highest correlation peak.

### CNN Attack
A Convolutional Neural Network with 2 Conv1D layers is trained on full 10-round power traces to predict:
```
Target = Plaintext ⊕ Key
```
The CNN achieves ~95-100% accuracy on unprotected traces.

### Countermeasures
Masking is implemented as:
```python
mask = np.random.randint(0, 256)
masked_plaintext = plaintext ^ mask
intermediate = masked_plaintext ^ key
```
This breaks the correlation and defeats both CPA and CNN attacks.

## Results Summary
- CPA successfully recovers AES keys from unprotected implementations
- CNN achieves ~95% accuracy on unprotected traces
- Both attacks fail against masked implementations (0.01-0.30% accuracy)
- Masking reduces leakage by over 94%

## Future Work
- **Real Hardware Implementation** – Deploy on Arduino/FPGA; capture actual power traces using oscilloscopes
- **Full AES-128 Recovery** – Attack all 16 S-Box outputs simultaneously to recover the complete key
- **Advanced Deep Learning** – Deploy deeper architectures like ResNets, Transformers, or Ensemble Models
- **Advanced Countermeasures** – Test shuffling, desynchronization, and hardware-level constant-power logic
- **Cross-Device Attacks** – Test transfer learning across different devices
