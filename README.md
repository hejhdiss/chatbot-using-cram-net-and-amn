# Hybrid Chatbot using CRAM-Net and AMN

A simple yet powerful hybrid chatbot that combines two experimental neural architectures: **Adaptive Memory Network (AMN)** and **CRAM-Net**. This implementation demonstrates how memory-native and logical-manifold-based networks can be used for intent classification and response selection.

---

## 🚀 Features

- **Hybrid Architecture:** Uses AMN for high-level intent classification and CRAM-Net for contextual response refinement.  
- **Adaptive Memory:** Leverages a memory manifold that stabilizes over training epochs.  
- **Hebbian Learning:** Incorporates Hebbian decay and context-sensitive logic via CRAM-Net.  
- **Lightweight:** Minimal dependencies, focused on `numpy` and specialized architectures.  

---

## 🛠️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/hejhdiss/chatbot-using-cram-net-and-amn
cd chatbot-using-cram-net-and-amn
```

## 2. Add Dependencies

This chatbot requires the core logic from two other repositories. You must download the following files and place them in the root directory of this project:

AMN (Adaptive Memory Network):
Download amn.py from: MEMORY-NATIVE-NEURAL_NETWORK

CRAM-Net:
Download cram_net.py (or cram-net.py) from: CRAM-Net
⚠️ If the file is named cram-net.py, rename it to cram_net.py so the Python import works:
```
from cram_net import CRAMNet
```

## 3. Install Requirements
```
pip install numpy
```

## 🧠 How it Works

The chatbot processes input in several stages:

- Tokenizer/Embedder: Converts text into normalized mean-vector embeddings.

- AMN (Intent Classifier): The Adaptive Memory Network predicts the "intent" (e.g., greeting, farewell, emotion) based on learned memory manifolds.

- CRAM-Net (Logic Processor): Processes the input vector against a workspace to refine the output vector.

- Response Selection: Calculates cosine similarity between the CRAM-Net output and potential responses within the predicted intent category to choose the most relevant reply.

## 🖥️ Usage

Run the chatbot script to train the model and start a conversation:
```
python chatbot.py
```

Upon running, the bot will:

- Train the AMN for 2000 epochs.

- Run a set of built-in test strings.

- Open an interactive CLI for you to chat. Type exit to stop.

## License 
This project is licensed under the **GPL V3 License** – see the LICENSE file for details.
