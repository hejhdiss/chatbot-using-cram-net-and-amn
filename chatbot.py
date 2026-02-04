#!/usr/bin/env python3
#licensed under GPL V3.
import numpy as np
import re
import random
from amn import AdaptiveMemoryNetwork
from cram_net import CRAMNet
np.random.seed(3)
random.seed(3)
class Tokenizer:
    def __init__(self):
        self.word2id = {"<unk>": 0}
    def fit(self, texts):
        for t in texts:
            for w in re.findall(r"[a-z']+", t.lower()):
                if w not in self.word2id:
                    self.word2id[w] = len(self.word2id)
    def encode(self, text):
        return [self.word2id.get(w, 0) for w in re.findall(r"[a-z']+", text)]
    @property
    def vocab_size(self):
        return len(self.word2id)

INTENTS = {
    "greeting": ["hello", "hi", "hey there"],
    "farewell": ["goodbye", "see you later"],
    "wellbeing": ["i feel good", "i am okay"],
    "identity": ["i am a chatbot", "i am not human"],
    "ability": ["i can chat", "i help people"],
    "emotion": ["i am sorry to hear that", "that sounds difficult"],
    "thanks": ["you are welcome", "no problem"],
}

TRAIN_MAP = {
    "hi": "greeting", "hello": "greeting", "hey": "greeting",
    "bye": "farewell", "goodbye": "farewell", "see you": "farewell",
    "how are you": "wellbeing", "how are you doing": "wellbeing",
    "who are you": "identity", "what are you": "identity",
    "what can you do": "ability", "what is your job": "ability",
    "i am sad": "emotion", "i feel bad": "emotion",
    "thanks": "thanks", "thank you": "thanks",
}

ALL_TEXT = list(TRAIN_MAP.keys()) + [r for responses in INTENTS.values() for r in responses]

class Embedder:
    def __init__(self, vocab_size, dim=32):
        self.W = np.random.randn(vocab_size, dim).astype(np.float32) * 0.01 # Smaller init

    def encode(self, ids):
        if not ids:
            return np.zeros(32, dtype=np.float32)
        vec = np.mean(self.W[ids], axis=0)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

class HybridChatbot:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit(ALL_TEXT)
        self.embedder = Embedder(self.tokenizer.vocab_size, dim=32)

        self.amn = AdaptiveMemoryNetwork(
            input_size=32, hidden_size=64, output_size=len(INTENTS),
            memory_manifold_size=48, 
            learning_rate=0.002, # STABILIZED
            dt=0.05,               # STABILIZED
            memory_decay=0.995
        )
        self.intent2id = {k: i for i, k in enumerate(INTENTS)}
        self.id2intent = {i: k for k, i in self.intent2id.items()}

        self.cram = CRAMNet(
            input_size=32, hidden_size=64, output_size=32,
            logic_manifold_size=24, workspace_size=16,
            learning_rate=0.005, hebbian_lr=0.03,
            hebbian_decay=0.001, context_decay=0.995
        )

        self.response_vectors = {}
        for intent, responses in INTENTS.items():
            self.response_vectors[intent] = [self.embedder.encode(self.tokenizer.encode(r)) for r in responses]

    def train(self, epochs=2000): # Increased epochs
        X, Y = [], []
        for text, intent in TRAIN_MAP.items():
            vec = self.embedder.encode(self.tokenizer.encode(text))
            X.append(vec)
            Y.append(self.intent2id[intent])
        X = np.array(X, dtype=np.float32)
        Y = np.eye(len(INTENTS))[Y]
        
        print(f"Training AMN for {epochs} epochs...")
        self.amn.fit(X, Y, epochs=epochs, batch_size=len(X), verbose=1)

        for intent, responses in INTENTS.items():
            for r_vec in self.response_vectors[intent]:
                self.cram.fit(np.array([r_vec]), np.array([r_vec]), epochs=100, batch_size=1, verbose=0)
        print("✓ Training complete\n")

    def reply(self, text):
        vec = self.embedder.encode(self.tokenizer.encode(text))
        out = self.amn.predict(vec[None], reset_memory=False)[0]
        intent = self.id2intent[int(np.argmax(out))]
        cram_out = self.cram.predict(vec[None])[0]
        candidates = self.response_vectors[intent]
        sims = [np.dot(cram_out, c) / (np.linalg.norm(cram_out) * np.linalg.norm(c) + 1e-8) for c in candidates]
        top_idx = np.argsort(sims)[-min(2, len(candidates)):].tolist() # FIXED
        return INTENTS[intent][random.choice(top_idx)]

def main():
    bot = HybridChatbot()
    bot.train(epochs=2000)

    tests = ["hi", "bye", "how are you", "who are you", "what can you do", "i am sad", "thanks"]
    print("\nRunning tests...\n")
    for t in tests:
        vec = bot.embedder.encode(bot.tokenizer.encode(t))
        out = bot.amn.predict(vec[None], reset_memory=False)[0]
        intent = bot.id2intent[int(np.argmax(out))]
        print(f"Input: {t} | Intent: {intent} | Reply: {bot.reply(t)}")

    while True:
        try:
            msg = input("You: ").strip()
            if msg.lower() in ("exit", "quit"): break
            print("Bot:", bot.reply(msg))
        except EOFError: break
import numpy as np
import random

def find_best_seed():
    best_seeds = []
    test_queries = {
        "hi": "greeting",
        "i am sad": "emotion",
        "thanks": "thanks"
    }

    print("Searching for stable seeds...")
    for seed in range(50): 
        np.random.seed(seed)
        random.seed(seed)
        
        bot = HybridChatbot()
        bot.train(epochs=1000) 
        
        correct = 0
        for text, expected_intent in test_queries.items():
            vec = bot.embedder.encode(bot.tokenizer.encode(text))
            out = bot.amn.predict(vec[None], reset_memory=False)[0]
            predicted_intent = bot.id2intent[int(np.argmax(out))]
            if predicted_intent == expected_intent:
                correct += 1
        
        if correct == len(test_queries):
            print(f"✅ Found accurate seed: {seed}")
            best_seeds.append(seed)
            
    return best_seeds
def a():
    a = [5, 6, 11, 12, 14, 17, 18, 29, 30, 33, 34, 35, 36, 39, 41, 44, 46]
    for s in a:
        np.random.seed(s)
        random.seed(s)
        
        bot = HybridChatbot()
        bot.train(epochs=1000)
        
        # Test a difficult sentence
        test_vec = bot.embedder.encode(bot.tokenizer.encode("i am sad"))
        prediction = bot.amn.predict(test_vec[None], reset_memory=False)[0]
        intent = bot.id2intent[int(np.argmax(prediction))]
        
        print(f"Seed {s} predicted intent: {intent}")
        if intent == "emotion":
            print(f"🌟 SEED {s} IS A WINNER!")
if __name__ == "__main__":
    main()