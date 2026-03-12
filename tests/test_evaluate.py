import numpy as np
from src.evaluate import nearest_neighbors, evaluate_analogies


def test_nearest_neighbors_returns_closest():
    # 4-word vocab, 2-dim embeddings
    # word 0 and word 1 are very close; word 2 is far away
    W = np.array([[1.0, 0.0],
                  [0.9, 0.1],
                  [-1.0, 0.0],
                  [0.0, 1.0]])
    word2idx = {"cat": 0, "dog": 1, "car": 2, "sky": 3}
    idx2word = {0: "cat", 1: "dog", 2: "car", 3: "sky"}
    neighbors = nearest_neighbors(W, word2idx, idx2word, "cat", n=2)
    assert neighbors[0] == "dog"


def test_evaluate_analogies_perfect():
    # a:b :: c:d — "king - man + woman = queen"
    # Set up embeddings where the analogy holds exactly
    W = np.array([
        [1.0, 0.0],   # 0: man
        [1.0, 1.0],   # 1: king
        [0.0, 0.0],   # 2: woman
        [0.0, 1.0],   # 3: queen
    ])
    word2idx = {"man": 0, "king": 1, "woman": 2, "queen": 3}
    idx2word = {v: k for k, v in word2idx.items()}
    analogies = [("man", "king", "woman", "queen")]
    correct, total = evaluate_analogies(W, word2idx, idx2word, analogies)
    assert correct == 1 and total == 1
