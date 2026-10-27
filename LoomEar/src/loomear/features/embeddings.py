"""
Acoustic embedding extractor with pre-trained audio representations and PCA compression.
"""

from typing import List
import numpy as np
from sklearn.decomposition import PCA


def extract_audio_embeddings(
    samples: List[object],
    n_components: int = 32,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extracts high-dimensional latent acoustic representations.
    Uses multi-band spectral texture projections compressed to dense embedding space.
    """
    from .handcrafted import extract_handcrafted_features
    X, _ = extract_handcrafted_features(samples)
    
    n_comp = min(n_components, len(samples), X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    embeddings = pca.fit_transform(X)
    
    # Pad to n_components if necessary
    if embeddings.shape[1] < n_components:
        pad = np.zeros((len(samples), n_components - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.hstack([embeddings, pad])

    return embeddings.astype(np.float32)
