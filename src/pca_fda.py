import numpy as np
import umap

def perform_pca(X_train, X_test, variance_threshold=None, n_comp=None):
    """
    Applies Principal Component Analysis (PCA) exactly as written in your SML A2 assignment.
    It manually computes the covariance matrix and eigenvalues.
    
    Note: Your SML code assumed X was shaped (Features, Samples). 
    We transpose X here because CNN embeddings are shaped (Samples, Features).
    """
    # Transpose to match your SML logic (Features x Samples)
    X_train_t = X_train.T
    X_test_t = X_test.T
    
    # 1. Center the data
    mu = np.mean(X_train_t, axis=1, keepdims=True)
    Xc_train = X_train_t - mu
    Xc_test = X_test_t - mu

    # 2. Compute Covariance Matrix
    S = (Xc_train @ Xc_train.T) / (X_train_t.shape[1] - 1)
    
    # 3. Eigendecomposition
    evals, evecs = np.linalg.eigh(S)

    # 4. Sort in descending order
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    # 5. Determine k (number of components)
    if n_comp is not None:
        k = n_comp
    elif variance_threshold is not None:
        total_var = np.sum(evals)
        k = np.argmax(np.cumsum(evals) / total_var >= variance_threshold) + 1
    else:
        k = X_train_t.shape[0]

    # 6. Projection
    Up = evecs[:, :k]
    
    # Project and transpose back to (Samples, Features) for standard ML compatibility
    Y_train = (Up.T @ Xc_train).T
    Y_test = (Up.T @ Xc_test).T
    
    return Y_train, Y_test, Up, mu, k

def perform_fda(X_train, y_train, X_test, n_components=None):
    """
    Applies Fisher's Discriminant Analysis (FDA) as written in your SML A2 assignment.
    Maximizes class separability by using Within-class (Sw) and Between-class (Sb) scatter matrices.
    """
    # Transpose to match your SML logic (Features x Samples)
    X_train_t = X_train.T
    X_test_t = X_test.T
    
    n_features = X_train_t.shape[0]
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # Default to C-1 components if not specified (FDA mathematically produces at most C-1)
    if n_components is None:
        n_components = n_classes - 1

    overall_mean = np.mean(X_train_t, axis=1, keepdims=True)
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    # Calculate Scatter Matrices
    for c in classes:
        X_c = X_train_t[:, y_train == c]
        n_c = X_c.shape[1]
        mu_c = np.mean(X_c, axis=1, keepdims=True)
        
        diff = X_c - mu_c
        Sw += diff @ diff.T
        Sb += n_c * ((mu_c - overall_mean) @ (mu_c - overall_mean).T)

    # Add a small constant to diagonal to prevent singular matrix errors (regularization)
    Sw_reg = Sw + np.eye(n_features) * 1e-6
    
    # Solve generalized eigenvalue problem
    evals_f, evecs_f = np.linalg.eig(np.linalg.inv(Sw_reg) @ Sb)
    
    # Sort and pick top k components (we take the real part to handle complex floating point inaccuracies)
    idx = np.argsort(evals_f.real)[::-1]
    W = evecs_f[:, idx[:n_components]].real

    # Project and transpose back
    X_fda_train = (W.T @ X_train_t).T
    X_fda_test = (W.T @ X_test_t).T

    return X_fda_train, X_fda_test, W

def apply_umap(X_train, X_test, n_components=2, random_state=42):
    """
    A modern alternative to PCA/FDA for visualizing the 2048-dimensional embeddings.
    UMAP excels at preserving non-linear cluster structures in medical images.
    """
    print(f"Applying UMAP (reducing to {n_components} dimensions)...")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    
    # UMAP expects (Samples, Features) so no transposing needed
    Y_train = reducer.fit_transform(X_train)
    Y_test = reducer.transform(X_test)
    
    return Y_train, Y_test
