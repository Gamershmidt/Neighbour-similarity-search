import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


def PPA(X_train):
    # First PCA to get Top Components
    pca = PCA(n_components=50)
    X_train = X_train - np.mean(X_train)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    pca_embeddings = {}
    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
        for u in U1[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)

    z = np.asarray(z).astype(np.float32)
    X_train = z - np.mean(z)

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    return X_train


def PPA_PCA(X_train, n=100):
    X_train = PPA(X_train)
    # PCA for Dim Reduction
    pca = PCA(n_components=n)
    X_new = pca.fit_transform(X_train)
    return X_new
