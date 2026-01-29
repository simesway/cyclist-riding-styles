import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


N_INIT = 20

class Clusterer:
  def get_params(self) -> dict:
    raise NotImplementedError

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def predict(self, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  @property
  def n_clusters(self) -> int:
    raise NotImplementedError


class KMeansClusterer(Clusterer):
  def __init__(
      self,
      k: int,
      random_state: int | None = None,
  ):
    self.k_ = k
    self.random_state = random_state
    self.model_ = None

  def get_params(self):
    return {"k": self.k_, "random_state": self.random_state}

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    self.model_ = KMeans(
      n_clusters=self.k_,
      random_state=self.random_state,
      n_init=N_INIT
    )
    return self.model_.fit_predict(X)

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.model_.predict(X)

  @property
  def n_clusters(self):
    return self.k_



class AutoKMeansClusterer(Clusterer):
  def __init__(
      self,
      k_range: range,
      random_state: int | None = None,
  ):
    self.k_range = k_range
    self.random_state = random_state
    self.k_ = None
    self.model_ = None

  def get_params(self):
    return {"k_range": self.k_range, "random_state": self.random_state}

  def _select_k(self, X: np.ndarray) -> int:
    inertias = []
    silhouettes = []

    for k in self.k_range:
      model = KMeans(n_clusters=k, random_state=self.random_state, n_init=N_INIT)
      labels = model.fit_predict(X)

      inertias.append(model.inertia_)
      silhouettes.append(silhouette_score(X, labels))

    elbow_k = self.k_range[np.argmin(np.gradient(inertias))]
    sil_k = self.k_range[np.argmax(silhouettes)]

    return min(elbow_k, sil_k)

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    self.k_ = self._select_k(X)
    self.model_ = KMeans(
      n_clusters=self.k_,
      random_state=self.random_state
    )
    return self.model_.fit_predict(X)

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.model_.predict(X)

  @property
  def n_clusters(self) -> int:
    return self.k_


class GMMClusterer(Clusterer):
  def __init__(self, n_components: int, random_state: int | None = None):
    self.n_components = n_components
    self.random_state = random_state
    self.model_ = None

  def get_params(self) -> dict:
    return {"n_components": self.n_components, "random_state": self.random_state}

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    self.model_ = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
    labels = self.model_.fit_predict(X)
    return labels

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.model_.predict(X)

  @property
  def n_clusters(self):
    return self.n_components


class DBSCANClusterer(Clusterer):
  def __init__(self, eps: float, min_samples: int):
    self.eps = eps # Max distance between two samples for one to be considered as in the neighborhood of the other
    self.min_samples = min_samples # Number of samples in a neighborhood for a point to be considered as a core point
    self.model_ = None

  def get_params(self) -> dict:
    return {"eps": self.eps, "min_samples": self.min_samples}

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    labels = self.model_.fit_predict(X)
    return labels

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.model_.fit_predict(X)

  @property
  def n_clusters(self):
    return len(set(self.model_.labels_)) - (1 if -1 in self.model_.labels_ else 0)