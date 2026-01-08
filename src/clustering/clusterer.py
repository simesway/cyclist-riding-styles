import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


N_INIT = 20

class Clusterer:
  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def predict(self, X: np.ndarray) -> np.ndarray:
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

  def fit_predict(self, X: np.ndarray) -> np.ndarray:
    self.model_ = KMeans(
      n_clusters=self.k_,
      random_state=self.random_state,
      n_init=N_INIT
    )
    return self.model_.fit_predict(X)

  def predict(self, X: np.ndarray) -> np.ndarray:
    return self.model_.predict(X)



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
