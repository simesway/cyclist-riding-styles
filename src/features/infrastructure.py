from maneuvers.base import InfrastructureFeatures


class InfrastructureFeatureExtractor:
  """Infers slow-varying infrastructure features per window"""

  def extract(self, window_df) -> InfrastructureFeatures:
    pass