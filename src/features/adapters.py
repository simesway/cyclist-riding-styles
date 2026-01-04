from dataclasses import fields as dataclass_fields

class FeatureAdapter:
    def __init__(self, feature_class):
        self.feature_class = feature_class
        self.feature_names = [f.name for f in dataclass_fields(feature_class)]

    def to_vector(self, f) -> list[float]:
        return [getattr(f, name) for name in self.feature_names]

    def from_vector(self, v: list[float]):
        if len(v) != len(self.feature_names):
            raise ValueError("Vector length mismatch")
        return self.feature_class(**dict(zip(self.feature_names, v)))
