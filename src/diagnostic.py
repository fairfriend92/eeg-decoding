from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4, fmin=4.0, fmax=38.0, tmin=0.0, tmax=4.0, resample=250.0)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print(metadata["session"].unique())
print(metadata["session"].value_counts())