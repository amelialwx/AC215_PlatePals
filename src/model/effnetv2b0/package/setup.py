from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ["wandb==0.15.11", "tensorflow==2.9.0","tensorflow-hub==0.14.0", "google-cloud-storage==1.42.0"]

setup(
    name="platepals-effnetv2b0-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="PlatePals EfficientNetV2B0 Trainer Application",
)
