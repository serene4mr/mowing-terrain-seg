from setuptools import setup, find_packages

setup(
    name="offroad-segmentation-benchmark",
    version="0.1.0",
    description="Off-road navigation segmentation benchmark",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "mmcv>=2.0.0",
        "mmsegmentation>=1.0.0",
        "mmengine>=0.7.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
)
