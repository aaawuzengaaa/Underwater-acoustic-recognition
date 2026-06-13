from setuptools import setup, find_packages

setup(
    name="underwater-acoustic-recognition",
    version="1.0.0",
    author="Zhang Zeng",
    author_email="739155644@qq.com",
    description="水下声学信号智能识别系统",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhangzeng-ai/underwater-acoustic-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
)