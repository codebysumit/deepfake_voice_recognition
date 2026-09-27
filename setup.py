from setuptools import setup, find_packages

setup(
    name="deepfake-voice-recognition",
    version="1.0.0",
    description="End-to-End Deepfake Audio & AI Voice Cloning Detection using Wav2Vec 2.0 and 1D ResNet.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["deepfake_voice_recognition", "deepfake_voice_recognition.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.35.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
    ],
    extras_require={
        "onnx": ["onnx>=1.15.0", "onnxruntime>=1.16.0", "onnxscript>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "deepfake-voice-recognition=deepfake_voice_recognition.main:main",
        ],
    },
)
