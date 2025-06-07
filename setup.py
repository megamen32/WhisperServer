from setuptools import setup, find_packages

setup(
    name="bezrabotnyiwhisper",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "aiohttp",
        "requests",
        "faster-whisper",
        "python-dotenv",
    ],
    python_requires=">=3.7",
    author="roomhacker",
    description="Client for whisper.bezrabotnyi.com",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)