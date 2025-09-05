#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Markovian Musical Composition System
"""

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding='utf-8')

# Version
VERSION = '1.0.0'

setup(
    name="markovian-musical-composition",
    version=VERSION,
    description="Advanced musical composition system based on Markov Chains with multiple synthesis methods",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Markovian Musical Composition Team",
    author_email="contact@markovian-music.org",
    url="https://github.com/yourusername/markovian-musical-composition",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="audio music composition markov chains synthesis granular spectral concatenative machine-learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "librosa>=0.8.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "soundfile>=0.10.0",
        "pandas>=1.3.0",
        "networkx>=2.6",
    ],
    extras_require={
        "streamlit": [
            "streamlit>=1.28.0",
        ],
        "desktop": [
            "pygame>=2.1.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "markovian-desktop=audioMarkov_gui_VFrame:main",
            "markovian-streamlit=markov_audio_streamlit:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/markovian-musical-composition/issues",
        "Source": "https://github.com/yourusername/markovian-musical-composition",
        "Documentation": "https://github.com/yourusername/markovian-musical-composition/blob/main/API_DOCUMENTATION.md",
        "Examples": "https://github.com/yourusername/markovian-musical-composition/blob/main/EXAMPLES.md",
    },
)