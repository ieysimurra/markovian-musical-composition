#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markovian Musical Composition System
====================================

A comprehensive system for musical composition based on Markov Chains with
multiple audio synthesis methods and both desktop and web interfaces.

This package provides:
- Audio analysis and feature extraction
- K-means clustering for audio segmentation
- Markov chain modeling for sequence generation
- Multiple synthesis methods (concatenative, granular, spectral)
- Desktop interface with Tkinter
- Web interface with Streamlit
- Batch processing capabilities

Main Components:
- MultiAudioAnalyzer: Audio feature extraction and clustering
- MarkovTrackGenerator: Markov chain modeling and track generation
- AudioSynthesizer classes: Different synthesis approaches
- GUI interfaces: Desktop and web-based user interfaces

Example Usage:
    Basic usage with the core engine:
    
    >>> from markovian_musical_composition import MultiAudioAnalyzer, MarkovTrackGenerator
    >>> analyzer = MultiAudioAnalyzer(window_length_ms=500)
    >>> analyzer.load_audio_files(['audio1.wav', 'audio2.wav'])
    >>> analyzer.analyze_all_files()
    >>> generator = MarkovTrackGenerator(analyzer)
    >>> generator.generate_tracks(num_tracks=3, duration_seconds=60)
    >>> generator.export_tracks('output_folder')

Authors: Markovian Musical Composition Team
License: MIT
Version: 1.0.0
"""

import sys
import warnings
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "Markovian Musical Composition Team"
__email__ = "contact@markovian-music.org"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Markovian Musical Composition Team"

# Package metadata
__title__ = "markovian-musical-composition"
__description__ = "Advanced musical composition system based on Markov Chains"
__url__ = "https://github.com/yourusername/markovian-musical-composition"

# Minimum Python version check
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Suppress common warnings during import
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Import core classes and functions
try:
    # Core analysis and generation classes
    from .script11_Markov_Audio1GeraCompMult_GUI import (
        # Main classes
        MultiAudioAnalyzer,
        MarkovTrackGenerator,
        AudioProcessor,
        
        # Synthesis classes
        AudioSynthesizer,
        ConcatenativeSynthesizer,
        GranularSynthesizer,
        SpectralSynthesizer,
        
        # Enums
        SynthesisType,
        DurationMode,
        WindowType,
        
        # Data classes
        AudioData,
    )
    
    # GUI module (optional import)
    _GUI_AVAILABLE = True
    try:
        from . import audioMarkov_gui_VFrame
    except ImportError as e:
        _GUI_AVAILABLE = False
        _GUI_IMPORT_ERROR = str(e)
    
    # Streamlit module (optional import)
    _STREAMLIT_AVAILABLE = True
    try:
        from . import markov_audio_streamlit
    except ImportError as e:
        _STREAMLIT_AVAILABLE = False
        _STREAMLIT_IMPORT_ERROR = str(e)

except ImportError as e:
    raise ImportError(
        f"Failed to import core modules: {e}\n"
        "Please ensure all dependencies are installed:\n"
        "pip install -r requirements.txt"
    )

# Define public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Core classes
    'MultiAudioAnalyzer',
    'MarkovTrackGenerator', 
    'AudioProcessor',
    
    # Synthesis classes
    'AudioSynthesizer',
    'ConcatenativeSynthesizer',
    'GranularSynthesizer',
    'SpectralSynthesizer',
    
    # Enums
    'SynthesisType',
    'DurationMode',
    'WindowType',
    
    # Data classes
    'AudioData',
    
    # Utility functions
    'get_version',
    'check_dependencies',
    'get_system_info',
]

def get_version():
    """Return the package version."""
    return __version__

def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Dictionary with dependency status information
    """
    deps_status = {
        'core': True,
        'gui': _GUI_AVAILABLE,
        'streamlit': _STREAMLIT_AVAILABLE,
        'missing_deps': [],
        'warnings': []
    }
    
    # Check core dependencies
    required_modules = [
        'numpy', 'librosa', 'matplotlib', 'seaborn', 
        'sklearn', 'scipy', 'soundfile', 'pandas'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            deps_status['core'] = False
            deps_status['missing_deps'].append(module)
    
    # Add warnings for optional dependencies
    if not _GUI_AVAILABLE:
        deps_status['warnings'].append(f"GUI not available: {_GUI_IMPORT_ERROR}")
    
    if not _STREAMLIT_AVAILABLE:
        deps_status['warnings'].append(f"Streamlit interface not available: {_STREAMLIT_IMPORT_ERROR}")
    
    return deps_status

def get_system_info():
    """
    Get system information relevant for troubleshooting.
    
    Returns:
        dict: System information dictionary
    """
    import platform
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'package_version': __version__,
        'package_location': str(Path(__file__).parent),
        'dependencies': check_dependencies()
    }
    
    return info

def print_system_info():
    """Print system information for debugging purposes."""
    info = get_system_info()
    
    print("Markovian Musical Composition System - System Information")
    print("=" * 60)
    print(f"Package Version: {info['package_version']}")
    print(f"Python Version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Package Location: {info['package_location']}")
    
    deps = info['dependencies']
    print(f"\nDependency Status:")
    print(f"  Core dependencies: {'✓' if deps['core'] else '✗'}")
    print(f"  GUI available: {'✓' if deps['gui'] else '✗'}")
    print(f"  Streamlit available: {'✓' if deps['streamlit'] else '✗'}")
    
    if deps['missing_deps']:
        print(f"\nMissing dependencies: {', '.join(deps['missing_deps'])}")
        print("Install with: pip install -r requirements.txt")
    
    if deps['warnings']:
        print(f"\nWarnings:")
        for warning in deps['warnings']:
            print(f"  - {warning}")

# Convenience functions for quick usage
def quick_analyze(audio_files, window_length_ms=500, k_clusters=None):
    """
    Quick audio analysis function.
    
    Args:
        audio_files (list): List of audio file paths
        window_length_ms (float): Analysis window size in milliseconds
        k_clusters (int, optional): Number of clusters (auto-detect if None)
    
    Returns:
        MultiAudioAnalyzer: Configured and analyzed audio analyzer
    """
    analyzer = MultiAudioAnalyzer(window_length_ms=window_length_ms)
    analyzer.load_audio_files(audio_files)
    analyzer.analyze_all_files(k=k_clusters)
    return analyzer

def quick_generate(analyzer, num_tracks=3, duration_seconds=60, 
                  synthesis_type=None, output_folder='quick_output'):
    """
    Quick track generation function.
    
    Args:
        analyzer (MultiAudioAnalyzer): Analyzed audio data
        num_tracks (int): Number of tracks to generate
        duration_seconds (float): Duration of each track
        synthesis_type (SynthesisType, optional): Synthesis method
        output_folder (str): Output folder name
    
    Returns:
        MarkovTrackGenerator: Generator with created tracks
    """
    if synthesis_type is None:
        synthesis_type = SynthesisType.CONCATENATIVE
    
    generator = MarkovTrackGenerator(analyzer)
    generator.generate_tracks(
        num_tracks=num_tracks,
        duration_seconds=duration_seconds,
        synthesis_type=synthesis_type
    )
    generator.export_tracks(output_folder)
    return generator

# Add quick functions to public API
__all__.extend(['quick_analyze', 'quick_generate', 'print_system_info'])

# Package initialization message (only in debug mode)
if __debug__ and sys.flags.verbose:
    print(f"Markovian Musical Composition System v{__version__} loaded successfully")
    deps = check_dependencies()
    if not deps['core']:
        print(f"Warning: Some core dependencies are missing: {deps['missing_deps']}")
    if deps['warnings']:
        for warning in deps['warnings']:
            print(f"Warning: {warning}")
