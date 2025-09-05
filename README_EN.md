# Markovian Musical Composition System

An advanced musical composition system based on Markov Chains with graphical interface and support for multiple audio synthesis methods.

## 🎵 About the Project

This project implements an intelligent musical composition system that uses **Markov Chains** to analyze existing audio files and generate new sonic material. The system offers three distinct audio synthesis methods and two user interfaces for maximum flexibility.

### Key Features

- **Automatic Analysis**: Spectral feature extraction and K-means clustering
- **Multiple Synthesis Methods**:
  - Concatenative with crossfade
  - Granular with density and pitch control
  - Spectral with frequency manipulation
- **Flexible Duration Modes**:
  - Fixed duration based on analysis window
  - Average duration per cluster
  - Duration based on consecutive sequences
- **Two Interfaces**:
  - Desktop interface with Tkinter (local)
  - Web interface with Streamlit (cloud)

## 🚀 Installation and Usage

### Option 1: Desktop Interface (Local)

#### Prerequisites
```bash
pip install numpy librosa matplotlib seaborn scikit-learn scipy soundfile pandas pygame tkinter
```

#### Execution
```bash
python audioMarkov_gui_VFrame.py
```

The desktop interface offers:
- Guided step-by-step navigation
- Integrated audio player
- Detailed visualizations
- Complete statistical analysis
- Organized folder export

### Option 2: Web Interface (Streamlit)

#### Prerequisites
```bash
pip install streamlit numpy librosa matplotlib seaborn scikit-learn scipy soundfile pandas
```

#### Local Execution
```bash
streamlit run markov_audio_streamlit.py
```

#### Deploy on Streamlit Cloud
1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the `markov_audio_streamlit.py` file

The web interface offers:
- Responsive and intuitive interface
- Direct file upload
- Interactive visualizations
- ZIP download of results
- Mobile device compatibility

## 📋 How to Use

### Workflow

1. **File Selection**: Load audio files (WAV, MP3, FLAC, OGG, AIFF)
2. **Analysis Configuration**: 
   - Set window size (100-1000ms)
   - Choose number of clusters (automatic or manual)
3. **Analysis**: System extracts features and creates clusters
4. **Generation Configuration**:
   - Number of desired tracks
   - Total duration
   - Synthesis method
   - Duration mode
5. **Generation**: Creation of Markov Chain-based compositions
6. **Playback and Download**: Test and download results

### Synthesis Parameters

#### Concatenative Synthesis
- **Crossfade Duration**: Transition duration between segments (0.01-0.5s)

#### Granular Synthesis
- **Grain Size**: Size of sound grains (0.01-0.5s)
- **Density**: Number of grains per second (10-500)
- **Pitch Shift**: Pitch alteration in semitones (-12 to +12)
- **Position/Duration Jitter**: Random variation (0-0.5)

#### Spectral Synthesis
- **Preserve Transients**: Maintains transient characteristics
- **Spectral Stretch**: Spectral stretching factor (0.1-3.0)
- **FFT Size**: FFT window size (512, 1024, 2048, 4096)

## 🔧 Code Structure

### Main Files

- `audioMarkov_gui_VFrame.py`: Complete desktop interface with Tkinter
- `script11_Markov_Audio1GeraCompMult_GUI.py`: Main analysis and synthesis engine
- `markov_audio_streamlit.py`: Web interface with Streamlit

### Main Classes

- `MultiAudioAnalyzer`: Multi-file audio analysis
- `MarkovTrackGenerator`: Track generation using Markov chains
- `AudioSynthesizer`: Base classes for synthesis
- `ConcatenativeSynthesizer`: Concatenative synthesis
- `GranularSynthesizer`: Granular synthesis
- `SpectralSynthesizer`: Spectral synthesis

## 📊 Technical Features

### Audio Analysis
- MFCC extraction (13 coefficients)
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- RMS Energy
- Zero Crossing Rate

### Clustering
- K-means with automatic optimal cluster detection
- Feature normalization
- Silhouette analysis for validation

### Markovian Modeling
- Initial probability calculation
- Transition matrix based on real data
- Probabilistic sequence generation

## 📁 Output Structure

```
output_multitrack_[id]/
├── track_1/
│   ├── audio.wav
│   ├── metadata.json
│   └── analysis/
│       ├── sequence.csv
│       ├── transition_matrix.csv
│       ├── statistics.txt
│       └── analysis.png
├── track_2/
│   └── ...
├── final_mix.wav
└── mix_analysis/
    ├── combined_analysis.png
    └── summary.txt
```

## 🎯 Use Cases

- **Composers**: Creation of experimental musical material
- **Sound Designers**: Generation of unique sonic textures
- **Researchers**: Study of musical structures and synthesis
- **Educators**: Demonstration of musical analysis concepts
- **Artists**: Exploration of new creative possibilities

## 🔬 Scientific Methodology

The system is based on solid principles of:
- **Digital Signal Processing**: Advanced spectral analysis
- **Machine Learning**: Unsupervised clustering
- **Information Theory**: Probabilistic modeling
- **Audio Synthesis**: Multiple sound generation approaches

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is under the MIT license. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- Librosa library for audio analysis
- Scikit-learn for machine learning algorithms
- Streamlit for modern web interface
- Tkinter for robust desktop interface

## 📞 Contact

For questions, suggestions, or collaborations, open an issue on GitHub.

---

**Transform your audio files into new compositions with the power of Markov Chains!**