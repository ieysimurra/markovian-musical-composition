# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-language support (English/Portuguese)
- Advanced error handling and recovery
- Mobile-responsive Streamlit interface

### Changed
- Improved clustering algorithm performance
- Enhanced audio visualization quality

### Fixed
- Memory usage optimization for large files
- Cross-platform compatibility issues

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Markovian Musical Composition System
- Desktop interface with Tkinter (audioMarkov_gui_VFrame.py)
- Core analysis engine (script11_Markov_Audio1GeraCompMult_GUI.py)
- Streamlit web interface (markov_audio_streamlit.py)
- Three synthesis methods:
  - Concatenative synthesis with crossfade
  - Granular synthesis with parameter control
  - Spectral synthesis with frequency manipulation
- Multiple duration modes:
  - Fixed duration based on analysis window
  - Cluster-average based duration
  - Consecutive sequence based duration
- Automatic K-means clustering with optimal k detection
- Comprehensive audio feature extraction:
  - MFCCs (13 coefficients)
  - Spectral centroid, bandwidth, rolloff
  - RMS energy and zero crossing rate
- Audio file format support: WAV, MP3, FLAC, OGG, AIFF
- Interactive visualizations and analysis
- Detailed export with metadata and statistics
- Multi-track generation and final mix creation

### Features
- **Analysis Engine**: Advanced audio feature extraction and clustering
- **Markov Modeling**: Probabilistic sequence generation
- **Synthesis Methods**: Multiple audio synthesis approaches
- **User Interfaces**: Both desktop and web-based interfaces
- **Export System**: Organized output with analysis data
- **Visualization**: Interactive plots and spectrograms
- **Parameter Control**: Extensive customization options

### Technical Details
- Python 3.8+ compatibility
- NumPy-based audio processing
- Librosa for advanced audio analysis
- Scikit-learn for machine learning
- Matplotlib/Seaborn for visualizations
- Streamlit for web interface
- Tkinter for desktop interface

### Documentation
- Comprehensive README in English and Portuguese
- API documentation with examples
- Installation guide for multiple platforms
- Contributing guidelines
- Example usage patterns

---

# Registro de Mudanças

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

## [Não Lançado]

### Adicionado
- Suporte multi-idioma (Inglês/Português)
- Tratamento avançado de erros e recuperação
- Interface Streamlit responsiva para mobile

### Alterado
- Performance melhorada do algoritmo de clustering
- Qualidade aprimorada das visualizações de áudio

### Corrigido
- Otimização do uso de memória para arquivos grandes
- Problemas de compatibilidade entre plataformas

## [1.0.0] - 2024-01-XX

### Adicionado
- Lançamento inicial do Sistema de Composição Musical Markoviano
- Interface desktop com Tkinter (audioMarkov_gui_VFrame.py)
- Engine principal de análise (script11_Markov_Audio1GeraCompMult_GUI.py)
- Interface web Streamlit (markov_audio_streamlit.py)
- Três métodos de síntese:
  - Síntese concatenativa com crossfade
  - Síntese granular com controle de parâmetros
  - Síntese espectral com manipulação de frequências
- Múltiplos modos de duração:
  - Duração fixa baseada na janela de análise
  - Duração baseada na média do cluster
  - Duração baseada em sequências consecutivas
- Clustering K-means automático com detecção de k ótimo
- Extração abrangente de características de áudio:
  - MFCCs (13 coeficientes)
  - Centroide, largura de banda e rolloff espectrais
  - Energia RMS e taxa de cruzamento por zero
- Suporte a formatos de arquivo: WAV, MP3, FLAC, OGG, AIFF
- Visualizações interativas e análises
- Exportação detalhada com metadados e estatísticas
- Geração multi-track e criação de mix final