# API Documentation / Documentação da API

## English

### Core Classes

#### MultiAudioAnalyzer

Main class for analyzing multiple audio files and extracting features.

```python
from script11_Markov_Audio1GeraCompMult_GUI import MultiAudioAnalyzer

# Initialize analyzer
analyzer = MultiAudioAnalyzer(window_length_ms=500)

# Load audio files
file_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
analyzer.load_audio_files(file_paths)

# Analyze files
analyzer.analyze_all_files(k=5)  # k=None for automatic detection
```

**Parameters:**
- `window_length_ms` (float): Analysis window size in milliseconds
- `k` (int, optional): Number of clusters. If None, automatically determined

**Methods:**
- `load_audio_files(file_paths)`: Load audio files from paths
- `analyze_all_files(k=None)`: Extract features and perform clustering
- `extract_features(audio_data)`: Extract features from single audio file

#### MarkovTrackGenerator

Class for generating musical tracks using Markov chains.

```python
from script11_Markov_Audio1GeraCompMult_GUI import MarkovTrackGenerator, SynthesisType, DurationMode

# Initialize generator
generator = MarkovTrackGenerator(analyzer)

# Set duration mode
generator.set_duration_mode(DurationMode.CLUSTER_MEAN)

# Generate tracks
generator.generate_tracks(
    num_tracks=3,
    duration_seconds=60,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params={
        'grain_size': 0.1,
        'density': 100,
        'pitch_shift': 0
    }
)

# Export results
generator.export_tracks('output_folder')
```

**Parameters:**
- `num_tracks` (int): Number of tracks to generate
- `duration_seconds` (float): Duration of each track
- `synthesis_type` (SynthesisType): Type of synthesis to use
- `synthesis_params` (dict): Parameters specific to synthesis method

#### Synthesis Classes

##### ConcatenativeSynthesizer

```python
from script11_Markov_Audio1GeraCompMult_GUI import ConcatenativeSynthesizer

synthesizer = ConcatenativeSynthesizer(sr=44100, crossfade_duration=0.1)
result = synthesizer.synthesize(segments, crossfade_duration=0.05)
```

##### GranularSynthesizer

```python
from script11_Markov_Audio1GeraCompMult_GUI import GranularSynthesizer

synthesizer = GranularSynthesizer(sr=44100)
synthesizer.grain_size = 0.1
synthesizer.density = 100
synthesizer.pitch_shift = 2
result = synthesizer.synthesize(source_audio, duration=30.0)
```

##### SpectralSynthesizer

```python
from script11_Markov_Audio1GeraCompMult_GUI import SpectralSynthesizer

synthesizer = SpectralSynthesizer(sr=44100)
synthesizer.preserve_transients = True
synthesizer.spectral_stretch = 1.5
result = synthesizer.synthesize(source_audio, target_features)
```

### Enums

#### SynthesisType
- `CONCATENATIVE`: Concatenative synthesis with crossfade
- `GRANULAR`: Granular synthesis
- `SPECTRAL`: Spectral synthesis

#### DurationMode
- `FIXED`: Fixed duration based on window size
- `CLUSTER_MEAN`: Duration based on cluster averages
- `SEQUENCE_LENGTH`: Duration based on consecutive sequences

#### WindowType
- `HANN`: Hann window
- `HAMMING`: Hamming window
- `BLACKMAN`: Blackman window
- `GAUSSIAN`: Gaussian window
- `KAISER`: Kaiser window
- `RECTANGULAR`: Rectangular window

### Example: Complete Workflow

```python
import numpy as np
from script11_Markov_Audio1GeraCompMult_GUI import (
    MultiAudioAnalyzer, MarkovTrackGenerator, 
    SynthesisType, DurationMode
)

# 1. Initialize and analyze
analyzer = MultiAudioAnalyzer(window_length_ms=500)
analyzer.load_audio_files(['input1.wav', 'input2.wav'])
analyzer.analyze_all_files()

# 2. Generate tracks
generator = MarkovTrackGenerator(analyzer)
generator.set_duration_mode(DurationMode.CLUSTER_MEAN)

generator.generate_tracks(
    num_tracks=5,
    duration_seconds=90,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params={
        'grain_size': 0.15,
        'density': 80,
        'pitch_shift': 0,
        'position_jitter': 0.2,
        'duration_jitter': 0.1
    }
)

# 3. Exportar resultados
generator.export_tracks('minha_composicao')
```

### Estrutura de Dados de Saída

#### AudioData
```python
@dataclass
class AudioData:
    filename: str
    signal: np.ndarray
    sr: int
    features: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
```

#### Track Dictionary
```python
track = {
    'sequence': List[int],          # Sequência de estados
    'times': List[float],           # Tempos correspondentes
    'initial_probs': np.ndarray,    # Probabilidades iniciais
    'transition_probs': np.ndarray, # Matriz de transição
    'audio': np.ndarray,            # Áudio gerado
    'durations': List[float]        # Durações (se aplicável)
}
```

### Parâmetros de Síntese Detalhados

#### Síntese Concatenativa
```python
synthesis_params = {
    'crossfade_duration': 0.1  # Duração do crossfade em segundos
}
```

#### Síntese Granular
```python
synthesis_params = {
    'grain_size': 0.1,          # Tamanho do grão em segundos
    'density': 100,             # Grãos por segundo
    'pitch_shift': 0,           # Mudança de pitch em semitons
    'position_jitter': 0.1,     # Variação de posição (0-1)
    'duration_jitter': 0.1      # Variação de duração (0-1)
}
```

#### Síntese Espectral
```python
synthesis_params = {
    'preserve_transients': True,  # Preservar transientes
    'spectral_stretch': 1.0,      # Fator de esticamento
    'fft_size': 2048             # Tamanho da FFT
}
```

### Tratamento de Erros

```python
try:
    analyzer.analyze_all_files()
except ValueError as e:
    print(f"Erro na análise: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")
```

### Otimização de Performance

#### Para Arquivos Grandes
```python
# Reduzir tamanho da janela
analyzer = MultiAudioAnalyzer(window_length_ms=250)

# Usar menos clusters
analyzer.analyze_all_files(k=3)
```

#### Para Múltiplos Processamentos
```python
# Reutilizar analisador
analyzer = MultiAudioAnalyzer(window_length_ms=500)
for file_batch in file_batches:
    analyzer.load_audio_files(file_batch)
    analyzer.analyze_all_files()
    # ... processar resultados
```seconds=90,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params={
        'grain_size': 0.15,
        'density': 80,
        'pitch_shift': 0,
        'position_jitter': 0.2,
        'duration_jitter': 0.1
    }
)

# 3. Export results
generator.export_tracks('my_composition')
```

---

## Português

### Classes Principais

#### MultiAudioAnalyzer

Classe principal para analisar múltiplos arquivos de áudio e extrair características.

```python
from script11_Markov_Audio1GeraCompMult_GUI import MultiAudioAnalyzer

# Inicializar analisador
analyzer = MultiAudioAnalyzer(window_length_ms=500)

# Carregar arquivos de áudio
file_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
analyzer.load_audio_files(file_paths)

# Analisar arquivos
analyzer.analyze_all_files(k=5)  # k=None para detecção automática
```

**Parâmetros:**
- `window_length_ms` (float): Tamanho da janela de análise em milissegundos
- `k` (int, opcional): Número de clusters. Se None, determinado automaticamente

**Métodos:**
- `load_audio_files(file_paths)`: Carrega arquivos de áudio dos caminhos
- `analyze_all_files(k=None)`: Extrai características e realiza clustering
- `extract_features(audio_data)`: Extrai características de um único arquivo

#### MarkovTrackGenerator

Classe para gerar tracks musicais usando cadeias de Markov.

```python
from script11_Markov_Audio1GeraCompMult_GUI import MarkovTrackGenerator, SynthesisType, DurationMode

# Inicializar gerador
generator = MarkovTrackGenerator(analyzer)

# Definir modo de duração
generator.set_duration_mode(DurationMode.CLUSTER_MEAN)

# Gerar tracks
generator.generate_tracks(
    num_tracks=3,
    duration_seconds=60,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params={
        'grain_size': 0.1,
        'density': 100,
        'pitch_shift': 0
    }
)

# Exportar resultados
generator.export_tracks('pasta_de_saida')
```

**Parâmetros:**
- `num_tracks` (int): Número de tracks a gerar
- `duration_seconds` (float): Duração de cada track
- `synthesis_type` (SynthesisType): Tipo de síntese a usar
- `synthesis_params` (dict): Parâmetros específicos do método de síntese

### Exemplo: Fluxo Completo

```python
import numpy as np
from script11_Markov_Audio1GeraCompMult_GUI import (
    MultiAudioAnalyzer, MarkovTrackGenerator, 
    SynthesisType, DurationMode
)

# 1. Inicializar e analisar
analyzer = MultiAudioAnalyzer(window_length_ms=500)
analyzer.load_audio_files(['entrada1.wav', 'entrada2.wav'])
analyzer.analyze_all_files()

# 2. Gerar tracks
generator = MarkovTrackGenerator(analyzer)
generator.set_duration_mode(DurationMode.CLUSTER_MEAN)

generator.generate_tracks(
    num_tracks=5,
    duration_