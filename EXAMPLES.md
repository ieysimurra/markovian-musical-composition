# Examples / Exemplos

## English

### Basic Usage Examples

#### Example 1: Simple Analysis and Generation

```python
from script11_Markov_Audio1GeraCompMult_GUI import MultiAudioAnalyzer, MarkovTrackGenerator, SynthesisType

# Analyze audio files
analyzer = MultiAudioAnalyzer(window_length_ms=500)
analyzer.load_audio_files(['drums.wav', 'bass.wav', 'melody.wav'])
analyzer.analyze_all_files(k=4)

# Generate composition
generator = MarkovTrackGenerator(analyzer)
generator.generate_tracks(
    num_tracks=3,
    duration_seconds=60,
    synthesis_type=SynthesisType.CONCATENATIVE
)

# Export results
generator.export_tracks('my_composition')
```

#### Example 2: Granular Synthesis with Custom Parameters

```python
from script11_Markov_Audio1GeraCompMult_GUI import *

# Setup analyzer
analyzer = MultiAudioAnalyzer(window_length_ms=300)
analyzer.load_audio_files(['texture1.wav', 'texture2.wav'])
analyzer.analyze_all_files()

# Configure granular synthesis
granular_params = {
    'grain_size': 0.05,        # Small grains for dense texture
    'density': 200,            # High density
    'pitch_shift': 3,          # Shift up 3 semitones
    'position_jitter': 0.3,    # High position variation
    'duration_jitter': 0.2     # Moderate duration variation
}

# Generate ambient tracks
generator = MarkovTrackGenerator(analyzer)
generator.generate_tracks(
    num_tracks=2,
    duration_seconds=120,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params=granular_params
)

generator.export_tracks('ambient_textures')
```

#### Example 3: Spectral Synthesis for Harmonic Content

```python
# Analyze harmonic content
analyzer = MultiAudioAnalyzer(window_length_ms=1000)  # Longer window for harmonics
analyzer.load_audio_files(['piano.wav', 'strings.wav', 'choir.wav'])
analyzer.analyze_all_files(k=6)

# Spectral synthesis parameters
spectral_params = {
    'preserve_transients': True,
    'spectral_stretch': 1.5,   # Stretch spectrum
    'fft_size': 4096          # High resolution
}

# Generate with cluster-based durations
generator = MarkovTrackGenerator(analyzer)
generator.set_duration_mode(DurationMode.CLUSTER_MEAN)

generator.generate_tracks(
    num_tracks=4,
    duration_seconds=180,
    synthesis_type=SynthesisType.SPECTRAL,
    synthesis_params=spectral_params
)

generator.export_tracks('harmonic_variations')
```

#### Example 4: Batch Processing Multiple Compositions

```python
import os
from pathlib import Path

def process_audio_folder(input_folder, output_base):
    """Process all audio files in a folder"""
    
    # Get all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(input_folder).glob(ext))
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return
    
    # Analyze
    analyzer = MultiAudioAnalyzer(window_length_ms=400)
    analyzer.load_audio_files([str(f) for f in audio_files])
    analyzer.analyze_all_files()
    
    # Generate multiple versions with different synthesis
    synthesis_configs = [
        (SynthesisType.CONCATENATIVE, {'crossfade_duration': 0.05}),
        (SynthesisType.GRANULAR, {'grain_size': 0.1, 'density': 80}),
        (SynthesisType.SPECTRAL, {'spectral_stretch': 1.2})
    ]
    
    for i, (synth_type, params) in enumerate(synthesis_configs):
        generator = MarkovTrackGenerator(analyzer)
        generator.generate_tracks(
            num_tracks=3,
            duration_seconds=90,
            synthesis_type=synth_type,
            synthesis_params=params
        )
        
        output_folder = f"{output_base}_{synth_type.value}_{i+1}"
        generator.export_tracks(output_folder)
        print(f"Generated composition in {output_folder}")

# Usage
process_audio_folder('input_samples', 'batch_output')
```

#### Example 5: Real-time Parameter Exploration

```python
def explore_parameters():
    """Explore different parameter combinations"""
    
    # Base setup
    analyzer = MultiAudioAnalyzer(window_length_ms=500)
    analyzer.load_audio_files(['sample1.wav', 'sample2.wav'])
    analyzer.analyze_all_files()
    
    # Parameter ranges to explore
    grain_sizes = [0.05, 0.1, 0.2]
    densities = [50, 100, 150]
    pitch_shifts = [-2, 0, 2, 5]
    
    results = []
    
    for grain_size in grain_sizes:
        for density in densities:
            for pitch_shift in pitch_shifts:
                params = {
                    'grain_size': grain_size,
                    'density': density,
                    'pitch_shift': pitch_shift,
                    'position_jitter': 0.1,
                    'duration_jitter': 0.1
                }
                
                generator = MarkovTrackGenerator(analyzer)
                generator.generate_tracks(
                    num_tracks=1,
                    duration_seconds=30,  # Short for quick exploration
                    synthesis_type=SynthesisType.GRANULAR,
                    synthesis_params=params
                )
                
                # Store result info
                result_info = {
                    'params': params,
                    'output_folder': f'exploration_g{grain_size}_d{density}_p{pitch_shift}'
                }
                results.append(result_info)
                
                generator.export_tracks(result_info['output_folder'])
                print(f"Generated: {result_info['output_folder']}")
    
    return results

# Run exploration
exploration_results = explore_parameters()
```

### Advanced Usage Patterns

#### Custom Feature Extraction

```python
class CustomAnalyzer(MultiAudioAnalyzer):
    def extract_features(self, audio_data):
        """Override to add custom features"""
        
        # Get standard features
        standard_features = super().extract_features(audio_data)
        
        # Add custom features (example: spectral contrast)
        hop_length = int(audio_data.sr * self.window_length_ms / 2000)
        contrast = librosa.feature.spectral_contrast(
            y=audio_data.signal, 
            sr=audio_data.sr,
            hop_length=hop_length
        ).mean(axis=1)
        
        # Combine features
        if len(standard_features) > 0:
            enhanced_features = []
            for i, frame_features in enumerate(standard_features):
                enhanced_frame = np.concatenate([frame_features, contrast])
                enhanced_features.append(enhanced_frame)
            return np.array(enhanced_features)
        
        return standard_features

# Use custom analyzer
custom_analyzer = CustomAnalyzer(window_length_ms=500)
# ... rest of workflow
```

---

## Português

### Exemplos de Uso Básico

#### Exemplo 1: Análise e Geração Simples

```python
from script11_Markov_Audio1GeraCompMult_GUI import MultiAudioAnalyzer, MarkovTrackGenerator, SynthesisType

# Analisar arquivos de áudio
analyzer = MultiAudioAnalyzer(window_length_ms=500)
analyzer.load_audio_files(['bateria.wav', 'baixo.wav', 'melodia.wav'])
analyzer.analyze_all_files(k=4)

# Gerar composição
generator = MarkovTrackGenerator(analyzer)
generator.generate_tracks(
    num_tracks=3,
    duration_seconds=60,
    synthesis_type=SynthesisType.CONCATENATIVE
)

# Exportar resultados
generator.export_tracks('minha_composicao')
```

#### Exemplo 2: Síntese Granular com Parâmetros Personalizados

```python
# Configurar síntese granular
parametros_granular = {
    'grain_size': 0.05,        # Grãos pequenos para textura densa
    'density': 200,            # Alta densidade
    'pitch_shift': 3,          # Elevar 3 semitons
    'position_jitter': 0.3,    # Alta variação de posição
    'duration_jitter': 0.2     # Variação moderada de duração
}

# Gerar tracks ambientes
generator = MarkovTrackGenerator(analyzer)
generator.generate_tracks(
    num_tracks=2,
    duration_seconds=120,
    synthesis_type=SynthesisType.GRANULAR,
    synthesis_params=parametros_granular
)

generator.export_tracks('texturas_ambiente')
```

#### Exemplo 3: Processamento em Lote

```python
def processar_pasta_audio(pasta_entrada, saida_base):
    """Processar todos os arquivos de áudio em uma pasta"""
    
    # Obter todos os arquivos de áudio
    arquivos_audio = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        arquivos_audio.extend(Path(pasta_entrada).glob(ext))
    
    if not arquivos_audio:
        print(f"Nenhum arquivo de áudio encontrado em {pasta_entrada}")
        return
    
    # Analisar
    analyzer = MultiAudioAnalyzer(window_length_ms=400)
    analyzer.load_audio_files([str(f) for f in arquivos_audio])
    analyzer.analyze_all_files()
    
    # Gerar múltiplas versões com diferentes sínteses
    configuracoes_sintese = [
        (SynthesisType.CONCATENATIVE, {'crossfade_duration': 0.05}),
        (SynthesisType.GRANULAR, {'grain_size': 0.1, 'density': 80}),
        (SynthesisType.SPECTRAL, {'spectral_stretch': 1.2})
    ]
    
    for i, (tipo_sintese, params) in enumerate(configuracoes_sintese):
        generator = MarkovTrackGenerator(analyzer)
        generator.generate_tracks(
            num_tracks=3,
            duration_seconds=90,
            synthesis_type=tipo_sintese,
            synthesis_params=params
        )
        
        pasta_saida = f"{saida_base}_{tipo_sintese.value}_{i+1}"
        generator.export_tracks(pasta_saida)
        print(f"Composição gerada em {pasta_saida}")

# Uso
processar_pasta_audio('amostras_entrada', 'saida_lote')
```

### Casos de Uso Específicos

#### Música Ambiente/Textural
```python
# Parâmetros para texturas ambientes
ambient_params = {
    'grain_size': 0.2,
    'density': 30,
    'pitch_shift': 0,
    'position_jitter': 0.5,
    'duration_jitter': 0.3
}
```

#### Música Rítmica/Percussiva
```python
# Parâmetros para material rítmico
rhythmic_params = {
    'crossfade_duration': 0.001,  # Crossfade muito curto
}
```

#### Exploração Harmônica
```python
# Parâmetros para exploração harmônica
harmonic_params = {
    'preserve_transients': False,
    'spectral_stretch': 0.8,
    'fft_size': 4096
}
```