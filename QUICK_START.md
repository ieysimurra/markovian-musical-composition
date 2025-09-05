# Quick Start Guide / Guia de Início Rápido

## English

### 🚀 Get Started in 5 Minutes

#### Option A: Desktop Interface (Recommended for Beginners)

1. **Install Python 3.8+** and clone the repository:
```bash
git clone https://github.com/yourusername/markovian-musical-composition.git
cd markovian-musical-composition
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create test audio** (optional):
```bash
python scripts/create_test_audio.py
```

4. **Run the desktop application**:
```bash
python src/audioMarkov_gui_VFrame.py
```

5. **Follow the guided workflow**:
   - **Step 1**: Select audio files (WAV, MP3, FLAC, etc.)
   - **Step 2**: Configure analysis parameters
   - **Step 3**: Set generation parameters and synthesis method
   - **Step 4**: Play and download your generated compositions!

#### Option B: Web Interface (Streamlit)

1. **Install Streamlit dependencies**:
```bash
pip install -r requirements-streamlit.txt
```

2. **Run Streamlit app**:
```bash
streamlit run src/markov_audio_streamlit.py
```

3. **Open browser** at `http://localhost:8501`

4. **Upload files and generate** music directly in your browser!

#### Option C: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub and deploy `src/markov_audio_streamlit.py`
4. Share your app with the world!

### 🎵 First Composition

Try these settings for your first composition:

- **Window Size**: 500ms
- **Clusters**: Auto-detect
- **Tracks**: 3
- **Duration**: 60 seconds
- **Synthesis**: Concatenative
- **Mode**: Fixed duration

### 📁 What You'll Get

Your output folder will contain:
```
output_multitrack_[id]/
├── track_1/audio.wav          # Individual tracks
├── track_2/audio.wav
├── track_3/audio.wav
├── final_mix.wav              # Mixed composition
└── analysis/                  # Detailed analysis data
```

### 🔧 Troubleshooting

**Common Issues:**
- **Import errors**: Run `pip install -r requirements.txt`
- **Audio not loading**: Check file format (WAV recommended)
- **Slow processing**: Reduce window size or use shorter audio files

---

## Português

### 🚀 Comece em 5 Minutos

#### Opção A: Interface Desktop (Recomendado para Iniciantes)

1. **Instale Python 3.8+** e clone o repositório:
```bash
git clone https://github.com/yourusername/markovian-musical-composition.git
cd markovian-musical-composition
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Crie áudio de teste** (opcional):
```bash
python scripts/create_test_audio.py
```

4. **Execute a aplicação desktop**:
```bash
python src/audioMarkov_gui_VFrame.py
```

5. **Siga o fluxo guiado**:
   - **Etapa 1**: Selecione arquivos de áudio (WAV, MP3, FLAC, etc.)
   - **Etapa 2**: Configure parâmetros de análise
   - **Etapa 3**: Defina parâmetros de geração e método de síntese
   - **Etapa 4**: Reproduza e baixe suas composições geradas!

#### Opção B: Interface Web (Streamlit)

1. **Instale dependências do Streamlit**:
```bash
pip install -r requirements-streamlit.txt
```

2. **Execute o app Streamlit**:
```bash
streamlit run src/markov_audio_streamlit.py
```

3. **Abra o navegador** em `http://localhost:8501`

4. **Faça upload de arquivos e gere** música diretamente no navegador!

#### Opção C: Deploy no Streamlit Cloud

1. Faça fork deste repositório
2. Vá para [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conecte o GitHub e faça deploy de `src/markov_audio_streamlit.py`
4. Compartilhe seu app com o mundo!

### 🎵 Primeira Composição

Tente estas configurações para sua primeira composição:

- **Tamanho da Janela**: 500ms
- **Clusters**: Detecção automática
- **Tracks**: 3
- **Duração**: 60 segundos
- **Síntese**: Concatenativa
- **Modo**: Duração fixa

### 📁 O Que Você Receberá

Sua pasta de saída conterá:
```
output_multitrack_[id]/
├── track_1/audio.wav          # Tracks individuais
├── track_2/audio.wav
├── track_3/audio.wav
├── final_mix.wav              # Composição mixada
└── analysis/                  # Dados de análise detalhados
```

### 🔧 Solução de Problemas

**Problemas Comuns:**
- **Erros de importação**: Execute `pip install -r requirements.txt`
- **Áudio não carrega**: Verifique formato do arquivo (WAV recomendado)
- **Processamento lento**: Reduza tamanho da janela ou use arquivos mais curtos

---

## 🎯 Next Steps / Próximos Passos

### Explore Advanced Features / Explore Funcionalidades Avançadas

1. **Try Different Synthesis Methods**:
   - **Granular**: For textural, ambient compositions
   - **Spectral**: For harmonic transformations
   - **Concatenative**: For rhythmic, percussive material

2. **Experiment with Parameters**:
   - **Window Size**: Smaller = more detail, Larger = broader patterns
   - **Duration Modes**: Try cluster-based or sequence-based timing
   - **Synthesis Parameters**: Adjust grain size, pitch shift, crossfade

3. **Batch Processing**:
```bash
python scripts/batch_process.py input_folder/ -c config.json
```

4. **API Usage**:
```python
from src.script11_Markov_Audio1GeraCompMult_GUI import *

analyzer = MultiAudioAnalyzer(window_length_ms=500)
analyzer.load_audio_files(['file1.wav', 'file2.wav'])
analyzer.analyze_all_files()

generator = MarkovTrackGenerator(analyzer)
generator.generate_tracks(num_tracks=3, duration_seconds=60)
generator.export_tracks('output')
```

### Get Help / Obter Ajuda

- 📖 **Read the docs**: Check `API_DOCUMENTATION.md` and `EXAMPLES.md`
- 🐛 **Report bugs**: Open an issue on GitHub
- 💡 **Feature requests**: Use the GitHub issues template
- 🤝 **Contribute**: See `CONTRIBUTING.md` for guidelines

---

**Happy composing! / Boa composição!** 🎼