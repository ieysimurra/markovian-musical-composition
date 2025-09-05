# Installation Guide / Guia de Instalação

## English

### System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM
- Audio file support (WAV, MP3, FLAC, OGG, AIFF)
- For desktop interface: Display with minimum 1000x700 resolution

### Desktop Installation (Local)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/markovian-musical-composition.git
cd markovian-musical-composition
```

2. **Create virtual environment (recommended):**
```bash
python -m venv markov_env
# Windows
markov_env\Scripts\activate
# macOS/Linux
source markov_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the desktop application:**
```bash
python audioMarkov_gui_VFrame.py
```

### Web Interface (Streamlit)

#### Local Development

1. **Install Streamlit dependencies:**
```bash
pip install -r requirements-streamlit.txt
```

2. **Run Streamlit app:**
```bash
streamlit run markov_audio_streamlit.py
```

3. **Access the application:**
Open your browser at `http://localhost:8501`

#### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your forked repository
5. Set main file path: `markov_audio_streamlit.py`
6. Deploy!

### Troubleshooting

#### Common Issues

**Import Error: No module named 'librosa'**
```bash
pip install librosa
```

**Import Error: No module named 'tkinter'**
- On Ubuntu/Debian: `sudo apt-get install python3-tk`
- On macOS: Usually included with Python
- On Windows: Usually included with Python

**Audio File Not Loading**
- Ensure file format is supported (WAV, MP3, FLAC, OGG, AIFF)
- Check file is not corrupted
- Try converting to WAV format

**Memory Error with Large Files**
- Reduce window size in analysis parameters
- Use shorter audio files
- Increase system RAM if possible

---

## Português

### Requisitos do Sistema

- Python 3.8 ou superior
- Mínimo 4GB RAM
- Suporte a arquivos de áudio (WAV, MP3, FLAC, OGG, AIFF)
- Para interface desktop: Display com resolução mínima 1000x700

### Instalação Desktop (Local)

1. **Clone o repositório:**
```bash
git clone https://github.com/yourusername/markovian-musical-composition.git
cd markovian-musical-composition
```

2. **Crie ambiente virtual (recomendado):**
```bash
python -m venv markov_env
# Windows
markov_env\Scripts\activate
# macOS/Linux
source markov_env/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação desktop:**
```bash
python audioMarkov_gui_VFrame.py
```

### Interface Web (Streamlit)

#### Desenvolvimento Local

1. **Instale dependências do Streamlit:**
```bash
pip install -r requirements-streamlit.txt
```

2. **Execute a aplicação Streamlit:**
```bash
streamlit run markov_audio_streamlit.py
```

3. **Acesse a aplicação:**
Abra seu navegador em `http://localhost:8501`

#### Deploy no Streamlit Cloud

1. Faça fork deste repositório para sua conta GitHub
2. Vá para [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecte sua conta GitHub
4. Selecione seu repositório fork
5. Defina o caminho do arquivo principal: `markov_audio_streamlit.py`
6. Deploy!

### Solução de Problemas

#### Problemas Comuns

**Erro de Importação: No module named 'librosa'**
```bash
pip install librosa
```

**Erro de Importação: No module named 'tkinter'**
- No Ubuntu/Debian: `sudo apt-get install python3-tk`
- No macOS: Geralmente incluído com Python
- No Windows: Geralmente incluído com Python

**Arquivo de Áudio Não Carrega**
- Certifique-se de que o formato é suportado (WAV, MP3, FLAC, OGG, AIFF)
- Verifique se o arquivo não está corrompido
- Tente converter para formato WAV

**Erro de Memória com Arquivos Grandes**
- Reduza o tamanho da janela nos parâmetros de análise
- Use arquivos de áudio mais curtos
- Aumente a RAM do sistema se possível