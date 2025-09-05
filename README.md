# Sistema de Composição Musical Markoviano

Um sistema avançado de composição musical baseado em Cadeias de Markov com interface gráfica e suporte para múltiplos métodos de síntese de áudio.

## 🌐 Aplicação Online / Live Application

**🚀 Acesse a aplicação web aqui / Access the web app here:**  
**https://markovian-musical-composition-gui.streamlit.app/**

*A aplicação roda diretamente no navegador - não é necessário instalar nada! / The app runs directly in your browser - no installation needed!*

---

## 🎵 Sobre o Projeto

Este projeto implementa um sistema inteligente de composição musical que utiliza **Cadeias de Markov** para analisar arquivos de áudio existentes e gerar novo material sonoro. O sistema oferece três métodos distintos de síntese de áudio e duas interfaces de usuário para máxima flexibilidade.

### Características Principais

- **Análise Automática**: Extração de características espectrais e clustering K-means
- **Múltiplos Métodos de Síntese**:
  - Concatenativa com crossfade
  - Granular com controle de densidade e pitch
  - Espectral com manipulação de frequências
- **Modos de Duração Flexíveis**:
  - Duração fixa baseada na janela de análise
  - Duração média por cluster
  - Duração baseada em sequências consecutivas
- **Duas Interfaces**:
  - Interface desktop com Tkinter (local)
  - Interface web com Streamlit (cloud)

## 🚀 Instalação e Uso

### Opção 1: Interface Web (Recomendado - Sem Instalação)

**Simplesmente acesse:** https://markovian-musical-composition-gui.streamlit.app/

1. Faça upload dos seus arquivos de áudio (WAV, MP3, FLAC, OGG, AIFF)
2. Configure os parâmetros de análise
3. Escolha o método de síntese e parâmetros
4. Gere e baixe suas composições!

### Opção 2: Interface Desktop (Local)

#### Pré-requisitos
```bash
pip install numpy librosa matplotlib seaborn scikit-learn scipy soundfile pandas pygame tkinter
```

#### Execução
```bash
git clone https://github.com/yourusername/markovian-musical-composition.git
cd markovian-musical-composition
python src/audioMarkov_gui_VFrame.py
```

A interface desktop oferece:
- Navegação por etapas guiada
- Player de áudio integrado
- Visualizações detalhadas
- Análises estatísticas completas
- Exportação organizada em pastas

### Opção 3: Interface Streamlit Local

#### Pré-requisitos
```bash
pip install -r requirements-streamlit.txt
```

#### Execução Local
```bash
streamlit run src/markov_audio_streamlit.py
```

## 📋 Como Usar

### Fluxo de Trabalho

1. **Seleção de Arquivos**: Carregue arquivos de áudio (WAV, MP3, FLAC, OGG, AIFF)
2. **Configuração de Análise**: 
   - Defina o tamanho da janela (100-1000ms)
   - Escolha o número de clusters (automático ou manual)
3. **Análise**: O sistema extrai características e cria clusters
4. **Configuração de Geração**:
   - Número de tracks desejadas
   - Duração total
   - Método de síntese
   - Modo de duração
5. **Geração**: Criação das composições baseadas na cadeia de Markov
6. **Reprodução e Download**: Teste e baixe os resultados

### Parâmetros de Síntese

#### Síntese Concatenativa
- **Crossfade Duration**: Duração da transição entre segmentos (0.01-0.5s)

#### Síntese Granular
- **Grain Size**: Tamanho dos grãos sonoros (0.01-0.5s)
- **Density**: Número de grãos por segundo (10-500)
- **Pitch Shift**: Alteração de altura em semitons (-12 a +12)
- **Position/Duration Jitter**: Variação aleatória (0-0.5)

#### Síntese Espectral
- **Preserve Transients**: Mantém características transitórias
- **Spectral Stretch**: Fator de esticamento espectral (0.1-3.0)
- **FFT Size**: Tamanho da janela FFT (512, 1024, 2048, 4096)

## 🔧 Estrutura do Código

### Arquivos Principais

- `src/audioMarkov_gui_VFrame.py`: Interface desktop completa com Tkinter
- `src/script11_Markov_Audio1GeraCompMult_GUI.py`: Engine principal de análise e síntese
- `src/markov_audio_streamlit.py`: Interface web com Streamlit

### Classes Principais

- `MultiAudioAnalyzer`: Análise de múltiplos arquivos de áudio
- `MarkovTrackGenerator`: Geração de tracks usando cadeias de Markov
- `AudioSynthesizer`: Classes base para síntese
- `ConcatenativeSynthesizer`: Síntese por concatenação
- `GranularSynthesizer`: Síntese granular
- `SpectralSynthesizer`: Síntese espectral

## 📊 Funcionalidades Técnicas

### Análise de Áudio
- Extração de MFCCs (13 coeficientes)
- Centroide espectral
- Largura de banda espectral
- Rolloff espectral
- RMS Energy
- Zero Crossing Rate

### Clustering
- K-means com detecção automática do número ótimo de clusters
- Normalização de características
- Análise de silhueta para validação

### Modelagem Markoviana
- Cálculo de probabilidades iniciais
- Matriz de transição baseada em dados reais
- Geração probabilística de sequências

## 📁 Estrutura de Saída

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

## 🎯 Casos de Uso

- **Compositores**: Criação de material musical experimental
- **Sound Designers**: Geração de texturas sonoras únicas
- **Pesquisadores**: Estudo de estruturas musicais e síntese
- **Educadores**: Demonstração de conceitos de análise musical
- **Artistas**: Exploração de novas possibilidades criativas

## 🔬 Metodologia Científica

O sistema baseia-se em princípios sólidos de:
- **Processamento Digital de Sinais**: Análise espectral avançada
- **Aprendizado de Máquina**: Clustering não-supervisionado
- **Teoria da Informação**: Modelagem probabilística
- **Síntese de Áudio**: Múltiplas abordagens de geração sonora

## 📚 Documentação

- 📖 **[Guia de Início Rápido](QUICK_START.md)** - Comece em 5 minutos
- 📋 **[Guia de Instalação](INSTALLATION.md)** - Instruções detalhadas
- 🔧 **[Documentação da API](API_DOCUMENTATION.md)** - Referência completa
- 💡 **[Exemplos de Uso](EXAMPLES.md)** - Casos práticos
- 🌐 **[Links de Deploy](DEPLOYMENT.md)** - Acesso à aplicação online

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:
1. Faça fork do projeto
2. Crie uma branch para sua funcionalidade
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes detalhadas.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Biblioteca librosa para análise de áudio
- Scikit-learn para algoritmos de machine learning
- Streamlit para interface web moderna
- Tkinter para interface desktop robusta

## 📞 Contato

Para dúvidas, sugestões ou colaborações:
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/markovian-musical-composition/issues)
- 📧 **Email**: contact@markovian-music.org
- 🌐 **Website**: https://markovian-musical-composition-gui.streamlit.app/

---

**🎼 Transforme seus arquivos de áudio em novas composições com o poder das Cadeias de Markov!**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://markovian-musical-composition-gui.streamlit.app/)
