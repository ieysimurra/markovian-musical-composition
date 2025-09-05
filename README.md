# Sistema de Composição Musical Markoviano

Um sistema avançado de composição musical baseado em Cadeias de Markov com interface gráfica e suporte para múltiplos métodos de síntese de áudio.

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

### Opção 1: Interface Desktop (Local)

#### Pré-requisitos
```bash
pip install numpy librosa matplotlib seaborn scikit-learn scipy soundfile pandas pygame tkinter
```

#### Execução
```bash
python audioMarkov_gui_VFrame.py
```

A interface desktop oferece:
- Navegação por etapas guiada
- Player de áudio integrado
- Visualizações detalhadas
- Análises estatísticas completas
- Exportação organizada em pastas

### Opção 2: Interface Web (Streamlit)

#### Pré-requisitos
```bash
pip install streamlit numpy librosa matplotlib seaborn scikit-learn scipy soundfile pandas
```

#### Execução Local
```bash
streamlit run markov_audio_streamlit.py
```

#### Deploy no Streamlit Cloud
1. Faça fork deste repositório
2. Conecte sua conta GitHub ao [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy o arquivo `markov_audio_streamlit.py`

A interface web oferece:
- Interface responsiva e intuitiva
- Upload direto de arquivos
- Visualizações interativas
- Download de resultados em ZIP
- Compatibilidade com dispositivos móveis

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

- `audioMarkov_gui_VFrame.py`: Interface desktop completa com Tkinter
- `script11_Markov_Audio1GeraCompMult_GUI.py`: Engine principal de análise e síntese
- `markov_audio_streamlit.py`: Interface web com Streamlit

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

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:
1. Faça fork do projeto
2. Crie uma branch para sua funcionalidade
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🙏 Agradecimentos

- Biblioteca librosa para análise de áudio
- Scikit-learn para algoritmos de machine learning
- Streamlit para interface web moderna
- Tkinter para interface desktop robusta

## 📞 Contato

Para dúvidas, sugestões ou colaborações, abra uma issue no GitHub.

---

**Transforme seus arquivos de áudio em novas composições com o poder das Cadeias de Markov!**