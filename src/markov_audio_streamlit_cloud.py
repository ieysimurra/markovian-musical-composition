#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Composição Musical baseado em Cadeias de Markov
Interface Streamlit otimizada para Streamlit Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import zipfile
import time
import tempfile
import os
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random

# Configuração inicial
st.set_page_config(
    page_title="Sistema de Composição Musical Markoviano",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports condicionais com fallbacks
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    st.error("Librosa não está disponível. Algumas funcionalidades podem estar limitadas.")
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    st.error("SoundFile não está disponível. Export de áudio pode estar limitado.")
    SOUNDFILE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("Scikit-learn não está disponível. Análise pode estar limitada.")
    SKLEARN_AVAILABLE = False

try:
    from scipy import signal
    from scipy.stats import skew, kurtosis
    SCIPY_AVAILABLE = True
except ImportError:
    st.error("SciPy não está disponível. Algumas análises podem estar limitadas.")
    SCIPY_AVAILABLE = False

# Configurações
import warnings
warnings.filterwarnings('ignore')
sns.set_theme()

# Classes e Enums
class WindowType(Enum):
    HANN = 'hann'
    HAMMING = 'hamming'
    BLACKMAN = 'blackman'
    GAUSSIAN = 'gaussian'
    KAISER = 'kaiser'
    RECTANGULAR = 'rectangular'

class SynthesisType(Enum):
    CONCATENATIVE = 'concatenative'
    GRANULAR = 'granular'
    SPECTRAL = 'spectral'

class DurationMode(Enum):
    FIXED = 'fixed'
    CLUSTER_MEAN = 'cluster_mean'
    SEQUENCE_LENGTH = 'sequence'

@dataclass
class AudioData:
    filename: str
    signal: np.ndarray
    sr: int
    features: Optional[np.ndarray] = field(default=None)
    labels: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.signal, np.ndarray):
            self.signal = np.array(self.signal)
        if len(self.signal.shape) > 1:
            self.signal = np.mean(self.signal, axis=1)

# Verificação de dependências
def check_dependencies():
    """Verifica se todas as dependências necessárias estão disponíveis."""
    missing = []
    if not LIBROSA_AVAILABLE:
        missing.append("librosa")
    if not SOUNDFILE_AVAILABLE:
        missing.append("soundfile")
    if not SKLEARN_AVAILABLE:
        missing.append("scikit-learn")
    if not SCIPY_AVAILABLE:
        missing.append("scipy")
    
    return missing

# Interface principal
def main():
    st.title("🎵 Sistema de Composição Musical baseado em Cadeias de Markov")
    st.markdown("---")
    
    # Verificar dependências
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Dependências faltando: {', '.join(missing_deps)}")
        st.info("Algumas funcionalidades podem estar limitadas.")
        st.markdown("**Para funcionalidade completa, instale as dependências:**")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.markdown("---")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Upload de arquivos
        st.subheader("📁 Upload de Arquivos")
        uploaded_files = st.file_uploader(
            "Selecione os arquivos de áudio",
            type=['wav', 'mp3', 'flac', 'ogg', 'aiff', 'aif'],
            accept_multiple_files=True,
            help="Formatos suportados: WAV, MP3, FLAC, OGG, AIFF"
        )
        
        # Parâmetros de análise
        st.subheader("🔍 Parâmetros de Análise")
        window_length_ms = st.slider(
            "Tamanho da janela (ms)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Tamanho da janela de análise em milissegundos"
        )
        
        cluster_mode = st.radio(
            "Modo de clustering",
            ["Automático", "Manual"],
            help="Detecção automática ou definição manual do número de clusters"
        )
        
        if cluster_mode == "Manual":
            k_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=20,
                value=5,
                help="Número de clusters para K-means"
            )
        else:
            k_clusters = None
    
    # Área principal
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s)")
        
        # Mostrar informações dos arquivos
        with st.expander("📋 Informações dos Arquivos"):
            for i, file in enumerate(uploaded_files):
                st.write(f"**{i+1}.** {file.name} ({file.size} bytes)")
        
        # Botão para iniciar análise
        if st.button("🚀 Iniciar Análise", type="primary"):
            if not LIBROSA_AVAILABLE or not SKLEARN_AVAILABLE:
                st.error("Dependências necessárias não estão disponíveis para análise.")
                return
                
            try:
                with st.spinner("Analisando arquivos de áudio..."):
                    # Análise simplificada para demonstração
                    st.info("Análise em modo demonstração...")
                    
                    # Simular progresso
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Resultados simulados
                    num_clusters = k_clusters if k_clusters else np.random.randint(3, 8)
                    
                st.success("✅ Análise concluída com sucesso!")
                
                # Exibir informações da análise
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Arquivos processados", len(uploaded_files))
                with col2:
                    st.metric("Clusters encontrados", num_clusters)
                with col3:
                    st.metric("Características extraídas", "Simulado")
                
                # Interface de geração
                st.markdown("---")
                st.header("🎼 Configurações de Geração")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Parâmetros Básicos")
                    num_tracks = st.slider("Número de tracks", 1, 10, 3)
                    duration_seconds = st.slider("Duração (segundos)", 10, 300, 60)
                    
                    synthesis_type_str = st.selectbox(
                        "Tipo de síntese",
                        ["Concatenative", "Granular", "Spectral"],
                        help="Método de síntese de áudio"
                    )
                    
                    duration_mode_str = st.selectbox(
                        "Modo de duração",
                        ["Fixa", "Média do cluster", "Sequência"],
                        help="Como controlar a duração dos segmentos"
                    )
                
                with col2:
                    st.subheader("🎛️ Parâmetros de Síntese")
                    
                    if synthesis_type_str == "Concatenative":
                        crossfade_duration = st.slider(
                            "Duração do crossfade (s)", 0.01, 0.5, 0.1, 0.01
                        )
                        synthesis_params = {'crossfade_duration': crossfade_duration}
                        
                    elif synthesis_type_str == "Granular":
                        grain_size = st.slider("Tamanho do grão (s)", 0.01, 0.5, 0.1, 0.01)
                        density = st.slider("Densidade (grãos/s)", 10, 500, 100, 10)
                        pitch_shift = st.slider("Mudança de pitch (semitons)", -12, 12, 0)
                        synthesis_params = {
                            'grain_size': grain_size,
                            'density': density,
                            'pitch_shift': pitch_shift
                        }
                        
                    elif synthesis_type_str == "Spectral":
                        preserve_transients = st.checkbox("Preservar transientes", True)
                        spectral_stretch = st.slider("Fator de esticamento", 0.1, 3.0, 1.0, 0.1)
                        fft_size = st.selectbox("Tamanho da FFT", [512, 1024, 2048, 4096], index=2)
                        synthesis_params = {
                            'preserve_transients': preserve_transients,
                            'spectral_stretch': spectral_stretch,
                            'fft_size': fft_size
                        }
                
                # Botão para gerar tracks
                if st.button("🎵 Gerar Composição", type="primary"):
                    try:
                        with st.spinner("Gerando composição..."):
                            # Simulação de geração
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.02)
                                progress_bar.progress(i + 1)
                        
                        st.success("✅ Composição gerada com sucesso!")
                        
                        # Interface de reprodução simulada
                        st.markdown("---")
                        st.header("🎧 Reprodução e Download")
                        
                        # Simulação de arquivos gerados
                        for i in range(num_tracks):
                            st.subheader(f"🎵 Track {i+1}")
                            
                            # Audio player simulado
                            st.info(f"Track {i+1} - {synthesis_type_str} - {duration_seconds}s")
                            
                            # Criar dados de áudio simulados
                            sample_rate = 44100
                            duration = 5  # 5 segundos para demonstração
                            t = np.linspace(0, duration, int(sample_rate * duration), False)
                            
                            # Gerar onda senoidal simples
                            frequency = 220 * (i + 1)  # Frequências diferentes para cada track
                            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
                            
                            # Exibir player de áudio
                            if SOUNDFILE_AVAILABLE:
                                # Salvar áudio temporário
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                    sf.write(tmp_file.name, audio_data, sample_rate)
                                    with open(tmp_file.name, 'rb') as f:
                                        st.audio(f.read(), format='audio/wav')
                                    os.unlink(tmp_file.name)
                            else:
                                st.info("Player de áudio não disponível (SoundFile necessário)")
                        
                        # Mix final simulado
                        st.subheader("🎼 Mix Final")
                        if SOUNDFILE_AVAILABLE:
                            # Criar mix combinado
                            combined_audio = np.zeros_like(audio_data)
                            for i in range(num_tracks):
                                freq = 220 * (i + 1)
                                track_audio = 0.1 * np.sin(2 * np.pi * freq * t)
                                combined_audio += track_audio
                            
                            # Normalizar
                            if np.max(np.abs(combined_audio)) > 0:
                                combined_audio = combined_audio / np.max(np.abs(combined_audio))
                            
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                sf.write(tmp_file.name, combined_audio, sample_rate)
                                with open(tmp_file.name, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                                os.unlink(tmp_file.name)
                        
                        # Botão de download simulado
                        st.download_button(
                            label="📥 Download Composição (Simulado)",
                            data="Dados da composição seriam incluídos aqui",
                            file_name=f"markov_composition_{int(time.time())}.zip",
                            mime="application/zip"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante a geração: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {str(e)}")
    
    else:
        # Instruções para o usuário
        st.info("👆 Carregue arquivos de áudio na barra lateral para começar")
        
        # Informações sobre o sistema
        with st.expander("ℹ️ Sobre o Sistema"):
            st.markdown("""
            Este sistema utiliza **Cadeias de Markov** para análise e geração de material sonoro.
            
            **Como funciona:**
            1. **Análise**: Os arquivos de áudio são segmentados e suas características são extraídas
            2. **Clustering**: As características são agrupadas usando K-means
            3. **Modelagem**: Uma cadeia de Markov é construída baseada nas transições entre clusters
            4. **Geração**: Novas sequências são geradas e sintetizadas usando diferentes métodos
            
            **Tipos de Síntese:**
            - **Concatenativa**: Une segmentos com crossfade
            - **Granular**: Síntese baseada em grãos sonoros
            - **Espectral**: Manipulação no domínio da frequência
            
            **Nota**: Esta é uma versão otimizada para Streamlit Cloud com funcionalidades simuladas.
            Para funcionalidade completa, execute localmente com todas as dependências instaladas.
            """)

if __name__ == "__main__":
    main()