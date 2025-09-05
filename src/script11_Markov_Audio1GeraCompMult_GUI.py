# -*- coding: utf-8 -*-
"""
Sistema de Composição Musical baseado em Cadeias de Markov
com suporte a múltiplos arquivos de entrada e geração de múltiplas tracks
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
from scipy import signal
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional, Union, Tuple
import random
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import soundfile as sf
import os
from dataclasses import dataclass, field
import pandas as pd
import uuid
from enum import Enum
import json
import warnings

# Configuração do estilo de visualização
sns.set_theme()  # Configura o estilo do seaborn

# Classes principais
@dataclass
class WindowType(Enum):
    """Tipos de janelamento disponíveis."""
    HANN = 'hann'
    HAMMING = 'hamming'
    BLACKMAN = 'blackman'
    GAUSSIAN = 'gaussian'
    KAISER = 'kaiser'
    RECTANGULAR = 'rectangular'

class SynthesisType(Enum):
    """Tipos de síntese disponíveis."""
    CONCATENATIVE = 'concatenative'
    GRANULAR = 'granular'
    SPECTRAL = 'spectral'

class DurationMode(Enum):
    """Modos de controle de duração para síntese."""
    FIXED = 'fixed'                # Duração fixa baseada em window_length_ms
    CLUSTER_MEAN = 'cluster_mean'  # Duração baseada na média de cada cluster
    SEQUENCE_LENGTH = 'sequence'   # Duração baseada no comprimento de sequências consecutivas


class AudioSynthesizer:
    """Classe base para síntese de áudio."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.window_type = WindowType.HANN
        self.window_size = 2048
        self.hop_size = 512
        
    def get_window(self, size: int) -> np.ndarray:
        """Retorna a janela especificada."""
        if self.window_type == WindowType.GAUSSIAN:
            return signal.gaussian(size, std=size/6.0)
        elif self.window_type == WindowType.KAISER:
            return signal.kaiser(size, beta=14)
        else:
            return signal.get_window(self.window_type.value, size)

class ConcatenativeSynthesizer(AudioSynthesizer):
    """Sintetizador por concatenação com crossfade."""
    
    def __init__(self, sr: int = 44100, crossfade_duration: float = 0.1):
        super().__init__(sr)
        self.crossfade_duration = crossfade_duration
        
    def synthesize(self, segments: List[np.ndarray], 
                  crossfade_duration: Optional[float] = None) -> np.ndarray:
        """Sintetiza áudio por concatenação com crossfade."""
        if crossfade_duration is None:
            crossfade_duration = self.crossfade_duration
            
        fade_length = int(crossfade_duration * self.sr)
        
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
            
        result = segments[0]
        for i in range(1, len(segments)):
            if len(result) < fade_length or len(segments[i]) < fade_length:
                result = np.concatenate([result, segments[i]])
                continue
                
            fade_out = np.linspace(1.0, 0.0, fade_length)
            fade_in = np.linspace(0.0, 1.0, fade_length)
            
            result_end = result[-fade_length:] * fade_out
            next_start = segments[i][:fade_length] * fade_in
            
            result = np.concatenate([
                result[:-fade_length],
                result_end + next_start,
                segments[i][fade_length:]
            ])
            
        return result

class GranularSynthesizer(AudioSynthesizer):
    """Sintetizador granular com controle de densidade e pitch."""
    
    def __init__(self, sr: int = 44100):
        super().__init__(sr)
        self.grain_size = 0.1  # segundos
        self.density = 100  # grãos por segundo
        self.pitch_shift = 0  # semitons
        self.position_jitter = 0.1  # variação da posição (0-1)
        self.duration_jitter = 0.1  # variação da duração (0-1)
        
    def synthesize(self, source: np.ndarray, duration: float) -> np.ndarray:
        """Sintetiza áudio usando síntese granular."""
        grain_samples = int(self.grain_size * self.sr)
        num_grains = int(duration * self.density)
        output = np.zeros(int(duration * self.sr))
        
        window = self.get_window(grain_samples)
        
        for i in range(num_grains):
            # Posição do grão com jitter
            position = (i / self.density) + np.random.uniform(
                -self.position_jitter, 
                self.position_jitter
            ) * self.grain_size
            
            # Duração do grão com jitter
            grain_duration = self.grain_size * (1 + np.random.uniform(
                -self.duration_jitter,
                self.duration_jitter
            ))
            
            grain_size = int(grain_duration * self.sr)
            if grain_size != len(window):
                window = self.get_window(grain_size)
            
            # Extrai e processa o grão
            start_pos = int((position % (len(source) / self.sr)) * self.sr)
            if start_pos + grain_size > len(source):
                continue
                
            grain = source[start_pos:start_pos + grain_size]
            
            # Aplica pitch shift se necessário
            if self.pitch_shift != 0:
                grain = librosa.effects.pitch_shift(
                    grain,
                    sr=self.sr,
                    n_steps=self.pitch_shift
                )
            
            # Aplica janela
            grain = grain * window
            
            # Adiciona ao output
            out_start = int(position * self.sr)
            if out_start + len(grain) > len(output):
                continue
            output[out_start:out_start + len(grain)] += grain
            
        return output

class SpectralSynthesizer(AudioSynthesizer):
    """Sintetizador baseado em características espectrais."""
    
    def __init__(self, sr: int = 44100):
        super().__init__(sr)
        self.fft_size = 2048
        self.preserve_transients = True
        self.spectral_stretch = 1.0
        
    def synthesize(self, source: np.ndarray, target_features: Dict) -> np.ndarray:
        """Sintetiza áudio usando características espectrais."""
        # Análise
        D = librosa.stft(source, n_fft=self.fft_size, 
                        hop_length=self.hop_size,
                        window=self.get_window(self.fft_size))
        
        # Modifica o espectro baseado nas características alvo
        if 'spectral_centroid' in target_features:
            self._adjust_centroid(D, target_features['spectral_centroid'])
        
        if 'spectral_bandwidth' in target_features:
            self._adjust_bandwidth(D, target_features['spectral_bandwidth'])
            
        if self.preserve_transients:
            D = self._preserve_transient_features(D)
            
        if self.spectral_stretch != 1.0:
            D = self._stretch_spectrum(D)
            
        # Síntese
        return librosa.istft(D, hop_length=self.hop_size,
                           window=self.get_window(self.fft_size))
    
    def _adjust_centroid(self, D: np.ndarray, target_centroid: float) -> None:
        """Ajusta o centroide espectral."""
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        current_centroid = np.sum(frequencies[:, np.newaxis] * np.abs(D)) / np.sum(np.abs(D))
        
        ratio = target_centroid / current_centroid
        shift = np.exp(np.log(ratio) * frequencies[:, np.newaxis])
        D *= shift
    
    def _adjust_bandwidth(self, D: np.ndarray, target_bandwidth: float) -> None:
        """Ajusta a largura de banda espectral."""
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        centroid = np.sum(frequencies[:, np.newaxis] * np.abs(D)) / np.sum(np.abs(D))
        
        current_bandwidth = np.sqrt(
            np.sum(((frequencies[:, np.newaxis] - centroid) ** 2) * np.abs(D)) / 
            np.sum(np.abs(D))
        )
        
        ratio = target_bandwidth / current_bandwidth
        D *= np.exp(-((frequencies[:, np.newaxis] - centroid) ** 2) * (1 - ratio))
    
    def _preserve_transient_features(self, D: np.ndarray) -> np.ndarray:
        """Preserva características transitórias."""
        onset_env = librosa.onset.onset_strength(S=np.abs(D), sr=self.sr)
        onset_mask = librosa.util.normalize(onset_env)
        return D * onset_mask
    
    def _stretch_spectrum(self, D: np.ndarray) -> np.ndarray:
        """Estica ou comprime o espectro."""
        n_freqs = D.shape[0]
        new_freqs = np.interp(
            np.linspace(0, n_freqs - 1, int(n_freqs * self.spectral_stretch)),
            np.arange(n_freqs),
            np.arange(n_freqs)
        )
        return librosa.phase_vocoder(D, self.spectral_stretch)

class AudioProcessor:
    """Classe para processamento de áudio em lote."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.concatenative = ConcatenativeSynthesizer(sr)
        self.granular = GranularSynthesizer(sr)
        self.spectral = SpectralSynthesizer(sr)
        
    def process_batch(
        self,
        sources: List[np.ndarray],
        synthesis_type: SynthesisType,
        duration: float,
        **kwargs
    ) -> np.ndarray:
        """Processa um lote de arquivos de áudio."""
        if synthesis_type == SynthesisType.CONCATENATIVE:
            return self.concatenative.synthesize(sources, **kwargs)
        elif synthesis_type == SynthesisType.GRANULAR:
            return self.granular.synthesize(np.concatenate(sources), duration)
        elif synthesis_type == SynthesisType.SPECTRAL:
            # Combina características espectrais
            combined_features = self._combine_spectral_features(sources)
            return self.spectral.synthesize(np.concatenate(sources), combined_features)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normaliza o áudio."""
        return audio / (np.max(np.abs(audio)) + 1e-10)
    
    def adjust_duration(
        self,
        audio: np.ndarray,
        target_duration: float,
        method: str = 'stretch'
    ) -> np.ndarray:
        """Ajusta a duração do áudio."""
        current_duration = len(audio) / self.sr
        
        if method == 'stretch':
            return librosa.effects.time_stretch(audio, rate=current_duration/target_duration)
        elif method == 'loop':
            num_repeats = int(np.ceil(target_duration / current_duration))
            return np.tile(audio, num_repeats)[:int(target_duration * self.sr)]
        else:
            raise ValueError(f"Método de ajuste de duração desconhecido: {method}")
    
    def _combine_spectral_features(
        self,
        sources: List[np.ndarray]
    ) -> Dict[str, float]:
        """Combina características espectrais de múltiplas fontes."""
        features = {}
        
        for source in sources:
            spec_cent = librosa.feature.spectral_centroid(y=source, sr=self.sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=source, sr=self.sr)
            
            features.setdefault('spectral_centroid', []).append(np.mean(spec_cent))
            features.setdefault('spectral_bandwidth', []).append(np.mean(spec_bw))
        
        return {
            'spectral_centroid': np.mean(features['spectral_centroid']),
            'spectral_bandwidth': np.mean(features['spectral_bandwidth'])
        }
    
@dataclass
class AudioData:
    """Classe para armazenar dados de um arquivo de áudio e suas análises."""
    filename: str
    signal: np.ndarray
    sr: int
    features: Optional[np.ndarray] = field(default=None)
    labels: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        """Verificações e conversões pós-inicialização."""
        # Garantir que signal é um array numpy
        if not isinstance(self.signal, np.ndarray):
            self.signal = np.array(self.signal)
        
        # Converter para mono se for estéreo
        if len(self.signal.shape) > 1:
            self.signal = np.mean(self.signal, axis=1)
    
class MultiAudioAnalyzer:
    """Classe para análise de múltiplos arquivos de áudio."""
    
    def __init__(self, window_length_ms: float = 500):
        self.window_length_ms = window_length_ms
        self.audio_files: Dict[str, AudioData] = {}
        self.k: Optional[int] = None
        self.kmeans = None
        self.scaler = StandardScaler()
        
        # Calcular n_fft baseado no tamanho da janela
        self.base_sr = 44100  # Taxa de amostragem base para referência
        self.base_samples = int(self.base_sr * window_length_ms / 1000)
        self.n_fft = 2 ** int(np.ceil(np.log2(self.base_samples)))
        
        print(f"\nConfigurações de análise:")
        print(f"- Tamanho da janela: {window_length_ms} ms")
        print(f"- Tamanho da FFT (n_fft): {self.n_fft} amostras")
        print(f"- Resolução frequencial: {self.base_sr / self.n_fft:.2f} Hz")
        
    def load_audio_files(self, file_paths: List[str]) -> None:
        """Carrega múltiplos arquivos de áudio."""
        for path in file_paths:
            try:
                print(f"\nCarregando arquivo: {path}")
                signal, sr = librosa.load(path, sr=None)
                
                # Criar instância de AudioData
                audio_data = AudioData(
                    filename=path,
                    signal=signal,
                    sr=sr
                )
                
                self.audio_files[path] = audio_data
                print(f"Arquivo carregado com sucesso:")
                print(f"- Taxa de amostragem: {sr} Hz")
                print(f"- Duração: {len(signal)/sr:.2f} segundos")
                
            except Exception as e:
                print(f"Erro ao carregar {path}: {str(e)}")
                print(f"Detalhes do erro: {type(e).__name__}")
                continue
            
    def extract_features(self, audio_data: AudioData) -> np.ndarray:
        """Extrai características do áudio com verificações de erro."""
        # Ajusta o tamanho da janela para a taxa de amostragem específica do arquivo
        frame_length = int(audio_data.sr * self.window_length_ms / 1000)
        hop_length = frame_length // 2
        
        # Ajusta n_fft para ser potência de 2 mais próxima
        n_fft = 2 ** int(np.ceil(np.log2(frame_length)))
        
        features = []
        n_mfcc = 13
        
        print(f"\nProcessando arquivo: {audio_data.filename}")
        print(f"Configurações de análise:")
        print(f"- Taxa de amostragem: {audio_data.sr} Hz")
        print(f"- Tamanho da janela: {frame_length} amostras ({self.window_length_ms} ms)")
        print(f"- Hop length: {hop_length} amostras")
        print(f"- FFT size: {n_fft} amostras")
        
        total_frames = len(audio_data.signal) // hop_length
        processed_frames = 0
        valid_frames = 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(0, len(audio_data.signal), hop_length):
                try:
                    processed_frames += 1
                    
                    # Extrai o frame
                    frame = audio_data.signal[i:i + frame_length]
                    if len(frame) < frame_length:
                        frame = np.pad(frame, (0, frame_length - len(frame)))
                    
                    # Normaliza o frame
                    frame = frame / (np.max(np.abs(frame)) + 1e-10)
                    
                    # Verifica energia do frame
                    if np.sum(frame**2) < 1e-6:
                        continue
                    
                    # Extração de características
                    try:
                        # MFCCs
                        mfccs = librosa.feature.mfcc(
                            y=frame, 
                            sr=audio_data.sr,
                            n_mfcc=n_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length
                        )
                        mfccs = mfccs.mean(axis=1)  # Média ao longo do tempo
                        
                        # Características espectrais
                        spec_cent = librosa.feature.spectral_centroid(
                            y=frame, 
                            sr=audio_data.sr,
                            n_fft=n_fft,
                            hop_length=hop_length
                        ).mean()
                        
                        spec_bw = librosa.feature.spectral_bandwidth(
                            y=frame, 
                            sr=audio_data.sr,
                            n_fft=n_fft,
                            hop_length=hop_length
                        ).mean()
                        
                        spec_rolloff = librosa.feature.spectral_rolloff(
                            y=frame,
                            sr=audio_data.sr,
                            n_fft=n_fft,
                            hop_length=hop_length
                        ).mean()
                        
                        rms = librosa.feature.rms(
                            y=frame,
                            frame_length=n_fft,
                            hop_length=hop_length
                        ).mean()
                        
                        zcr = librosa.feature.zero_crossing_rate(
                            frame
                        ).mean()
                        
                        # Combina características em um vetor
                        feature_vector = np.concatenate([
                            mfccs,
                            [spec_cent],
                            [spec_bw],
                            [spec_rolloff],
                            [rms],
                            [zcr]
                        ])
                        
                        # Verifica se há valores inválidos
                        if not np.any(np.isnan(feature_vector)) and not np.any(np.isinf(feature_vector)):
                            features.append(feature_vector)
                            valid_frames += 1
                        
                        # Feedback do progresso
                        if processed_frames % 100 == 0:
                            progress = (processed_frames / total_frames) * 100
                            print(f"Progresso: {progress:.1f}% ({valid_frames} frames válidos)")
                            
                    except Exception as e:
                        print(f"Erro na extração de características do frame: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Erro ao processar frame: {str(e)}")
                    continue
        
        if not features:
            print(f"\nAVISO: Nenhuma característica válida extraída de {audio_data.filename}")
            print("Razões possíveis:")
            print("- Arquivo muito curto")
            print("- Janela de análise muito grande")
            print("- Sinal muito fraco ou silencioso")
            print("- Problemas na decodificação do áudio")
            return np.array([])
        
        features_array = np.array(features)
        print(f"\nExtração concluída:")
        print(f"- Total de frames processados: {processed_frames}")
        print(f"- Frames válidos: {valid_frames}")
        print(f"- Shape das características: {features_array.shape}")
        
        return features_array

    def analyze_all_files(self, k: int = None) -> None:
        """Analisa todos os arquivos carregados."""
        if not self.audio_files:
            raise ValueError("Nenhum arquivo de áudio foi carregado")
            
        print("\nIniciando extração de características...")
        all_features = []
        valid_files = []
        
        for path, audio_data in self.audio_files.items():
            features = self.extract_features(audio_data)
            
            if len(features) > 0:
                print(f"\nCaracterísticas extraídas com sucesso de {os.path.basename(path)}")
                audio_data.features = features
                all_features.append(features)
                valid_files.append(path)
            else:
                print(f"\nAVISO: Arquivo ignorado - {os.path.basename(path)}")
        
        if not all_features:
            raise ValueError("Nenhuma característica válida extraída dos arquivos de áudio. "
                            "Tente ajustar o tamanho da janela de análise ou verificar os arquivos.")
        
        # Combina características
        combined_features = np.vstack(all_features)
        print(f"\nTotal de características extraídas: {combined_features.shape}")
        
        # Normalização
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Clustering
        if k is None:
            k = self._find_optimal_k(scaled_features)
        self.k = k
        
        print(f"\nAplicando K-means com {k} clusters...")
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        labels = self.kmeans.fit_predict(scaled_features)
        
        # Distribui labels
        start_idx = 0
        for path in valid_files:
            audio_data = self.audio_files[path]
            if audio_data.features is not None:
                end_idx = start_idx + len(audio_data.features)
                audio_data.labels = labels[start_idx:end_idx]
                start_idx = end_idx
        
        print("Análise concluída com sucesso!")

            # Adicionar exportação de análise
        unique_id = str(uuid.uuid4())[:8]  # Pega apenas os primeiros 8 caracteres para ter um ID mais curto
        analysis_output = f'analysis_output_{unique_id}'
        self._export_analysis_data(analysis_output)
            
    def _find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        """Encontra o número ideal de clusters."""
        print("\nProcurando número ideal de clusters...")
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
            print(f"k = {k}: score = {score:.4f}")
        best_k = np.argmax(silhouette_scores) + 2
        print(f"Melhor k encontrado: {best_k}")
        return best_k
    
    def _export_analysis_data(self, output_folder: str) -> None:
        """Exporta dados detalhados da análise."""
        analysis_folder = os.path.join(output_folder, 'analysis_data')
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Informações do K-Means
        kmeans_info = {
            'n_clusters': int(self.k),  # Convertendo para int Python padrão
            'cluster_centers': self.kmeans.cluster_centers_.tolist(),
            'inertia': float(self.kmeans.inertia_),  # Convertendo para float Python padrão
            'n_iter': int(self.kmeans.n_iter_)  # Convertendo para int Python padrão
        }
        
        with open(os.path.join(analysis_folder, 'kmeans_analysis.json'), 'w') as f:
            json.dump(kmeans_info, f, indent=4)
        
        # Estatísticas dos clusters
        cluster_stats = {}
        labels = np.concatenate([audio.labels for audio in self.audio_files.values() if audio.labels is not None])
        
        for i in range(self.k):
            cluster_mask = labels == i
            cluster_stats[f'cluster_{i}'] = {
                'size': int(np.sum(cluster_mask)),  # Convertendo para int Python padrão
                'percentage': float(np.sum(cluster_mask) / len(labels) * 100)  # Convertendo para float Python padrão
            }
        
        with open(os.path.join(analysis_folder, 'cluster_statistics.json'), 'w') as f:
            json.dump(cluster_stats, f, indent=4)
        
        # Descritores de áudio por cluster
        descriptor_stats = {}
        for i in range(self.k):
            descriptor_stats[f'cluster_{i}'] = self._calculate_cluster_descriptors(i)
        
        with open(os.path.join(analysis_folder, 'audio_descriptors.json'), 'w') as f:
            json.dump(descriptor_stats, f, indent=4)

    def _calculate_cluster_descriptors(self, cluster_id: int) -> Dict:
        """Calcula estatísticas dos descritores de áudio para um cluster."""
        descriptors = {
            'mfcc': [],
            'spectral_centroid': [],
            'spectral_bandwidth': [],
            'spectral_rolloff': [],
            'rms': [],
            'zero_crossing_rate': []
        }
        
        for audio_data in self.audio_files.values():
            if audio_data.labels is None:
                continue
                
            cluster_frames = np.where(audio_data.labels == cluster_id)[0]
            for idx in cluster_frames:
                frame_start = idx * int(self.window_length_ms * audio_data.sr / 1000)
                frame_end = frame_start + int(self.window_length_ms * audio_data.sr / 1000)
                frame = audio_data.signal[frame_start:frame_end]
                
                if len(frame) < frame_end - frame_start:
                    frame = np.pad(frame, (0, frame_end - frame_start - len(frame)))
                
                # Calcular descritores
                mfccs = librosa.feature.mfcc(y=frame, sr=audio_data.sr, n_mfcc=13)
                centroid = librosa.feature.spectral_centroid(y=frame, sr=audio_data.sr)
                bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=audio_data.sr)
                rolloff = librosa.feature.spectral_rolloff(y=frame, sr=audio_data.sr)
                rms = librosa.feature.rms(y=frame)
                zcr = librosa.feature.zero_crossing_rate(frame)
                
                # Convertendo valores numpy para Python padrão
                descriptors['mfcc'].append([float(x) for x in mfccs.mean(axis=1)])
                descriptors['spectral_centroid'].append(float(centroid.mean()))
                descriptors['spectral_bandwidth'].append(float(bandwidth.mean()))
                descriptors['spectral_rolloff'].append(float(rolloff.mean()))
                descriptors['rms'].append(float(rms.mean()))
                descriptors['zero_crossing_rate'].append(float(zcr.mean()))
        
        # Calcular estatísticas
        stats = {}
        for desc_name, values in descriptors.items():
            if desc_name == 'mfcc':
                stats[desc_name] = {
                    'mean': [float(x) for x in np.mean(values, axis=0)],
                    'std': [float(x) for x in np.std(values, axis=0)]
                }
            else:
                stats[desc_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return stats

def _plot_cluster_analysis(self, analysis_folder: str) -> None:
    """Gera visualizações da análise de clusters."""
    # PCA para visualização 2D
    pca = PCA(n_components=2)
    all_features = []
    all_labels = []
    
    for audio_data in self.audio_files.values():
        if audio_data.features is not None and audio_data.labels is not None:
            all_features.append(audio_data.features)
            all_labels.extend(audio_data.labels)
    
    features_2d = pca.fit_transform(np.vstack(all_features))
    
    # Plot dos clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=all_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Visualização 2D dos Clusters (PCA)')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.savefig(os.path.join(analysis_folder, 'cluster_visualization.png'))
    plt.close()

def verify_feature_dimensions(self, feature_vector: np.ndarray, n_mfcc: int = 13) -> bool:
    """Verifica se o vetor de características tem as dimensões corretas."""
    expected_size = n_mfcc + 5  # 13 MFCCs + 5 características espectrais
    
    if len(feature_vector) != expected_size:
        print(f"Dimensão incorreta do vetor de características:")
        print(f"- Esperado: {expected_size}")
        print(f"- Obtido: {len(feature_vector)}")
        return False
        
    return True

def print_feature_info(self, features: np.ndarray) -> None:
    """Imprime informações sobre as características extraídas."""
    if len(features) == 0:
        print("Nenhuma característica extraída")
        return
        
    print("\nInformações das características:")
    print(f"- Número de frames: {len(features)}")
    print(f"- Dimensões por frame: {features.shape[1]}")
    print(f"- Média das características: {np.mean(features, axis=0)}")
    print(f"- Desvio padrão: {np.std(features, axis=0)}")

class MarkovTrackGenerator:
    """Classe para geração de múltiplas tracks usando cadeias de Markov."""
    
    def __init__(self, analyzer: MultiAudioAnalyzer):
        self.analyzer = analyzer
        self.tracks: List[Dict] = []
        self.processor = AudioProcessor()
        self.duration_mode = DurationMode.FIXED  # Modo padrão: duração fixa

    def set_duration_mode(self, mode: DurationMode):
        """Define o modo de duração para a geração de tracks."""
        self.duration_mode = mode
        print(f"Modo de duração definido: {mode.value}")
        
    def _calculate_cluster_durations(self) -> Dict[int, float]:
        """Calcula a duração média de cada cluster nos dados originais."""
        cluster_durations = {}
        cluster_counts = {}
        
        # Para cada arquivo de áudio
        for audio_data in self.analyzer.audio_files.values():
            if audio_data.labels is None:
                continue
                
            labels = audio_data.labels
            
            # Encontra sequências consecutivas do mesmo estado
            current_state = labels[0]
            current_length = 1
            
            for i in range(1, len(labels)):
                if labels[i] == current_state:
                    current_length += 1
                else:
                    # Acumula duração para o estado atual
                    state_duration = current_length * (self.analyzer.window_length_ms / 1000)
                    
                    cluster_durations.setdefault(current_state, 0)
                    cluster_durations[current_state] += state_duration
                    
                    cluster_counts.setdefault(current_state, 0)
                    cluster_counts[current_state] += 1
                    
                    # Reinicia contagem para o próximo estado
                    current_state = labels[i]
                    current_length = 1
            
            # Processa o último estado da sequência
            state_duration = current_length * (self.analyzer.window_length_ms / 1000)
            cluster_durations.setdefault(current_state, 0)
            cluster_durations[current_state] += state_duration
            cluster_counts.setdefault(current_state, 0)
            cluster_counts[current_state] += 1
        
        # Calcula média para cada cluster
        mean_durations = {}
        for state in cluster_durations:
            if cluster_counts[state] > 0:
                mean_durations[state] = cluster_durations[state] / cluster_counts[state]
            else:
                mean_durations[state] = self.analyzer.window_length_ms / 1000
                
        print("\nDuração média por cluster:")
        for state, duration in sorted(mean_durations.items()):
            print(f"Cluster {state}: {duration:.3f} segundos")
            
        return mean_durations
        
    def _generate_markov_sequence(
        self, 
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        duration_seconds: float
    ) -> Tuple[List[int], List[float]]:
        """Gera sequência de estados usando cadeia de Markov com duração adaptativa."""
        if self.duration_mode == DurationMode.FIXED:
            # Implementação original com duração fixa
            n_frames = int((duration_seconds * 1000) / self.analyzer.window_length_ms)
            
            sequence = []
            times = []
            current_state = np.random.choice(self.analyzer.k, p=initial_probs)
            
            for i in range(n_frames):
                sequence.append(current_state)
                times.append(i * self.analyzer.window_length_ms / 1000)
                current_state = np.random.choice(
                    self.analyzer.k, 
                    p=transition_probs[current_state]
                )
                
            return sequence, times
            
        elif self.duration_mode == DurationMode.CLUSTER_MEAN:
            # Implementação com duração média por cluster
            mean_durations = self._calculate_cluster_durations()
            
            sequence = []
            times = []
            current_time = 0.0
            
            # Continua adicionando estados até atingir a duração alvo
            while current_time < duration_seconds:
                # Seleciona o primeiro estado ou usa a transição
                if not sequence:
                    current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                else:
                    current_state = np.random.choice(
                        self.analyzer.k, 
                        p=transition_probs[sequence[-1]]
                    )
                
                # Adiciona o estado e atualiza o tempo
                sequence.append(current_state)
                times.append(current_time)
                
                # Usa a duração média deste cluster
                current_time += mean_durations[current_state]
            
            return sequence, times
            
        elif self.duration_mode == DurationMode.SEQUENCE_LENGTH:
            # Implementação que preserva sequências consecutivas
            base_sequence, base_times = self._generate_base_sequence(
                initial_probs, transition_probs, duration_seconds
            )
            
            # Compacta sequências consecutivas do mesmo estado
            sequence = []
            times = []
            durations = []
            
            if not base_sequence:
                return [], []
                
            current_state = base_sequence[0]
            current_start = base_times[0]
            current_length = 1
            
            for i in range(1, len(base_sequence)):
                if base_sequence[i] == current_state:
                    current_length += 1
                else:
                    # Adiciona o estado atual com sua duração
                    sequence.append(current_state)
                    times.append(current_start)
                    
                    # Calcula duração baseada no número de frames consecutivos
                    state_duration = current_length * (self.analyzer.window_length_ms / 1000)
                    durations.append(state_duration)
                    
                    # Prepara para o próximo estado
                    current_state = base_sequence[i]
                    current_start = base_times[i]
                    current_length = 1
            
            # Processa o último estado
            sequence.append(current_state)
            times.append(current_start)
            state_duration = current_length * (self.analyzer.window_length_ms / 1000)
            durations.append(state_duration)
            
            return sequence, times, durations  # Retorna também as durações
            
        else:
            raise ValueError(f"Modo de duração não suportado: {self.duration_mode}")
    
    def _generate_base_sequence(
        self, 
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        duration_seconds: float
    ) -> Tuple[List[int], List[float]]:
        """Gera sequência base de estados com duração fixa para processamento posterior."""
        # Similar à implementação original, mas usado como base para o modo SEQUENCE_LENGTH
        n_frames = int((duration_seconds * 1000) / self.analyzer.window_length_ms) * 2  # Gera mais frames para ter margem
        
        sequence = []
        times = []
        current_state = np.random.choice(self.analyzer.k, p=initial_probs)
        
        for i in range(n_frames):
            sequence.append(current_state)
            times.append(i * self.analyzer.window_length_ms / 1000)
            current_state = np.random.choice(
                self.analyzer.k, 
                p=transition_probs[current_state]
            )
            
        return sequence, times
    
    def _generate_audio_from_sequence(self, sequence: List[int], times: List[float], durations: List[float] = None) -> np.ndarray:
        """Gera áudio a partir de uma sequência de estados com durações variáveis."""
        try:
            # Se não houver durações específicas, usa o modo fixo padrão
            if durations is None or self.duration_mode == DurationMode.FIXED:
                return self._generate_audio_fixed_duration(sequence)
                
            # Obtém parâmetros do primeiro arquivo de áudio
            first_audio = list(self.analyzer.audio_files.values())[0]
            sr = first_audio.sr
            
            # Aloca buffer para o áudio final
            total_samples = int(times[-1] * sr + durations[-1] * sr) + 1
            final_audio = np.zeros(total_samples)
            
            # Para cada estado na sequência
            for i, (state, start_time, duration) in enumerate(zip(sequence, times, durations)):
                # Coleta frames para este estado
                frames_in_state = self._collect_frames_for_state(state)
                
                if not frames_in_state:
                    continue
                    
                # Seleciona aleatoriamente um frame como base
                base_frame = random.choice(frames_in_state)
                
                # Ajusta a duração do segmento
                required_samples = int(duration * sr)
                
                if len(base_frame) > required_samples:
                    # Corta o frame se for maior que o necessário
                    segment = base_frame[:required_samples]
                else:
                    # Repete o frame ou faz time stretching se necessário
                    repetitions_needed = np.ceil(required_samples / len(base_frame))
                    if repetitions_needed <= 3:  # Se precisar de poucas repetições, usa loop
                        repeated = np.tile(base_frame, int(repetitions_needed))
                        segment = repeated[:required_samples]
                    else:  # Caso contrário, usa time stretching
                        stretch_ratio = len(base_frame) / required_samples
                        segment = librosa.effects.time_stretch(base_frame, rate=stretch_ratio)
                        # Ajusta tamanho final se necessário
                        if len(segment) > required_samples:
                            segment = segment[:required_samples]
                        elif len(segment) < required_samples:
                            segment = np.pad(segment, (0, required_samples - len(segment)))
                
                # Aplica janela de fade-in e fade-out
                fade_samples = min(int(0.02 * sr), len(segment) // 4)  # 20ms ou 1/4 do segmento
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    segment[:fade_samples] *= fade_in
                    segment[-fade_samples:] *= fade_out
                
                # Insere no áudio final
                start_sample = int(start_time * sr)
                end_sample = start_sample + len(segment)
                
                if end_sample > len(final_audio):
                    # Redimensiona o buffer se necessário
                    final_audio = np.pad(final_audio, (0, end_sample - len(final_audio)))
                
                # Adiciona o segmento com crossfade se houver sobreposição
                final_audio[start_sample:end_sample] += segment
            
            # Normaliza o resultado final
            final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-10)
            
            return final_audio
            
        except Exception as e:
            print(f"Erro na geração de áudio com duração variável: {str(e)}")
            raise
        
    def generate_tracks(
        self,
        num_tracks: int,
        duration_seconds: float,
        synthesis_type: SynthesisType = SynthesisType.CONCATENATIVE,
        synthesis_params: Dict = None
    ) -> None:
        """Gera múltiplas tracks usando diferentes métodos de síntese."""
        synthesis_params = synthesis_params or {}
        self.tracks = []  # Limpa tracks anteriores
        
        for track_id in range(num_tracks):
            try:
                print(f"\nGerando track {track_id + 1}/{num_tracks}")
                
                # Gera sequência Markov e áudio base
                track_data = self._generate_single_track(duration_seconds)
                
                # Aplica o método de síntese escolhido
                print(f"Aplicando síntese {synthesis_type.value}...")
                audio_segments = self._generate_audio_segments(
                    track_data['sequence'],
                    synthesis_type,
                    synthesis_params,
                    track_data.get('times'),  # Passa tempos para suportar duração variável
                    track_data.get('durations')  # Passa durações para suportar duração variável
                )
                
                # Processa áudio final
                final_audio = self.processor.process_batch(
                    audio_segments,
                    synthesis_type,
                    duration_seconds,
                    **synthesis_params
                )
                
                # Normaliza e ajusta duração
                final_audio = self.processor.normalize_audio(final_audio)
                final_audio = self.processor.adjust_duration(
                    final_audio,
                    duration_seconds
                )
                
                track_data['audio'] = final_audio
                self.tracks.append(track_data)
                
                print(f"Track {track_id + 1} gerada com sucesso")
                
            except Exception as e:
                print(f"Erro na geração da track {track_id + 1}: {str(e)}")
                continue

    def _generate_audio_segments(
        self,
        sequence: List[int],
        synthesis_type: SynthesisType,
        params: Dict,
        times: List[float] = None,
        durations: List[float] = None
    ) -> List[np.ndarray]:
        """Gera segmentos de áudio usando o método de síntese especificado."""
        print(f"Gerando segmentos de áudio usando síntese {synthesis_type.value}...")
        segments = []
        
        # Para modos de duração variável, gera um único segmento grande
        if self.duration_mode != DurationMode.FIXED and times is not None:
            try:
                # Gera um único segmento de áudio usando duração variável
                segment = self._generate_audio_from_sequence(sequence, times, durations)
                segments.append(segment)
                return segments
            except Exception as e:
                print(f"Erro ao gerar áudio com duração variável: {str(e)}")
                print("Recorrendo ao método de duração fixa...")
                # Continua com o método original em caso de erro
        
        # Método original para duração fixa
        for i, state in enumerate(sequence):
            try:
                # Coleta frames do estado atual
                frames = self._collect_frames_for_state(state)
                if not frames:
                    print(f"Aviso: Nenhum frame disponível para o estado {state}")
                    continue

                # Processamento específico para cada tipo de síntese
                if synthesis_type == SynthesisType.CONCATENATIVE:
                    if frames:
                        # Seleciona um frame aleatório para síntese concatenativa
                        selected_frame = random.choice(frames)
                        segments.append(selected_frame)
                
                elif synthesis_type == SynthesisType.GRANULAR:
                    if frames:
                        # Configura parâmetros granulares
                        self.processor.granular.grain_size = params.get('grain_size', 0.1)
                        self.processor.granular.density = params.get('density', 100)
                        self.processor.granular.pitch_shift = params.get('pitch_shift', 0)
                        self.processor.granular.position_jitter = params.get('position_jitter', 0.1)
                        self.processor.granular.duration_jitter = params.get('duration_jitter', 0.1)
                        
                        # Concatena os frames e aplica síntese granular
                        combined_frame = np.concatenate(frames)
                        segment = self.processor.granular.synthesize(
                            combined_frame,
                            duration=len(frames[0])/self.analyzer.audio_files[list(self.analyzer.audio_files.keys())[0]].sr
                        )
                        segments.append(segment)
                
                elif synthesis_type == SynthesisType.SPECTRAL:
                    if frames:
                        # Configura parâmetros espectrais
                        self.processor.spectral.fft_size = params.get('fft_size', 2048)
                        self.processor.spectral.preserve_transients = params.get('preserve_transients', True)
                        self.processor.spectral.spectral_stretch = params.get('spectral_stretch', 1.0)
                        
                        # Extrai características e aplica síntese espectral
                        target_features = self._extract_target_features(frames)
                        combined_frame = np.concatenate(frames)
                        segment = self.processor.spectral.synthesize(
                            combined_frame,
                            target_features
                        )
                        segments.append(segment)
                
                if i % 10 == 0:  # Feedback a cada 10 segmentos
                    print(f"Processados {i+1}/{len(sequence)} segmentos")
                    
            except Exception as e:
                print(f"Erro ao processar segmento {i} (estado {state}): {str(e)}")
                continue
        
        print(f"Gerados {len(segments)} segmentos de áudio")
        return segments if segments else [np.zeros(1024)]  # Retorna silêncio se não houver segmentos
    
    def _collect_frames_for_state(self, state: int) -> List[np.ndarray]:
        """Coleta todos os frames de áudio correspondentes a um estado."""
        frames = []
        first_audio = next(iter(self.analyzer.audio_files.values()))
        frame_length = int(first_audio.sr * self.analyzer.window_length_ms / 1000)
        hop_length = frame_length // 2

        try:
            for audio_data in self.analyzer.audio_files.values():
                if audio_data.labels is None:
                    continue
                    
                state_positions = np.where(audio_data.labels == state)[0]
                
                for pos in state_positions:
                    start = pos * hop_length
                    end = start + frame_length
                    
                    if start >= len(audio_data.signal):
                        continue
                        
                    frame = audio_data.signal[start:min(end, len(audio_data.signal))]
                    
                    if len(frame) < frame_length:
                        frame = np.pad(frame, (0, frame_length - len(frame)))
                    
                    # Normaliza o frame
                    frame = frame / (np.max(np.abs(frame)) + 1e-10)
                    frames.append(frame)

        except Exception as e:
            print(f"Erro ao coletar frames para o estado {state}: {str(e)}")
            
        return frames

    def process_batch(
        self,
        segments: List[np.ndarray],
        synthesis_type: SynthesisType,
        duration: float,
        **kwargs
    ) -> np.ndarray:
        """Processa um lote de segmentos de áudio."""
        try:
            if not segments:
                return np.zeros(int(duration * self.analyzer.audio_files[0].sr))
            
            if synthesis_type == SynthesisType.CONCATENATIVE:
                # Aplica crossfade entre segmentos
                crossfade_duration = kwargs.get('crossfade_duration', 0.1)
                return self.processor.concatenative.synthesize(
                    segments,
                    crossfade_duration=crossfade_duration
                )
            
            elif synthesis_type == SynthesisType.GRANULAR:
                # Concatena todos os segmentos e aplica síntese granular
                combined_audio = np.concatenate(segments)
                return self.processor.granular.synthesize(combined_audio, duration)
            
            elif synthesis_type == SynthesisType.SPECTRAL:
                # Combina características e aplica síntese espectral
                combined_features = self.processor._combine_spectral_features(segments)
                combined_audio = np.concatenate(segments)
                return self.processor.spectral.synthesize(combined_audio, combined_features)
            
        except Exception as e:
            print(f"Erro no processamento em lote: {str(e)}")
            return np.zeros(int(duration * self.analyzer.audio_files[0].sr))
    
    def _extract_target_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Extrai características espectrais médias dos frames."""
        features = {
            'spectral_centroid': [],
            'spectral_bandwidth': [],
            'spectral_rolloff': [],
            'spectral_flatness': []
        }
        
        for frame in frames:
            features['spectral_centroid'].append(
                librosa.feature.spectral_centroid(y=frame, sr=self.analyzer.audio_files[0].sr).mean()
            )
            features['spectral_bandwidth'].append(
                librosa.feature.spectral_bandwidth(y=frame, sr=self.analyzer.audio_files[0].sr).mean()
            )
            features['spectral_rolloff'].append(
                librosa.feature.spectral_rolloff(y=frame, sr=self.analyzer.audio_files[0].sr).mean()
            )
            features['spectral_flatness'].append(
                librosa.feature.spectral_flatness(y=frame).mean()
            )
        
        return {k: np.mean(v) for k, v in features.items()}
    
    def export_tracks(self, output_folder: str) -> None:
        """Exporta todas as tracks com metadados aprimorados."""
        os.makedirs(output_folder, exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]  # Pega apenas os primeiros 8 caracteres para ter um ID mais curto
        
        try:
            print("\nExportando tracks e análises...")
            
            # Exporta tracks individuais
            all_audio = []
            for i, track in enumerate(self.tracks):
                # Cria pasta para a track
                track_folder = os.path.join(output_folder, f'track_{i+1}')
                os.makedirs(track_folder, exist_ok=True)
                
                # Exporta áudio
                track_path = os.path.join(track_folder, f'audio.wav')
                
                # Obter sr do primeiro arquivo de áudio
                first_key = next(iter(self.analyzer.audio_files))
                sr = self.analyzer.audio_files[first_key].sr
                
                # Salva o arquivo de áudio
                sf.write(track_path, track['audio'], sr)
                all_audio.append(track['audio'])
                
                # Salva metadados
                metadata = {
                    'sequence_length': len(track['sequence']),
                    'unique_states': len(np.unique(track['sequence'])),
                    'duration_mode': self.duration_mode.value,
                    'unique_id': unique_id
                }
                
                # Adiciona metadados específicos para modos de duração variável
                if self.duration_mode != DurationMode.FIXED and 'durations' in track:
                    metadata['durations_stats'] = {
                        'min': float(np.min(track['durations'])),
                        'max': float(np.max(track['durations'])),
                        'mean': float(np.mean(track['durations'])),
                        'std': float(np.std(track['durations']))
                    }
                
                with open(os.path.join(track_folder, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # Exporta análises para esta track
                self._export_track_analysis(track, track_folder)
            
            # Verifica se há tracks para combinar
            if not all_audio:
                print("Nenhuma track foi gerada para exportação.")
                return
                
            # Ajusta os tamanhos dos arrays para combinação
            max_length = max(len(audio) for audio in all_audio)
            normalized_audio = []
            for audio in all_audio:
                if len(audio) < max_length:
                    # Preenche com zeros se necessário
                    padded = np.pad(audio, (0, max_length - len(audio)))
                    normalized_audio.append(padded)
                else:
                    normalized_audio.append(audio)
            
            # Mix final com normalização
            final_mix = np.sum(normalized_audio, axis=0)
            if np.max(np.abs(final_mix)) > 0:  # Evita divisão por zero
                final_mix = final_mix / np.max(np.abs(final_mix))
            
            # Salva mix final
            mix_path = os.path.join(output_folder, 'final_mix.wav')
            sf.write(mix_path, final_mix, sr)
            
            # Análise do mix final
            self._export_mix_analysis(output_folder)
            
            print(f"Exportação concluída. Arquivos salvos em: {output_folder}")
        
        except Exception as e:
            print(f"Erro durante exportação: {str(e)}")
            import traceback
            traceback.print_exc()

    def _generate_single_track(self, duration_seconds: float) -> Dict:
        """Gera uma única track com sua própria cadeia de Markov."""
        try:
            # Calcula probabilidades iniciais
            initial_probs = self._calculate_initial_probs()
            print("Probabilidades iniciais calculadas")
            
            # Calcula matriz de transição
            transition_probs = self._calculate_transition_matrix()
            print("Matriz de transição calculada")
            
            # Gera sequência Markov com tratamento adequado para diferentes modos de duração
            if self.duration_mode == DurationMode.FIXED:
                # No modo FIXED, a função retorna apenas sequence e times (durations é None)
                sequence, times = self._generate_markov_sequence(
                    initial_probs, 
                    transition_probs, 
                    duration_seconds
                )
                durations = None
                print(f"Sequência Markov gerada (modo FIXED): {len(sequence)} estados")
            else:
                # Nos outros modos, a função retorna sequence, times e durations
                try:
                    sequence, times, durations = self._generate_markov_sequence(
                        initial_probs, 
                        transition_probs, 
                        duration_seconds
                    )
                    print(f"Sequência Markov gerada (modo {self.duration_mode.value}): {len(sequence)} estados")
                except ValueError as e:
                    # Em caso de erro, usa o modo FIXED como fallback
                    print(f"Erro ao gerar sequência no modo {self.duration_mode.value}: {str(e)}")
                    print("Recorrendo ao modo FIXED como fallback")
                    sequence, times = self._generate_markov_sequence(
                        initial_probs, 
                        transition_probs, 
                        duration_seconds
                    )
                    durations = None
            
            # Verifica se há frames disponíveis para cada estado
            frames_available = self._verify_frames_availability(sequence)
            if not frames_available:
                print("AVISO: Alguns estados não têm frames disponíveis. A geração pode não ser ideal.")
            
            # Prepara o dicionário de resultado
            result = {
                'sequence': sequence,
                'times': times,
                'initial_probs': initial_probs,
                'transition_probs': transition_probs
            }
            
            # Adiciona durações apenas se disponíveis
            if durations is not None:
                result['durations'] = durations
            
            return result
            
        except Exception as e:
            print(f"Erro na geração da track: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Em caso de erro, retorna uma sequência mínima válida
            # para evitar que a aplicação falhe completamente
            fallback_sequence = [0]
            fallback_times = [0.0]
            return {
                'sequence': fallback_sequence,
                'times': fallback_times,
                'initial_probs': np.ones(self.analyzer.k) / self.analyzer.k,
                'transition_probs': np.ones((self.analyzer.k, self.analyzer.k)) / self.analyzer.k
            }
            
    def _verify_frames_availability(self, sequence: List[int]) -> bool:
        """Verifica se há frames disponíveis para cada estado na sequência."""
        for state in np.unique(sequence):
            frames = self._collect_frames_for_state(state)
            if not frames:
                print(f"Aviso: Nenhum frame disponível para o estado {state}")
                return False
        return True
        
    def _calculate_initial_probs(self) -> np.ndarray:
        """Calcula probabilidades iniciais dos estados."""
        initial_probs = np.zeros(self.analyzer.k)
        total_frames = 0
        
        for audio_data in self.analyzer.audio_files.values():
            for label in audio_data.labels:
                initial_probs[label] += 1
                total_frames += 1
                
        return initial_probs / total_frames
        
    def _calculate_transition_matrix(self) -> np.ndarray:
        """Calcula matriz de transição combinada de todos os arquivos."""
        k = self.analyzer.k
        transition_matrix = np.zeros((k, k))
        
        for audio_data in self.analyzer.audio_files.values():
            labels = audio_data.labels
            for i in range(len(labels) - 1):
                transition_matrix[labels[i], labels[i + 1]] += 1
                
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                   where=row_sums!=0)
        return transition_probs
        
    def _generate_markov_sequence(
        self, 
        initial_probs: np.ndarray,
        transition_probs: np.ndarray,
        duration_seconds: float
    ) -> Tuple[List[int], List[float], Optional[List[float]]]:
        """Gera sequência de estados usando cadeia de Markov com duração adaptativa."""
        try:
            if self.duration_mode == DurationMode.FIXED:
                # Implementação original com duração fixa
                n_frames = int((duration_seconds * 1000) / self.analyzer.window_length_ms)
                
                sequence = []
                times = []
                current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                
                for i in range(n_frames):
                    sequence.append(current_state)
                    times.append(i * self.analyzer.window_length_ms / 1000)
                    # Verifica se há probabilidades válidas para o estado atual
                    if np.sum(transition_probs[current_state]) > 0:
                        current_state = np.random.choice(
                            self.analyzer.k, 
                            p=transition_probs[current_state]
                        )
                    else:
                        # Se não houver transições válidas, escolhe aleatoriamente
                        current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                
                # No modo FIXED, retorna apenas sequence e times
                return sequence, times
                
            elif self.duration_mode == DurationMode.CLUSTER_MEAN:
                # Implementação com duração média por cluster
                mean_durations = self._calculate_cluster_durations()
                
                sequence = []
                times = []
                durations = []
                current_time = 0.0
                
                # Continua adicionando estados até atingir a duração alvo
                while current_time < duration_seconds:
                    # Seleciona o primeiro estado ou usa a transição
                    if not sequence:
                        current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                    else:
                        # Verifica se há probabilidades válidas para o estado atual
                        if np.sum(transition_probs[sequence[-1]]) > 0:
                            current_state = np.random.choice(
                                self.analyzer.k, 
                                p=transition_probs[sequence[-1]]
                            )
                        else:
                            # Se não houver transições válidas, escolhe aleatoriamente
                            current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                    
                    # Adiciona o estado e atualiza o tempo
                    sequence.append(current_state)
                    times.append(current_time)
                    
                    # Usa a duração média deste cluster
                    state_duration = mean_durations.get(current_state, self.analyzer.window_length_ms / 1000)
                    durations.append(state_duration)
                    current_time += state_duration
                
                return sequence, times, durations
                
            elif self.duration_mode == DurationMode.SEQUENCE_LENGTH:
                # Implementação que preserva sequências consecutivas
                try:
                    base_sequence, base_times = self._generate_base_sequence(
                        initial_probs, transition_probs, duration_seconds
                    )
                    
                    # Se não há sequência base, retorna sequências vazias
                    if not base_sequence:
                        return [], [], []
                    
                    # Compacta sequências consecutivas do mesmo estado
                    sequence = []
                    times = []
                    durations = []
                    
                    current_state = base_sequence[0]
                    current_start = base_times[0]
                    current_length = 1
                    
                    for i in range(1, len(base_sequence)):
                        if base_sequence[i] == current_state:
                            current_length += 1
                        else:
                            # Adiciona o estado atual com sua duração
                            sequence.append(current_state)
                            times.append(current_start)
                            
                            # Calcula duração baseada no número de frames consecutivos
                            state_duration = current_length * (self.analyzer.window_length_ms / 1000)
                            durations.append(state_duration)
                            
                            # Prepara para o próximo estado
                            current_state = base_sequence[i]
                            current_start = base_times[i]
                            current_length = 1
                    
                    # Processa o último estado
                    sequence.append(current_state)
                    times.append(current_start)
                    state_duration = current_length * (self.analyzer.window_length_ms / 1000)
                    durations.append(state_duration)
                    
                    return sequence, times, durations
                
                except Exception as e:
                    print(f"Erro no modo SEQUENCE_LENGTH: {str(e)}")
                    # Em caso de erro, usa o modo FIXED como fallback
                    print("Usando modo FIXED como fallback")
                    n_frames = int((duration_seconds * 1000) / self.analyzer.window_length_ms)
                    sequence = [np.random.choice(self.analyzer.k, p=initial_probs) for _ in range(n_frames)]
                    times = [i * self.analyzer.window_length_ms / 1000 for i in range(n_frames)]
                    return sequence, times
                
            else:
                raise ValueError(f"Modo de duração não suportado: {self.duration_mode}")
        
        except Exception as e:
            print(f"Erro ao gerar sequência: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Em caso de falha crítica, retorna uma sequência mínima para evitar quebra completa
            return [0], [0.0]
        
    def _generate_audio_from_sequence(self, sequence: List[int], times: List[float] = None, durations: List[float] = None) -> np.ndarray:
        """Gera áudio a partir de uma sequência de estados com suporte a durações variáveis."""
        try:
            # Se estiver no modo fixo ou não houver durações específicas, usa a implementação original
            if self.duration_mode == DurationMode.FIXED or durations is None:
                # Obtém parâmetros do primeiro arquivo de áudio
                first_audio = list(self.analyzer.audio_files.values())[0]
                frame_length = int(first_audio.sr * self.analyzer.window_length_ms / 1000)
                hop_length = frame_length // 2
                
                audio_segments = []
                for state in sequence:
                    frames_in_state = []
                    
                    # Coleta frames de todos os arquivos para este estado
                    for audio_data in self.analyzer.audio_files.values():
                        if audio_data.labels is None:
                            continue
                            
                        state_positions = np.where(audio_data.labels == state)[0]
                        
                        for pos in state_positions:
                            start = pos * hop_length
                            end = start + frame_length
                            
                            if start >= len(audio_data.signal):
                                continue
                                
                            frame = audio_data.signal[start:min(end, len(audio_data.signal))]
                            
                            if len(frame) < frame_length:
                                frame = np.pad(frame, (0, frame_length - len(frame)))
                                
                            frames_in_state.append(frame)
                    
                    if frames_in_state:
                        selected_frame = random.choice(frames_in_state)
                        audio_segments.append(selected_frame)
                    else:
                        # Se não houver frames para este estado, use silêncio
                        audio_segments.append(np.zeros(frame_length))
                
                if not audio_segments:
                    raise ValueError("Nenhum segmento de áudio foi gerado")
                
                # Aplica crossfade entre segmentos
                final_audio = self._apply_crossfade_to_segments(audio_segments)
                
                return final_audio
                
            else:
                # Implementação para durações variáveis (CLUSTER_MEAN ou SEQUENCE_LENGTH)
                # Obtém parâmetros do primeiro arquivo de áudio
                first_audio = list(self.analyzer.audio_files.values())[0]
                sr = first_audio.sr
                
                # Aloca buffer para o áudio final
                # Certifica-se de que times e durations existam
                if times is None:
                    raise ValueError("Parâmetro 'times' obrigatório para modos de duração variável")
                    
                total_samples = int(times[-1] * sr + durations[-1] * sr) + 1
                final_audio = np.zeros(total_samples)
                
                # Para cada estado na sequência
                for i, (state, start_time, duration) in enumerate(zip(sequence, times, durations)):
                    # Coleta frames para este estado
                    frames_in_state = self._collect_frames_for_state(state)
                    
                    if not frames_in_state:
                        print(f"Aviso: Nenhum frame disponível para o estado {state}")
                        continue
                        
                    # Seleciona aleatoriamente um frame como base
                    base_frame = random.choice(frames_in_state)
                    
                    # Ajusta a duração do segmento
                    required_samples = int(duration * sr)
                    
                    if len(base_frame) > required_samples:
                        # Corta o frame se for maior que o necessário
                        segment = base_frame[:required_samples]
                    else:
                        # Repete o frame ou faz time stretching se necessário
                        repetitions_needed = np.ceil(required_samples / len(base_frame))
                        if repetitions_needed <= 3:  # Se precisar de poucas repetições, usa loop
                            repeated = np.tile(base_frame, int(repetitions_needed))
                            segment = repeated[:required_samples]
                        else:  # Caso contrário, usa time stretching
                            try:
                                stretch_ratio = len(base_frame) / required_samples
                                segment = librosa.effects.time_stretch(base_frame, rate=stretch_ratio)
                                # Ajusta tamanho final se necessário
                                if len(segment) > required_samples:
                                    segment = segment[:required_samples]
                                elif len(segment) < required_samples:
                                    segment = np.pad(segment, (0, required_samples - len(segment)))
                            except Exception as e:
                                print(f"Erro no time stretching: {str(e)}. Usando looping.")
                                # Fallback para looping em caso de erro
                                repeated = np.tile(base_frame, int(repetitions_needed))
                                segment = repeated[:required_samples]
                    
                    # Aplica janela de fade-in e fade-out para evitar cliques
                    fade_samples = min(int(0.02 * sr), len(segment) // 4)  # 20ms ou 1/4 do segmento
                    if fade_samples > 0:
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        segment[:fade_samples] *= fade_in
                        segment[-fade_samples:] *= fade_out
                    
                    # Insere no áudio final
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + len(segment)
                    
                    if end_sample > len(final_audio):
                        # Redimensiona o buffer se necessário
                        final_audio = np.pad(final_audio, (0, end_sample - len(final_audio)))
                    
                    # Adiciona o segmento
                    final_audio[start_sample:end_sample] += segment
                
                # Normaliza o resultado final
                if np.max(np.abs(final_audio)) > 0:
                    final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-10)
                
                return final_audio
                
        except Exception as e:
            print(f"Erro na geração de áudio: {str(e)}")
            traceback.print_exc()  # Imprime o stack trace para depuração
            # Retorna silêncio em caso de erro para evitar falha completa
            return np.zeros(int(5 * first_audio.sr))  # 5 segundos de silêncio
    
    def _apply_crossfade_to_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """Aplica crossfade entre segmentos de áudio."""
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        fade_duration = 0.010  # 10ms de crossfade
        sr = self.analyzer.audio_files[list(self.analyzer.audio_files.keys())[0]].sr
        fade_length = int(fade_duration * sr)
        
        result = segments[0]
        for i in range(1, len(segments)):
            if len(result) < fade_length or len(segments[i]) < fade_length:
                result = np.concatenate([result, segments[i]])
                continue
                
            fade_out = np.linspace(1.0, 0.0, fade_length)
            fade_in = np.linspace(0.0, 1.0, fade_length)
            
            result_end = result[-fade_length:] * fade_out
            next_start = segments[i][:fade_length] * fade_in
            
            result = np.concatenate([
                result[:-fade_length],
                result_end + next_start,
                segments[i][fade_length:]
            ])
            
        return result
        
    def _export_track_analysis(self, track: Dict, output_folder: str) -> None:
        """Exporta análises para uma track individual com tratamento de erros aprimorado."""
        # Extrai o número da track do nome da pasta
        track_folder = os.path.basename(output_folder)
        track_num = track_folder.replace('track_', '').split('_')[0]
        
        try:
            analysis_folder = os.path.join(output_folder, 'analysis')
            os.makedirs(analysis_folder, exist_ok=True)
            
            # Verifica se os dados necessários estão presentes
            if 'sequence' not in track or 'times' not in track:
                print(f"Dados incompletos para análise da track {track_num}")
                
                # Cria um arquivo de aviso
                with open(os.path.join(analysis_folder, 'aviso.txt'), 'w') as f:
                    f.write(f"Dados incompletos para análise da track {track_num}\n")
                    f.write("Não foi possível gerar análises detalhadas.")
                return
            
            # Dados da sequência
            try:
                sequence_df = pd.DataFrame({
                    'Time': track['times'],
                    'State': track['sequence']
                })
                sequence_df.to_csv(os.path.join(analysis_folder, 'sequence.csv'), index=False)
            except Exception as e:
                print(f"Erro ao exportar dados da sequência: {str(e)}")
            
            # Matriz de transição
            try:
                if 'transition_probs' in track and isinstance(track['transition_probs'], np.ndarray):
                    transition_df = pd.DataFrame(
                        track['transition_probs'],
                        columns=[f'To_State_{i}' for i in range(self.analyzer.k)],
                        index=[f'From_State_{i}' for i in range(self.analyzer.k)]
                    )
                    transition_df.to_csv(os.path.join(analysis_folder, 'transition_matrix.csv'))
            except Exception as e:
                print(f"Erro ao exportar matriz de transição: {str(e)}")
            
            # Análise estatística
            try:
                with open(os.path.join(analysis_folder, 'statistics.txt'), 'w') as f:
                    f.write(f"Análise Estatística - Track {track_num}\n")
                    f.write("================================\n\n")
                    
                    # Distribuição dos estados
                    state_counts = np.bincount(track['sequence'], minlength=self.analyzer.k)
                    f.write("Distribuição dos Estados:\n")
                    for state, count in enumerate(state_counts):
                        if state < self.analyzer.k:  # Garante que estamos dentro dos limites
                            percentage = (count / len(track['sequence'])) * 100
                            f.write(f"Estado {state}: {count} ocorrências ({percentage:.2f}%)\n")
                    
                    # Transições mais frequentes
                    if 'transition_probs' in track:
                        f.write("\nTransições mais frequentes:\n")
                        for i in range(self.analyzer.k):
                            if np.sum(track['transition_probs'][i]) > 0:  # Verifica se há transições
                                max_prob_idx = np.argmax(track['transition_probs'][i])
                                f.write(f"Estado {i} -> Estado {max_prob_idx}: "
                                    f"{track['transition_probs'][i][max_prob_idx]:.4f}\n")
                            else:
                                f.write(f"Estado {i}: Sem transições\n")
            except Exception as e:
                print(f"Erro ao exportar estatísticas: {str(e)}")
                
            # Visualizações
            try:
                self._plot_track_analysis(track, track_num).savefig(
                    os.path.join(analysis_folder, 'analysis.png')
                )
            except Exception as e:
                print(f"Erro ao gerar visualização: {str(e)}")
                # Cria uma figura de erro
                fig = plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Erro ao gerar visualização: {str(e)}",
                        ha='center', va='center', fontsize=12)
                plt.savefig(os.path.join(analysis_folder, 'error.png'))
        
        except Exception as e:
            print(f"Erro ao exportar análise da track: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Cria pasta de análise mesmo em caso de erro
            try:
                analysis_folder = os.path.join(output_folder, 'analysis')
                os.makedirs(analysis_folder, exist_ok=True)
                
                # Salva informação do erro
                with open(os.path.join(analysis_folder, 'error_log.txt'), 'w') as f:
                    f.write(f"Erro ao exportar análise: {str(e)}\n\n")
                    f.write(traceback.format_exc())
            except:
                pass

    def _plot_track_analysis(self, track: Dict, track_num) -> plt.Figure:
        """Cria visualizações para análise de uma track."""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Sequência de estados
            plt.subplot(311)
            plt.plot(track['times'], track['sequence'], 'b-', alpha=0.6)
            plt.scatter(track['times'], track['sequence'], c=track['sequence'], 
                    cmap='viridis', alpha=0.6)
            plt.title(f'Sequência de Estados - Track {track_num}')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Estado')
            plt.colorbar(label='Estado')
            
            # Plot 2: Distribuição dos estados
            plt.subplot(312)
            state_counts = np.bincount(track['sequence'])
            plt.bar(range(len(state_counts)), state_counts)
            plt.title('Distribuição dos Estados')
            plt.xlabel('Estado')
            plt.ylabel('Número de Ocorrências')
            
            # Plot 3: Matriz de transição
            plt.subplot(313)
            sns.heatmap(track['transition_probs'], annot=True, fmt='.2f', cmap='Blues')
            plt.title('Matriz de Transição')
            plt.xlabel('Para Estado')
            plt.ylabel('De Estado')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Erro ao gerar visualização da track: {str(e)}")
            # Retorna uma figura vazia em caso de erro
            empty_fig = plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, f"Erro ao gerar visualização: {str(e)}",
                    ha='center', va='center')
            return empty_fig
    
    def _export_mix_analysis(self, output_folder: str) -> None:
        """Exporta análise do mix final."""
        analysis_folder = os.path.join(output_folder, 'mix_analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        
        try:
            # Combina dados de todas as tracks
            combined_data = {}
            
            # Em modos de duração variável, as sequências podem ter comprimentos diferentes
            # Precisamos adaptar a análise para lidar com isso
            
            # Cria figura para análise combinada
            fig = plt.figure(figsize=(15, 10))
            
            # Plot de densidade de estados por track
            plt.subplot(211)
            
            # Plota cada track separadamente, usando seus próprios tempos
            for i, track in enumerate(self.tracks):
                times = track['times']
                states = track['sequence']
                
                # Verifica se os dados são compatíveis
                if len(times) == len(states) and len(times) > 0:
                    plt.plot(times, states, label=f'Track {i+1}', alpha=0.6)
            
            plt.title('Estados por Track ao Longo do Tempo')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Estado')
            plt.legend()
            
            # Histograma 2D de estados - apenas se todos os dados tiverem a mesma dimensão
            plt.subplot(212)
            
            # Usa uma abordagem diferente para o histograma 2D
            # Coleta todos os pontos (tempo, estado) de todas as tracks
            all_times = []
            all_states = []
            
            for track in self.tracks:
                all_times.extend(track['times'])
                all_states.extend(track['sequence'])
            
            # Se houver dados suficientes, cria o histograma 2D
            if all_times and all_states:
                plt.hist2d(all_times, all_states, 
                        bins=[50, self.analyzer.k], 
                        cmap='Blues', alpha=0.7)
                
                plt.title('Densidade de Estados Combinada')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Estado')
                plt.colorbar(label='Número de Ocorrências')
            else:
                plt.text(0.5, 0.5, "Dados insuficientes para histograma", 
                    ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_folder, 'combined_analysis.png'))
            plt.close()
            
            # Salva informações adicionais em texto
            with open(os.path.join(analysis_folder, 'summary.txt'), 'w') as f:
                f.write("Análise do Mix Final\n")
                f.write("===================\n\n")
                f.write(f"Número de tracks: {len(self.tracks)}\n")
                f.write(f"Modo de duração: {self.duration_mode.value}\n\n")
                
                for i, track in enumerate(self.tracks):
                    f.write(f"Track {i+1}:\n")
                    f.write(f"- Número de estados: {len(track['sequence'])}\n")
                    f.write(f"- Estados únicos: {len(np.unique(track['sequence']))}\n")
                    f.write(f"- Duração total: {track['times'][-1]:.2f} segundos\n\n")
        
        except Exception as e:
            print(f"Erro ao gerar análise do mix: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Função principal do programa."""
    continue_program = True
    
    while continue_program:
        try:
            print("\n=== Sistema de Composição Musical baseado em Cadeias de Markov ===")
            print("=== Com Múltiplos Métodos de Síntese de Áudio ===\n")

            # Configuração da interface gráfica
            root = tk.Tk()
            root.withdraw()

            # Solicita seleção de arquivos
            print("Selecione os arquivos de áudio para análise...")
            audio_paths = filedialog.askopenfilenames(
                title="Selecione os arquivos de áudio",
                filetypes=[("Arquivos de Áudio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif")]
            )

            if not audio_paths:
                print("Nenhum arquivo foi selecionado. O programa será encerrado.")
                break

            print(f"\nArquivos selecionados: {len(audio_paths)}")
            for path in audio_paths:
                print(f"- {os.path.basename(path)}")

            # Parâmetros básicos
            window_length_ms = float(input("\nDigite o tamanho da janela de análise em milissegundos (ex: 500): "))
            num_tracks = int(input("Digite o número de tracks desejadas para a composição: "))
            duration_seconds = float(input("Digite a duração desejada da composição em segundos: "))

            # Modo de clustering
            while True:
                mode = input("\nDeseja definir o número de clusters manualmente ou automaticamente? "
                           "Digite 'manual' ou 'automático': ").strip().lower()
                if mode in ['manual', 'automático']:
                    break
                print("Opção inválida. Por favor, digite 'manual' ou 'automático'.")

            k = None
            if mode == 'manual':
                k = int(input("Digite o número de clusters desejado (ex: 4): "))

            # Inicializa e executa o analisador
            print("\nIniciando análise dos arquivos...")
            analyzer = MultiAudioAnalyzer(window_length_ms=window_length_ms)
            analyzer.load_audio_files(audio_paths)
            analyzer.analyze_all_files(k=k)

            # Escolha do método de síntese
            print("\nEscolha o tipo de síntese:")
            print("1. Concatenativa (com crossfade)")
            print("2. Granular")
            print("3. Espectral")
            
            synthesis_choice = int(input("\nDigite o número da opção desejada: "))
            if synthesis_choice not in [1, 2, 3]:
                raise ValueError("Opção inválida")

            synthesis_type = {
                1: SynthesisType.CONCATENATIVE,
                2: SynthesisType.GRANULAR,
                3: SynthesisType.SPECTRAL
            }[synthesis_choice]

            # Parâmetros específicos para cada tipo de síntese
            synthesis_params = {}
            if synthesis_type == SynthesisType.GRANULAR:
                print("\nConfigurações da Síntese Granular:")
                synthesis_params = {
                    'grain_size': float(input("Tamanho do grão (segundos, ex: 0.1): ")),
                    'density': float(input("Densidade (grãos/segundo, ex: 100): ")),
                    'pitch_shift': float(input("Mudança de pitch (semitons, ex: 0): ")),
                    'position_jitter': float(input("Variação da posição (0-1, ex: 0.1): ")),
                    'duration_jitter': float(input("Variação da duração (0-1, ex: 0.1): "))
                }
                
            elif synthesis_type == SynthesisType.SPECTRAL:
                print("\nConfigurações da Síntese Espectral:")
                synthesis_params = {
                    'preserve_transients': input("Preservar transientes? (s/n): ").lower() == 's',
                    'spectral_stretch': float(input("Fator de esticamento espectral (ex: 1.0): ")),
                    'fft_size': int(input("Tamanho da FFT (ex: 2048): "))
                }
                
            else:  # CONCATENATIVE
                print("\nConfigurações da Síntese Concatenativa:")
                synthesis_params = {
                    'crossfade_duration': float(input("Duração do crossfade (segundos, ex: 0.1): "))
                }

            # Escolha do modo de duração
            print("\nEscolha o modo de duração:")
            print("1. Fixa (baseada no tamanho da janela)")
            print("2. Baseada na duração média de cada cluster")
            print("3. Baseada em sequências consecutivas do mesmo estado")
            
            duration_mode_choice = int(input("\nDigite o número da opção desejada: "))
            if duration_mode_choice not in [1, 2, 3]:
                raise ValueError("Opção inválida")

            duration_mode = {
                1: DurationMode.FIXED,
                2: DurationMode.CLUSTER_MEAN,
                3: DurationMode.SEQUENCE_LENGTH
            }[duration_mode_choice]

            # Configurações de normalização e ajuste de duração
            print("\nConfigurações adicionais:")
            normalize_audio = input("Normalizar áudio final? (s/n): ").lower() == 's'
            adjust_duration_method = input("Método de ajuste de duração (stretch/loop): ").lower()
            if adjust_duration_method not in ['stretch', 'loop']:
                raise ValueError("Método de ajuste de duração inválido")

            # Gera as tracks
            print("\nGerando composição...")
            generator = MarkovTrackGenerator(analyzer)
            
            # Define o modo de duração
            generator.set_duration_mode(duration_mode)
            
            generator.generate_tracks(
                num_tracks=num_tracks,
                duration_seconds=duration_seconds,
                synthesis_type=synthesis_type,
                synthesis_params=synthesis_params
            )

            # Exporta resultados
            unique_id = str(uuid.uuid4())[:8]  # Pega apenas os primeiros 8 caracteres para ter um ID mais curto
            output_folder = f'output_multitrack_{unique_id}'
            print(f"\nExportando arquivos...")
            generator.export_tracks(output_folder)

            print(f"\nProcessamento concluído!")
            print(f"Arquivos gerados salvos em: {output_folder}")
            print(f"\nResumo da geração:")
            print(f"- Número de arquivos de entrada: {len(audio_paths)}")
            print(f"- Número de clusters: {analyzer.k}")
            print(f"- Número de tracks geradas: {num_tracks}")
            print(f"- Duração total: {duration_seconds} segundos")
            print(f"- Método de síntese: {synthesis_type.value}")
            print(f"- Modo de duração: {duration_mode.value}")
            
            # Abre o diretório de saída
            if input("\nDeseja abrir o diretório com os arquivos gerados? (s/n): ").lower() == 's':
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder)
                elif os.name == 'posix':  # macOS ou Linux
                    os.system(f'xdg-open {output_folder}')

            # Pergunta se deseja continuar
            continue_response = input("\nDeseja gerar mais material com outros arquivos? (s/n): ").lower()
            continue_program = continue_response == 's'

        except Exception as e:
            print(f"\nErro durante a execução: {str(e)}")
            import traceback
            traceback.print_exc()  # Imprime o traceback completo para depuração
            continue_response = input("\nDeseja tentar novamente com outros arquivos? (s/n): ").lower()
            continue_program = continue_response == 's'

    print("\nPrograma finalizado. Obrigado por usar!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nPrograma finalizado.")