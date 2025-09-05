#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Composição Musical baseado em Cadeias de Markov
Interface Streamlit para geração de múltiplas tracks com diferentes métodos de síntese
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import signal
from scipy.stats import skew, kurtosis
import soundfile as sf
import os
import tempfile
import io
import zipfile
import json
import uuid
import warnings
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import time

# Configurações
warnings.filterwarnings('ignore')
sns.set_theme()

# Enums e Classes
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

# Classes de Síntese
class AudioSynthesizer:
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.window_type = WindowType.HANN
        self.window_size = 2048
        self.hop_size = 512
        
    def get_window(self, size: int) -> np.ndarray:
        if self.window_type == WindowType.GAUSSIAN:
            return signal.gaussian(size, std=size/6.0)
        elif self.window_type == WindowType.KAISER:
            return signal.kaiser(size, beta=14)
        else:
            return signal.get_window(self.window_type.value, size)

class ConcatenativeSynthesizer(AudioSynthesizer):
    def __init__(self, sr: int = 44100, crossfade_duration: float = 0.1):
        super().__init__(sr)
        self.crossfade_duration = crossfade_duration
        
    def synthesize(self, segments: List[np.ndarray], 
                  crossfade_duration: Optional[float] = None) -> np.ndarray:
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
    def __init__(self, sr: int = 44100):
        super().__init__(sr)
        self.grain_size = 0.1
        self.density = 100
        self.pitch_shift = 0
        self.position_jitter = 0.1
        self.duration_jitter = 0.1
        
    def synthesize(self, source: np.ndarray, duration: float) -> np.ndarray:
        grain_samples = int(self.grain_size * self.sr)
        num_grains = int(duration * self.density)
        output = np.zeros(int(duration * self.sr))
        
        window = self.get_window(grain_samples)
        
        for i in range(num_grains):
            position = (i / self.density) + np.random.uniform(
                -self.position_jitter, 
                self.position_jitter
            ) * self.grain_size
            
            grain_duration = self.grain_size * (1 + np.random.uniform(
                -self.duration_jitter,
                self.duration_jitter
            ))
            
            grain_size = int(grain_duration * self.sr)
            if grain_size != len(window):
                window = self.get_window(grain_size)
            
            start_pos = int((position % (len(source) / self.sr)) * self.sr)
            if start_pos + grain_size > len(source):
                continue
                
            grain = source[start_pos:start_pos + grain_size]
            
            if self.pitch_shift != 0:
                grain = librosa.effects.pitch_shift(
                    grain, sr=self.sr, n_steps=self.pitch_shift
                )
            
            grain = grain * window
            
            out_start = int(position * self.sr)
            if out_start + len(grain) > len(output):
                continue
            output[out_start:out_start + len(grain)] += grain
            
        return output

class SpectralSynthesizer(AudioSynthesizer):
    def __init__(self, sr: int = 44100):
        super().__init__(sr)
        self.fft_size = 2048
        self.preserve_transients = True
        self.spectral_stretch = 1.0
        
    def synthesize(self, source: np.ndarray, target_features: Dict) -> np.ndarray:
        D = librosa.stft(source, n_fft=self.fft_size, 
                        hop_length=self.hop_size,
                        window=self.get_window(self.fft_size))
        
        if 'spectral_centroid' in target_features:
            self._adjust_centroid(D, target_features['spectral_centroid'])
        
        if 'spectral_bandwidth' in target_features:
            self._adjust_bandwidth(D, target_features['spectral_bandwidth'])
            
        if self.preserve_transients:
            D = self._preserve_transient_features(D)
            
        if self.spectral_stretch != 1.0:
            D = self._stretch_spectrum(D)
            
        return librosa.istft(D, hop_length=self.hop_size,
                           window=self.get_window(self.fft_size))
    
    def _adjust_centroid(self, D: np.ndarray, target_centroid: float) -> None:
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        current_centroid = np.sum(frequencies[:, np.newaxis] * np.abs(D)) / np.sum(np.abs(D))
        
        ratio = target_centroid / current_centroid
        shift = np.exp(np.log(ratio) * frequencies[:, np.newaxis])
        D *= shift
    
    def _adjust_bandwidth(self, D: np.ndarray, target_bandwidth: float) -> None:
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        centroid = np.sum(frequencies[:, np.newaxis] * np.abs(D)) / np.sum(np.abs(D))
        
        current_bandwidth = np.sqrt(
            np.sum(((frequencies[:, np.newaxis] - centroid) ** 2) * np.abs(D)) / 
            np.sum(np.abs(D))
        )
        
        ratio = target_bandwidth / current_bandwidth
        D *= np.exp(-((frequencies[:, np.newaxis] - centroid) ** 2) * (1 - ratio))
    
    def _preserve_transient_features(self, D: np.ndarray) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(S=np.abs(D), sr=self.sr)
        onset_mask = librosa.util.normalize(onset_env)
        return D * onset_mask
    
    def _stretch_spectrum(self, D: np.ndarray) -> np.ndarray:
        return librosa.phase_vocoder(D, self.spectral_stretch)

class AudioProcessor:
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.concatenative = ConcatenativeSynthesizer(sr)
        self.granular = GranularSynthesizer(sr)
        self.spectral = SpectralSynthesizer(sr)
        
    def process_batch(self, sources: List[np.ndarray], synthesis_type: SynthesisType,
                     duration: float, **kwargs) -> np.ndarray:
        if synthesis_type == SynthesisType.CONCATENATIVE:
            return self.concatenative.synthesize(sources, **kwargs)
        elif synthesis_type == SynthesisType.GRANULAR:
            return self.granular.synthesize(np.concatenate(sources), duration)
        elif synthesis_type == SynthesisType.SPECTRAL:
            combined_features = self._combine_spectral_features(sources)
            return self.spectral.synthesize(np.concatenate(sources), combined_features)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        return audio / (np.max(np.abs(audio)) + 1e-10)
    
    def adjust_duration(self, audio: np.ndarray, target_duration: float,
                       method: str = 'stretch') -> np.ndarray:
        current_duration = len(audio) / self.sr
        
        if method == 'stretch':
            return librosa.effects.time_stretch(audio, rate=current_duration/target_duration)
        elif method == 'loop':
            num_repeats = int(np.ceil(target_duration / current_duration))
            return np.tile(audio, num_repeats)[:int(target_duration * self.sr)]
        else:
            raise ValueError(f"Método de ajuste de duração desconhecido: {method}")
    
    def _combine_spectral_features(self, sources: List[np.ndarray]) -> Dict[str, float]:
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

class MultiAudioAnalyzer:
    def __init__(self, window_length_ms: float = 500):
        self.window_length_ms = window_length_ms
        self.audio_files: Dict[str, AudioData] = {}
        self.k: Optional[int] = None
        self.kmeans = None
        self.scaler = StandardScaler()
        
        self.base_sr = 44100
        self.base_samples = int(self.base_sr * window_length_ms / 1000)
        self.n_fft = 2 ** int(np.ceil(np.log2(self.base_samples)))
        
    def load_audio_files(self, file_contents: List[Tuple[str, bytes]]) -> None:
        """Carrega arquivos de áudio a partir de dados em bytes."""
        for filename, file_content in file_contents:
            try:
                # Salva temporariamente o arquivo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name
                
                # Carrega o áudio
                signal, sr = librosa.load(tmp_path, sr=None)
                
                # Remove o arquivo temporário
                os.unlink(tmp_path)
                
                audio_data = AudioData(filename=filename, signal=signal, sr=sr)
                self.audio_files[filename] = audio_data
                
            except Exception as e:
                st.error(f"Erro ao carregar {filename}: {str(e)}")
                continue
            
    def extract_features(self, audio_data: AudioData) -> np.ndarray:
        frame_length = int(audio_data.sr * self.window_length_ms / 1000)
        hop_length = frame_length // 2
        n_fft = 2 ** int(np.ceil(np.log2(frame_length)))
        
        features = []
        n_mfcc = 13
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(0, len(audio_data.signal), hop_length):
                try:
                    frame = audio_data.signal[i:i + frame_length]
                    if len(frame) < frame_length:
                        frame = np.pad(frame, (0, frame_length - len(frame)))
                    
                    frame = frame / (np.max(np.abs(frame)) + 1e-10)
                    
                    if np.sum(frame**2) < 1e-6:
                        continue
                    
                    mfccs = librosa.feature.mfcc(
                        y=frame, sr=audio_data.sr, n_mfcc=n_mfcc,
                        n_fft=n_fft, hop_length=hop_length
                    ).mean(axis=1)
                    
                    spec_cent = librosa.feature.spectral_centroid(
                        y=frame, sr=audio_data.sr, n_fft=n_fft, hop_length=hop_length
                    ).mean()
                    
                    spec_bw = librosa.feature.spectral_bandwidth(
                        y=frame, sr=audio_data.sr, n_fft=n_fft, hop_length=hop_length
                    ).mean()
                    
                    spec_rolloff = librosa.feature.spectral_rolloff(
                        y=frame, sr=audio_data.sr, n_fft=n_fft, hop_length=hop_length
                    ).mean()
                    
                    rms = librosa.feature.rms(
                        y=frame, frame_length=n_fft, hop_length=hop_length
                    ).mean()
                    
                    zcr = librosa.feature.zero_crossing_rate(frame).mean()
                    
                    feature_vector = np.concatenate([
                        mfccs, [spec_cent], [spec_bw], [spec_rolloff], [rms], [zcr]
                    ])
                    
                    if not np.any(np.isnan(feature_vector)) and not np.any(np.isinf(feature_vector)):
                        features.append(feature_vector)
                        
                except Exception:
                    continue
        
        return np.array(features) if features else np.array([])

    def analyze_all_files(self, k: int = None) -> None:
        if not self.audio_files:
            raise ValueError("Nenhum arquivo de áudio foi carregado")
            
        all_features = []
        valid_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(self.audio_files)
        
        for i, (path, audio_data) in enumerate(self.audio_files.items()):
            status_text.text(f"Processando arquivo {i+1}/{total_files}: {os.path.basename(path)}")
            
            features = self.extract_features(audio_data)
            
            if len(features) > 0:
                audio_data.features = features
                all_features.append(features)
                valid_files.append(path)
            
            progress_bar.progress((i + 1) / total_files)
        
        if not all_features:
            raise ValueError("Nenhuma característica válida extraída dos arquivos de áudio.")
        
        # Combina características
        combined_features = np.vstack(all_features)
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Clustering
        if k is None:
            k = self._find_optimal_k(scaled_features)
        self.k = k
        
        status_text.text(f"Aplicando K-means com {k} clusters...")
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
        
        status_text.text("Análise concluída com sucesso!")
        progress_bar.progress(1.0)
            
    def _find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> int:
        silhouette_scores = []
        for k in range(2, min(max_k + 1, len(features))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
        best_k = np.argmax(silhouette_scores) + 2
        return best_k

class MarkovTrackGenerator:
    def __init__(self, analyzer: MultiAudioAnalyzer):
        self.analyzer = analyzer
        self.tracks: List[Dict] = []
        self.processor = AudioProcessor()
        self.duration_mode = DurationMode.FIXED

    def set_duration_mode(self, mode: DurationMode):
        self.duration_mode = mode
        
    def _calculate_cluster_durations(self) -> Dict[int, float]:
        cluster_durations = {}
        cluster_counts = {}
        
        for audio_data in self.analyzer.audio_files.values():
            if audio_data.labels is None:
                continue
                
            labels = audio_data.labels
            current_state = labels[0]
            current_length = 1
            
            for i in range(1, len(labels)):
                if labels[i] == current_state:
                    current_length += 1
                else:
                    state_duration = current_length * (self.analyzer.window_length_ms / 1000)
                    
                    cluster_durations.setdefault(current_state, 0)
                    cluster_durations[current_state] += state_duration
                    
                    cluster_counts.setdefault(current_state, 0)
                    cluster_counts[current_state] += 1
                    
                    current_state = labels[i]
                    current_length = 1
            
            state_duration = current_length * (self.analyzer.window_length_ms / 1000)
            cluster_durations.setdefault(current_state, 0)
            cluster_durations[current_state] += state_duration
            cluster_counts.setdefault(current_state, 0)
            cluster_counts[current_state] += 1
        
        mean_durations = {}
        for state in cluster_durations:
            if cluster_counts[state] > 0:
                mean_durations[state] = cluster_durations[state] / cluster_counts[state]
            else:
                mean_durations[state] = self.analyzer.window_length_ms / 1000
                
        return mean_durations
        
    def generate_tracks(self, num_tracks: int, duration_seconds: float,
                       synthesis_type: SynthesisType = SynthesisType.CONCATENATIVE,
                       synthesis_params: Dict = None) -> None:
        synthesis_params = synthesis_params or {}
        self.tracks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for track_id in range(num_tracks):
            try:
                status_text.text(f"Gerando track {track_id + 1}/{num_tracks}")
                
                track_data = self._generate_single_track(duration_seconds)
                
                audio_segments = self._generate_audio_segments(
                    track_data['sequence'],
                    synthesis_type,
                    synthesis_params,
                    track_data.get('times'),
                    track_data.get('durations')
                )
                
                final_audio = self.processor.process_batch(
                    audio_segments, synthesis_type, duration_seconds, **synthesis_params
                )
                
                final_audio = self.processor.normalize_audio(final_audio)
                final_audio = self.processor.adjust_duration(final_audio, duration_seconds)
                
                track_data['audio'] = final_audio
                self.tracks.append(track_data)
                
                progress_bar.progress((track_id + 1) / num_tracks)
                
            except Exception as e:
                st.error(f"Erro na geração da track {track_id + 1}: {str(e)}")
                continue
        
        status_text.text("Geração concluída!")

    def _generate_single_track(self, duration_seconds: float) -> Dict:
        initial_probs = self._calculate_initial_probs()
        transition_probs = self._calculate_transition_matrix()
        
        if self.duration_mode == DurationMode.FIXED:
            sequence, times = self._generate_markov_sequence(
                initial_probs, transition_probs, duration_seconds
            )
            durations = None
        else:
            try:
                sequence, times, durations = self._generate_markov_sequence(
                    initial_probs, transition_probs, duration_seconds
                )
            except ValueError:
                sequence, times = self._generate_markov_sequence(
                    initial_probs, transition_probs, duration_seconds
                )
                durations = None
        
        result = {
            'sequence': sequence,
            'times': times,
            'initial_probs': initial_probs,
            'transition_probs': transition_probs
        }
        
        if durations is not None:
            result['durations'] = durations
        
        return result
        
    def _calculate_initial_probs(self) -> np.ndarray:
        initial_probs = np.zeros(self.analyzer.k)
        total_frames = 0
        
        for audio_data in self.analyzer.audio_files.values():
            for label in audio_data.labels:
                initial_probs[label] += 1
                total_frames += 1
                
        return initial_probs / total_frames
        
    def _calculate_transition_matrix(self) -> np.ndarray:
        k = self.analyzer.k
        transition_matrix = np.zeros((k, k))
        
        for audio_data in self.analyzer.audio_files.values():
            labels = audio_data.labels
            for i in range(len(labels) - 1):
                transition_matrix[labels[i], labels[i + 1]] += 1
                
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, where=row_sums!=0)
        return transition_probs
        
    def _generate_markov_sequence(self, initial_probs: np.ndarray,
                                 transition_probs: np.ndarray,
                                 duration_seconds: float):
        if self.duration_mode == DurationMode.FIXED:
            n_frames = int((duration_seconds * 1000) / self.analyzer.window_length_ms)
            
            sequence = []
            times = []
            current_state = np.random.choice(self.analyzer.k, p=initial_probs)
            
            for i in range(n_frames):
                sequence.append(current_state)
                times.append(i * self.analyzer.window_length_ms / 1000)
                if np.sum(transition_probs[current_state]) > 0:
                    current_state = np.random.choice(
                        self.analyzer.k, p=transition_probs[current_state]
                    )
                else:
                    current_state = np.random.choice(self.analyzer.k, p=initial_probs)
            
            return sequence, times
            
        elif self.duration_mode == DurationMode.CLUSTER_MEAN:
            mean_durations = self._calculate_cluster_durations()
            
            sequence = []
            times = []
            durations = []
            current_time = 0.0
            
            while current_time < duration_seconds:
                if not sequence:
                    current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                else:
                    if np.sum(transition_probs[sequence[-1]]) > 0:
                        current_state = np.random.choice(
                            self.analyzer.k, p=transition_probs[sequence[-1]]
                        )
                    else:
                        current_state = np.random.choice(self.analyzer.k, p=initial_probs)
                
                sequence.append(current_state)
                times.append(current_time)
                
                state_duration = mean_durations.get(current_state, 
                                                   self.analyzer.window_length_ms / 1000)
                durations.append(state_duration)
                current_time += state_duration
            
            return sequence, times, durations

    def _generate_audio_segments(self, sequence: List[int], synthesis_type: SynthesisType,
                                params: Dict, times: List[float] = None,
                                durations: List[float] = None) -> List[np.ndarray]:
        segments = []
        
        if self.duration_mode != DurationMode.FIXED and times is not None:
            try:
                segment = self._generate_audio_from_sequence(sequence, times, durations)
                segments.append(segment)
                return segments
            except Exception:
                pass
        
        for state in sequence:
            try:
                frames = self._collect_frames_for_state(state)
                if not frames:
                    continue

                if synthesis_type == SynthesisType.CONCATENATIVE:
                    if frames:
                        selected_frame = random.choice(frames)
                        segments.append(selected_frame)
                
                elif synthesis_type == SynthesisType.GRANULAR:
                    if frames:
                        self.processor.granular.grain_size = params.get('grain_size', 0.1)
                        self.processor.granular.density = params.get('density', 100)
                        self.processor.granular.pitch_shift = params.get('pitch_shift', 0)
                        self.processor.granular.position_jitter = params.get('position_jitter', 0.1)
                        self.processor.granular.duration_jitter = params.get('duration_jitter', 0.1)
                        
                        combined_frame = np.concatenate(frames)
                        segment = self.processor.granular.synthesize(
                            combined_frame,
                            duration=len(frames[0])/list(self.analyzer.audio_files.values())[0].sr
                        )
                        segments.append(segment)
                
                elif synthesis_type == SynthesisType.SPECTRAL:
                    if frames:
                        self.processor.spectral.fft_size = params.get('fft_size', 2048)
                        self.processor.spectral.preserve_transients = params.get('preserve_transients', True)
                        self.processor.spectral.spectral_stretch = params.get('spectral_stretch', 1.0)
                        
                        target_features = self._extract_target_features(frames)
                        combined_frame = np.concatenate(frames)
                        segment = self.processor.spectral.synthesize(combined_frame, target_features)
                        segments.append(segment)
                        
            except Exception:
                continue
        
        return segments if segments else [np.zeros(1024)]
    
    def _collect_frames_for_state(self, state: int) -> List[np.ndarray]:
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
                    
                    frame = frame / (np.max(np.abs(frame)) + 1e-10)
                    frames.append(frame)

        except Exception:
            pass
            
        return frames

    def _generate_audio_from_sequence(self, sequence: List[int], times: List[float] = None,
                                     durations: List[float] = None) -> np.ndarray:
        try:
            if self.duration_mode == DurationMode.FIXED or durations is None:
                first_audio = list(self.analyzer.audio_files.values())[0]
                frame_length = int(first_audio.sr * self.analyzer.window_length_ms / 1000)
                hop_length = frame_length // 2
                
                audio_segments = []
                for state in sequence:
                    frames_in_state = []
                    
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
                        audio_segments.append(np.zeros(frame_length))
                
                if not audio_segments:
                    raise ValueError("Nenhum segmento de áudio foi gerado")
                
                final_audio = self._apply_crossfade_to_segments(audio_segments)
                return final_audio
                
            else:
                first_audio = list(self.analyzer.audio_files.values())[0]
                sr = first_audio.sr
                
                if times is None:
                    raise ValueError("Parâmetro 'times' obrigatório para modos de duração variável")
                    
                total_samples = int(times[-1] * sr + durations[-1] * sr) + 1
                final_audio = np.zeros(total_samples)
                
                for i, (state, start_time, duration) in enumerate(zip(sequence, times, durations)):
                    frames_in_state = self._collect_frames_for_state(state)
                    
                    if not frames_in_state:
                        continue
                        
                    base_frame = random.choice(frames_in_state)
                    required_samples = int(duration * sr)
                    
                    if len(base_frame) > required_samples:
                        segment = base_frame[:required_samples]
                    else:
                        repetitions_needed = np.ceil(required_samples / len(base_frame))
                        if repetitions_needed <= 3:
                            repeated = np.tile(base_frame, int(repetitions_needed))
                            segment = repeated[:required_samples]
                        else:
                            try:
                                stretch_ratio = len(base_frame) / required_samples
                                segment = librosa.effects.time_stretch(base_frame, rate=stretch_ratio)
                                if len(segment) > required_samples:
                                    segment = segment[:required_samples]
                                elif len(segment) < required_samples:
                                    segment = np.pad(segment, (0, required_samples - len(segment)))
                            except Exception:
                                repeated = np.tile(base_frame, int(repetitions_needed))
                                segment = repeated[:required_samples]
                    
                    fade_samples = min(int(0.02 * sr), len(segment) // 4)
                    if fade_samples > 0:
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        segment[:fade_samples] *= fade_in
                        segment[-fade_samples:] *= fade_out
                    
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + len(segment)
                    
                    if end_sample > len(final_audio):
                        final_audio = np.pad(final_audio, (0, end_sample - len(final_audio)))
                    
                    final_audio[start_sample:end_sample] += segment
                
                if np.max(np.abs(final_audio)) > 0:
                    final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-10)
                
                return final_audio
                
        except Exception as e:
            st.error(f"Erro na geração de áudio: {str(e)}")
            return np.zeros(int(5 * first_audio.sr))
    
    def _apply_crossfade_to_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        fade_duration = 0.010
        sr = list(self.analyzer.audio_files.values())[0].sr
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

    def _extract_target_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        features = {
            'spectral_centroid': [],
            'spectral_bandwidth': [],
            'spectral_rolloff': [],
            'spectral_flatness': []
        }
        
        for frame in frames:
            sr = list(self.analyzer.audio_files.values())[0].sr
            features['spectral_centroid'].append(
                librosa.feature.spectral_centroid(y=frame, sr=sr).mean()
            )
            features['spectral_bandwidth'].append(
                librosa.feature.spectral_bandwidth(y=frame, sr=sr).mean()
            )
            features['spectral_rolloff'].append(
                librosa.feature.spectral_rolloff(y=frame, sr=sr).mean()
            )
            features['spectral_flatness'].append(
                librosa.feature.spectral_flatness(y=frame).mean()
            )
        
        return {k: np.mean(v) for k, v in features.items()}

# Interface Streamlit
def main():
    st.set_page_config(
        page_title="Sistema de Composição Musical Markoviano",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Sistema de Composição Musical baseado em Cadeias de Markov")
    st.markdown("---")
    
    # Inicialização do estado da sessão
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'tracks_generated' not in st.session_state:
        st.session_state.tracks_generated = False
    
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
        
        # Botão para iniciar análise
        if st.button("🚀 Iniciar Análise", type="primary"):
            try:
                with st.spinner("Analisando arquivos de áudio..."):
                    # Preparar dados dos arquivos
                    file_contents = []
                    for uploaded_file in uploaded_files:
                        file_contents.append((uploaded_file.name, uploaded_file.read()))
                    
                    # Criar e executar analisador
                    analyzer = MultiAudioAnalyzer(window_length_ms=window_length_ms)
                    analyzer.load_audio_files(file_contents)
                    analyzer.analyze_all_files(k=k_clusters)
                    
                    st.session_state.analyzer = analyzer
                    
                st.success("✅ Análise concluída com sucesso!")
                
                # Exibir informações da análise
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Arquivos processados", len(st.session_state.analyzer.audio_files))
                with col2:
                    st.metric("Clusters encontrados", st.session_state.analyzer.k)
                with col3:
                    total_features = sum(len(audio.features) for audio in st.session_state.analyzer.audio_files.values() if audio.features is not None)
                    st.metric("Características extraídas", total_features)
                
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {str(e)}")
    
    # Configurações de geração (só aparece após análise)
    if st.session_state.analyzer is not None:
        st.markdown("---")
        st.header("🎼 Configurações de Geração")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Parâmetros Básicos")
            num_tracks = st.slider("Número de tracks", 1, 10, 3)
            duration_seconds = st.slider("Duração (segundos)", 10, 300, 60)
            
            # Tipo de síntese
            synthesis_type_str = st.selectbox(
                "Tipo de síntese",
                ["Concatenative", "Granular", "Spectral"],
                help="Método de síntese de áudio"
            )
            
            synthesis_type = {
                "Concatenative": SynthesisType.CONCATENATIVE,
                "Granular": SynthesisType.GRANULAR,
                "Spectral": SynthesisType.SPECTRAL
            }[synthesis_type_str]
            
            # Modo de duração
            duration_mode_str = st.selectbox(
                "Modo de duração",
                ["Fixa", "Média do cluster", "Sequência"],
                help="Como controlar a duração dos segmentos"
            )
            
            duration_mode = {
                "Fixa": DurationMode.FIXED,
                "Média do cluster": DurationMode.CLUSTER_MEAN,
                "Sequência": DurationMode.SEQUENCE_LENGTH
            }[duration_mode_str]
        
        with col2:
            st.subheader("🎛️ Parâmetros de Síntese")
            synthesis_params = {}
            
            if synthesis_type == SynthesisType.CONCATENATIVE:
                synthesis_params['crossfade_duration'] = st.slider(
                    "Duração do crossfade (s)", 0.01, 0.5, 0.1, 0.01
                )
                
            elif synthesis_type == SynthesisType.GRANULAR:
                synthesis_params['grain_size'] = st.slider(
                    "Tamanho do grão (s)", 0.01, 0.5, 0.1, 0.01
                )
                synthesis_params['density'] = st.slider(
                    "Densidade (grãos/s)", 10, 500, 100, 10
                )
                synthesis_params['pitch_shift'] = st.slider(
                    "Mudança de pitch (semitons)", -12, 12, 0
                )
                synthesis_params['position_jitter'] = st.slider(
                    "Variação da posição", 0.0, 0.5, 0.1, 0.01
                )
                synthesis_params['duration_jitter'] = st.slider(
                    "Variação da duração", 0.0, 0.5, 0.1, 0.01
                )
                
            elif synthesis_type == SynthesisType.SPECTRAL:
                synthesis_params['preserve_transients'] = st.checkbox(
                    "Preservar transientes", True
                )
                synthesis_params['spectral_stretch'] = st.slider(
                    "Fator de esticamento", 0.1, 3.0, 1.0, 0.1
                )
                synthesis_params['fft_size'] = st.selectbox(
                    "Tamanho da FFT", [512, 1024, 2048, 4096], index=2
                )
        
        # Botão para gerar tracks
        if st.button("🎵 Gerar Composição", type="primary"):
            try:
                with st.spinner("Gerando composição..."):
                    generator = MarkovTrackGenerator(st.session_state.analyzer)
                    generator.set_duration_mode(duration_mode)
                    generator.generate_tracks(
                        num_tracks=num_tracks,
                        duration_seconds=duration_seconds,
                        synthesis_type=synthesis_type,
                        synthesis_params=synthesis_params
                    )
                    
                    st.session_state.generator = generator
                    st.session_state.tracks_generated = True
                
                st.success("✅ Composição gerada com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro durante a geração: {str(e)}")
    
    # Área de reprodução e download (só aparece após geração)
    if st.session_state.tracks_generated and st.session_state.generator is not None:
        st.markdown("---")
        st.header("🎧 Reprodução e Download")
        
        # Preparar arquivos para download
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar tracks individuais
            for i, track in enumerate(st.session_state.generator.tracks):
                # Obter taxa de amostragem
                sr = list(st.session_state.analyzer.audio_files.values())[0].sr
                
                # Salvar track individual
                track_buffer = io.BytesIO()
                sf.write(track_buffer, track['audio'], sr, format='WAV')
                track_buffer.seek(0)
                zip_file.writestr(f'track_{i+1}.wav', track_buffer.getvalue())
                
                # Exibir player para cada track
                st.subheader(f"🎵 Track {i+1}")
                st.audio(track_buffer.getvalue(), format='audio/wav')
            
            # Criar mix final
            try:
                max_length = max(len(track['audio']) for track in st.session_state.generator.tracks)
                normalized_audio = []
                
                for track in st.session_state.generator.tracks:
                    audio = track['audio']
                    if len(audio) < max_length:
                        audio = np.pad(audio, (0, max_length - len(audio)))
                    normalized_audio.append(audio)
                
                final_mix = np.sum(normalized_audio, axis=0)
                if np.max(np.abs(final_mix)) > 0:
                    final_mix = final_mix / np.max(np.abs(final_mix))
                
                # Salvar mix final
                mix_buffer = io.BytesIO()
                sf.write(mix_buffer, final_mix, sr, format='WAV')
                mix_buffer.seek(0)
                zip_file.writestr('final_mix.wav', mix_buffer.getvalue())
                
                # Exibir player do mix final
                st.subheader("🎼 Mix Final")
                st.audio(mix_buffer.getvalue(), format='audio/wav')
                
            except Exception as e:
                st.error(f"Erro ao criar mix final: {str(e)}")
        
        zip_buffer.seek(0)
        
        # Botão de download
        st.download_button(
            label="📥 Download de Todos os Arquivos (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"markov_composition_{int(time.time())}.zip",
            mime="application/zip"
        )
        
        # Visualizações
        st.markdown("---")
        st.header("📊 Visualizações")
        
        try:
            # Gráfico de distribuição de estados
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Distribuição de clusters
            all_labels = []
            for audio_data in st.session_state.analyzer.audio_files.values():
                if audio_data.labels is not None:
                    all_labels.extend(audio_data.labels)
            
            if all_labels:
                axes[0, 0].hist(all_labels, bins=st.session_state.analyzer.k, alpha=0.7)
                axes[0, 0].set_title('Distribuição dos Clusters')
                axes[0, 0].set_xlabel('Cluster')
                axes[0, 0].set_ylabel('Frequência')
            
            # Plot 2: Sequência da primeira track
            if st.session_state.generator.tracks:
                track = st.session_state.generator.tracks[0]
                axes[0, 1].plot(track['times'], track['sequence'])
                axes[0, 1].set_title('Sequência de Estados - Track 1')
                axes[0, 1].set_xlabel('Tempo (s)')
                axes[0, 1].set_ylabel('Estado')
            
            # Plot 3: Matriz de transição
            if st.session_state.generator.tracks:
                transition_matrix = st.session_state.generator.tracks[0]['transition_probs']
                im = axes[1, 0].imshow(transition_matrix, cmap='Blues')
                axes[1, 0].set_title('Matriz de Transição')
                axes[1, 0].set_xlabel('Para Estado')
                axes[1, 0].set_ylabel('De Estado')
                plt.colorbar(im, ax=axes[1, 0])
            
            # Plot 4: PCA dos features
            if st.session_state.analyzer.audio_files:
                all_features = []
                all_labels = []
                
                for audio_data in st.session_state.analyzer.audio_files.values():
                    if audio_data.features is not None and audio_data.labels is not None:
                        all_features.append(audio_data.features)
                        all_labels.extend(audio_data.labels)
                
                if all_features:
                    features_combined = np.vstack(all_features)
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(features_combined)
                    
                    scatter = axes[1, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                               c=all_labels, cmap='viridis', alpha=0.6)
                    axes[1, 1].set_title('Visualização PCA dos Features')
                    axes[1, 1].set_xlabel('PC1')
                    axes[1, 1].set_ylabel('PC2')
                    plt.colorbar(scatter, ax=axes[1, 1])
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro ao gerar visualizações: {str(e)}")
    
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
        
        **Modos de Duração:**
        - **Fixa**: Duração baseada no tamanho da janela
        - **Média do cluster**: Duração baseada na média de cada cluster
        - **Sequência**: Duração baseada em sequências consecutivas
        """)

if __name__ == "__main__":
    main() 