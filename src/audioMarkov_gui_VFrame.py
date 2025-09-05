#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interface Gráfica Aprimorada para o Sistema de Composição Musical Markoviana
com reprodução de áudio e visualizações gráficas - Versão Corrigida
"""
import os
import sys
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import json
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import uuid
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import pandas as pd

# Configure o logger
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Iniciar a aplicação
logger.debug("Iniciando aplicação")

# Importe suas classes principais - ajuste o caminho conforme necessário
try:
    from script11_Markov_Audio1GeraCompMult_GUI import (
        SynthesisType, DurationMode, MultiAudioAnalyzer, MarkovTrackGenerator
    )
except ImportError:
    # Mensagem para instruir o usuário a configurar o path
    print("Erro: Não foi possível importar o módulo do sistema de composição Markoviana.")
    print("Por favor, certifique-se de que o arquivo script11_Markov_Audio1GeraCompMult_GUI.py")
    print("está no mesmo diretório ou ajuste a importação conforme necessário.")
    sys.exit(1)

# Inicializa o mixer pygame para reprodução de áudio
pygame.mixer.init()

class AudioPlayer:
    """Classe para gerenciar reprodução de áudio."""
    
    def __init__(self):
        self.current_audio = None
        self.playing = False
        self.paused = False
        
    def load_audio(self, audio_path: str) -> bool:
        """Carrega um arquivo de áudio."""
        try:
            pygame.mixer.music.load(audio_path)
            self.current_audio = audio_path
            self.playing = False
            self.paused = False
            return True
        except Exception as e:
            print(f"Erro ao carregar áudio: {str(e)}")
            return False
            
    def play(self) -> None:
        """Inicia a reprodução."""
        if self.current_audio:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
            else:
                pygame.mixer.music.play()
            self.playing = True
            
    def pause(self) -> None:
        """Pausa a reprodução."""
        if self.playing:
            pygame.mixer.music.pause()
            self.playing = False
            self.paused = True
            
    def stop(self) -> None:
        """Interrompe a reprodução."""
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        
    def is_playing(self) -> bool:
        """Verifica se está tocando."""
        return self.playing

class MarkovGUIImproved:
    """Interface gráfica aprimorada para o Sistema de Composição Musical Markoviano."""
    
    def __init__(self, root: tk.Tk):
        # Configuração da janela principal
        self.root = root
        self.root.title("Sistema de Composição Musical Markoviano")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variáveis de controle
        self.audio_files = []
        self.output_folder = None
        self.analyzer = None
        self.generator = None
        self.generated_tracks = []
        self.current_step = 0
        
        # Player de áudio
        self.audio_player = AudioPlayer()
        
        # Variáveis para os parâmetros
        self.setup_variables()
        
        # Interface principal
        self.create_main_interface()
        
        # Polling para atualizar o estado do player
        self._update_player_state()
        
    def setup_variables(self):
        """Configura as variáveis de controle para os parâmetros."""
        # Variáveis de análise
        self.window_length_var = tk.StringVar(value="500")
        self.cluster_mode_var = tk.StringVar(value="auto")
        self.k_clusters_var = tk.StringVar(value="5")
        
        # Variáveis de geração
        self.duration_var = tk.StringVar(value="60")
        self.tracks_var = tk.StringVar(value="3")
        self.synthesis_type_var = tk.StringVar(value="concatenative")
        self.duration_mode_var = tk.StringVar(value="fixed")
        
        # Variáveis de síntese concatenativa
        self.crossfade_duration_var = tk.StringVar(value="0.05")
        
        # Variáveis de síntese granular
        self.grain_size_var = tk.StringVar(value="0.1")
        self.density_var = tk.StringVar(value="100")
        self.pitch_shift_var = tk.StringVar(value="0")
        self.position_jitter_var = tk.StringVar(value="0.1")
        self.duration_jitter_var = tk.StringVar(value="0.1")
        
        # Variáveis de síntese espectral
        self.preserve_transients_var = tk.BooleanVar(value=True)
        self.spectral_stretch_var = tk.StringVar(value="1.0")
        self.fft_size_var = tk.StringVar(value="2048")
        
        # Outras configurações
        self.normalize_var = tk.BooleanVar(value=True)
        self.adjust_method_var = tk.StringVar(value="stretch")
        
    def create_main_interface(self):
        """Cria a interface principal com sistema de navegação e verificações adicionais."""
        print("Criando interface principal...")
        
        # Container principal
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Container do conteúdo
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Inicializa os frames para cada etapa
        self.frames = {}
        self.initialize_frames()
        
        # Barra de navegação na parte inferior
        self.create_navigation_bar()
        
        # Mostra o primeiro frame
        print("Exibindo o primeiro frame (análise)...")
        self.show_frame("step1_analysis")
        
        # Força atualização da interface
        self.root.update_idletasks()
        print(f"Frame step1_analysis visível: {self._is_frame_visible('step1_analysis')}")

    def reconstruct_player_interface(self):
        """Reconstrói completamente a interface do player de forma segura."""
        print("Reconstruindo completamente a interface do player...")
        
        try:
            # 1. Salva referências importantes
            output_folder = self.output_folder
            audio_player = self.audio_player
            
            # 2. Limpa completamente o content_frame
            for widget in self.content_frame.winfo_children():
                widget.destroy()
            
            # 3. Recria o frame do player do zero
            new_player_frame = self.create_player_frame()
            self.frames["step3_player"] = new_player_frame
            
            # 4. Mostra o novo frame
            new_player_frame.pack(in_=self.content_frame, fill=tk.BOTH, expand=True)
            
            # 5. Restaura referências
            self.output_folder = output_folder
            self.audio_player = audio_player
            
            # 6. Atualiza o estado de navegação
            self.current_step = 3
            self.update_navigation_state()
            
            # 7. Atualiza a lista de áudio depois que a interface estiver visível
            self.root.after(100, self.update_audio_list)
            
            # 8. Força atualização da interface
            self.root.update_idletasks()
            
            print("Reconstrução da interface do player concluída")
            return True
        except Exception as e:
            print(f"Erro durante reconstrução da interface: {str(e)}")
            import traceback
            traceback.print_exc()
            return False        
        
    def initialize_frames(self):
        """Inicializa os frames para cada etapa do processo com verificações adicionais."""
        print("Inicializando frames para cada etapa...")
        
        # Frame 1: Seleção e Análise
        print("Criando frame de análise...")
        self.frames["step1_analysis"] = self.create_analysis_frame()
        
        # Frame 2: Configuração e Geração
        print("Criando frame de geração...")
        self.frames["step2_generation"] = self.create_generation_frame()
        
        # Frame 3: Player e Visualização
        print("Criando frame de player...")
        self.frames["step3_player"] = self.create_player_frame()
        
        # Frame 4: Visualizações e Análises Avançadas
        print("Criando frame de visualização...")
        self.frames["step4_visualization"] = self.create_visualization_frame()
        
        # Verifica se todos os frames foram criados corretamente
        expected_frames = ["step1_analysis", "step2_generation", "step3_player", "step4_visualization"]
        for frame_name in expected_frames:
            if frame_name not in self.frames:
                print(f"ERRO: Frame {frame_name} não foi criado corretamente!")
            elif self.frames[frame_name] is None:
                print(f"ERRO: Frame {frame_name} foi criado como None!")
            else:
                print(f"Frame {frame_name} criado com sucesso")
        
        # Adiciona os frames ao container principal, mas esconde todos inicialmente
        for frame_name, frame in self.frames.items():
            if frame is not None:
                # Adiciona ao container principal
                frame.pack(in_=self.content_frame, fill=tk.BOTH, expand=True)
                # E então esconde
                frame.pack_forget()
                print(f"Frame {frame_name} adicionado ao container principal e escondido")
            else:
                print(f"ERRO: Não foi possível adicionar frame {frame_name} (None)")
 
    def show_frame(self, frame_name):
        """Mostra o frame selecionado e esconde os outros com verificações extras e recuperação de erros."""
        # Verifica se o frame existe
        if frame_name not in self.frames:
            print(f"ERRO: Tentando mostrar frame inexistente: {frame_name}")
            return
            
        if self.frames[frame_name] is None:
            print(f"ERRO: O frame {frame_name} é None!")
            return
            
        try:
            # Esconde todos os frames
            for name, frame in self.frames.items():
                if frame is not None and frame.winfo_ismapped():
                    print(f"Escondendo frame: {name}")
                    frame.pack_forget()
            
            # Mostra o frame selecionado
            frame = self.frames[frame_name]
            frame.pack(in_=self.content_frame, fill=tk.BOTH, expand=True)
            print(f"Frame {frame_name} exibido: {frame.winfo_ismapped()}")
            
            # Força atualização da interface
            self.root.update_idletasks()
            
            # Verifica se o frame está realmente visível
            if not frame.winfo_ismapped():
                print(f"ALERTA: Frame {frame_name} não ficou visível após exibição!")
                # Tenta reparar
                self.fix_empty_interface()
        
        except Exception as e:
            print(f"ERRO ao mostrar frame {frame_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Tenta recuperar a interface
            self.fix_empty_interface()
        
    def create_navigation_bar(self):
        """Cria a barra de navegação inferior."""
        nav_frame = ttk.Frame(self.main_container)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Botões de navegação
        self.prev_button = ttk.Button(nav_frame, text="← Anterior", 
                                      command=self.go_to_previous_step)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        
        # Indicador de etapa atual
        self.step_indicator = ttk.Label(nav_frame, text="Etapa 1/4: Análise de Arquivos")
        self.step_indicator.pack(side=tk.LEFT, expand=True)
        
        self.next_button = ttk.Button(nav_frame, text="Próximo →", 
                                      command=self.go_to_next_step)
        self.next_button.pack(side=tk.RIGHT, padx=10)
        
        # Estado inicial
        self.current_step = 1
        self.update_navigation_state()
    
    def update_navigation_state(self):
        """Atualiza o estado dos botões de navegação."""
        # Atualiza o texto do indicador
        step_texts = {
            1: "Etapa 1/4: Análise de Arquivos",
            2: "Etapa 2/4: Configuração e Geração",
            3: "Etapa 3/4: Reprodução de Áudio",
            4: "Etapa 4/4: Visualizações e Análises"
        }
        self.step_indicator.config(text=step_texts[self.current_step])
        
        # Habilita/desabilita botões conforme necessário
        self.prev_button.config(state=tk.NORMAL if self.current_step > 1 else tk.DISABLED)
        
        # O botão "Próximo" é desabilitado se não completamos a etapa atual
        if self.current_step == 1 and not self.analyzer:
            self.next_button.config(state=tk.DISABLED)
        elif self.current_step == 2 and not self.output_folder:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL if self.current_step < 4 else tk.DISABLED)
    
    def go_to_previous_step(self):
        """Navega para a etapa anterior."""
        if self.current_step > 1:
            self.current_step -= 1
            frame_name = f"step{self.current_step}_" + ["analysis", "generation", "player", "visualization"][self.current_step-1]
            self.show_frame(frame_name)
    
    def go_to_next_step(self):
        """Navega para a próxima etapa com verificação aprimorada e recuperação de erros."""
        if self.current_step < 4:
            try:
                # Registra a etapa atual para depuração
                prev_step = self.current_step
                next_step = self.current_step + 1
                
                print(f"Navegando da etapa {prev_step} para etapa {next_step}")
                
                # Determina o nome do frame para a nova etapa
                frame_names = ["analysis", "generation", "player", "visualization"]
                frame_name = f"step{next_step}_{frame_names[next_step-1]}"
                
                # Verifica se o frame existe
                if frame_name not in self.frames:
                    print(f"ERRO: Frame '{frame_name}' não encontrado!")
                    return
                
                if self.frames[frame_name] is None:
                    print(f"ERRO: Frame '{frame_name}' é None!")
                    return
                
                # Se estamos indo para a etapa de player, prepara os dados primeiro
                if next_step == 3 and self.output_folder:
                    print("Preparando dados do player...")
                    # Força a atualização da etapa ANTES de mostrar o frame
                    self.current_step = next_step
                    self.update_navigation_state()
                    
                    # Mostra o frame do player
                    self.show_frame(frame_name)
                    
                    # Atualiza a lista de áudio DEPOIS de mostrar o frame
                    self.update_audio_list()
                
                # Se estamos indo para a etapa de visualização, prepara os dados primeiro    
                elif next_step == 4 and self.output_folder:
                    print("Preparando dados de visualização...")
                    # Atualiza a etapa
                    self.current_step = next_step
                    self.update_navigation_state()
                    
                    # Mostra o frame e depois atualiza os dados
                    self.show_frame(frame_name)
                    self._update_track_visualization_list()
                
                # Para outras transições
                else:
                    # Atualiza a etapa
                    self.current_step = next_step
                    self.update_navigation_state()
                    
                    # Mostra o frame
                    self.show_frame(frame_name)
                
                # Verifica se a transição foi bem-sucedida
                if not self._is_frame_visible(frame_name):
                    print(f"ALERTA: O frame {frame_name} não está visível após a transição!")
                    self.fix_empty_interface()
            
            except Exception as e:
                print(f"ERRO durante navegação: {str(e)}")
                import traceback
                traceback.print_exc()
                # Em caso de erro, tenta reparar a interface
                self.fix_empty_interface()

    def _is_frame_visible(self, frame_name):
        """Verifica se um frame está visível na interface de forma mais confiável."""
        if frame_name not in self.frames:
            print(f"Frame {frame_name} não existe!")
            return False
        
        try:
            frame = self.frames[frame_name]
            if frame is None:
                print(f"Frame {frame_name} é None!")
                return False
                
            # Métodos para verificar visibilidade
            is_mapped = frame.winfo_ismapped()
            is_viewable = frame.winfo_viewable()
            parent_visible = self.content_frame.winfo_ismapped() and self.content_frame.winfo_viewable()
            
            # Verifica também se o frame está contido no content_frame
            is_child = frame.master == self.content_frame
            
            print(f"Verificação detalhada de {frame_name}: mapped={is_mapped}, viewable={is_viewable}, parent_visible={parent_visible}, is_child={is_child}")
            
            return is_mapped and is_viewable and parent_visible
        except Exception as e:
            print(f"Erro ao verificar visibilidade do frame {frame_name}: {str(e)}")
            return False
    
    def fix_empty_interface(self):
        """Tenta reparar a interface quando ela fica vazia após transições."""
        print("Tentando reparar a interface...")
        
        # Verifica qual frame deveria estar visível
        if self.current_step < 1 or self.current_step > 4:
            print(f"Etapa inválida: {self.current_step}, redefinindo para 1")
            self.current_step = 1
        
        frame_names = ["analysis", "generation", "player", "visualization"]
        current_frame_name = f"step{self.current_step}_{frame_names[self.current_step-1]}"
        
        try:
            # Verifica estado da interface principal
            if not self.main_container.winfo_ismapped():
                print("ALERTA: Container principal não está visível!")
                
                # Recria a estrutura principal
                self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                self.content_frame.pack(in_=self.main_container, fill=tk.BOTH, expand=True)
            
            # Verifica se todos os frames existem
            for name, frame in self.frames.items():
                if frame is None:
                    print(f"ALERTA: Frame {name} é None, recriando...")
                    # Tenta recriar o frame faltante
                    if "analysis" in name:
                        self.frames[name] = self.create_analysis_frame()
                    elif "generation" in name:
                        self.frames[name] = self.create_generation_frame()
                    elif "player" in name:
                        self.frames[name] = self.create_player_frame()
                    elif "visualization" in name:
                        self.frames[name] = self.create_visualization_frame()
                    
                    # Adiciona o frame recriado ao content_frame
                    if self.frames[name] is not None:
                        self.frames[name].pack(in_=self.content_frame, fill=tk.BOTH, expand=True)
                        self.frames[name].pack_forget()  # Esconde inicialmente
            
            # Se o frame atual não estiver visível, força a exibição
            if not self._is_frame_visible(current_frame_name):
                print(f"Frame {current_frame_name} não está visível. Forçando exibição...")
                
                # Esconde todos os frames primeiro
                for name, frame in self.frames.items():
                    if frame is not None and frame.winfo_ismapped():
                        frame.pack_forget()
                
                # Mostra o frame correto
                frame = self.frames[current_frame_name]
                if frame is not None:
                    frame.pack(in_=self.content_frame, fill=tk.BOTH, expand=True)
                    
                    # Força atualização
                    self.root.update_idletasks()
                    
                    # Se o frame reparado for o player, atualiza a lista de áudio
                    if current_frame_name == "step3_player":
                        print("Atualizando lista de áudio após reparo...")
                        self.root.after(100, self.update_audio_list)
                    
                    # Se o frame reparado for a visualização, atualiza as visualizações
                    elif current_frame_name == "step4_visualization":
                        self._update_track_visualization_list()
                        if hasattr(self, 'track_viz_combo') and self.track_viz_combo.get():
                            self.update_visualizations()
            
            # Verifica estado final
            if not self._is_frame_visible(current_frame_name):
                print(f"FALHA: Reparo não conseguiu exibir {current_frame_name}")
                # Medida drástica: reconstruir interface para o player
                if current_frame_name == "step3_player":
                    self.reconstruct_player_interface()
            else:
                print(f"Reparo concluído com sucesso. Frame {current_frame_name} está visível.")
                
        except Exception as e:
            print(f"Erro durante reparo da interface: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_analysis_frame(self):
        """Cria o frame para seleção de arquivos e análise."""
        frame = ttk.Frame(self.content_frame, padding="10")
        
        # Título
        ttk.Label(frame, text="Análise de Arquivos de Áudio", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Frame para seleção de arquivos
        file_frame = ttk.LabelFrame(frame, text="Seleção de Arquivos", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # Botões para adicionar/remover arquivos
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Adicionar Arquivos", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Limpar Seleção", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        
        # Lista de arquivos selecionados
        ttk.Label(file_frame, text="Arquivos Selecionados:").pack(anchor="w", pady=(10, 0))
        
        # Frame para a lista e scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.files_listbox = tk.Listbox(list_frame, height=8, selectmode=tk.EXTENDED)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        files_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Parâmetros de Análise
        params_frame = ttk.LabelFrame(frame, text="Parâmetros de Análise", padding="10")
        params_frame.pack(fill=tk.X, pady=10)
        
        # Tamanho da janela
        window_frame = ttk.Frame(params_frame)
        window_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(window_frame, text="Tamanho da Janela (ms):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(window_frame, textvariable=self.window_length_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(window_frame, text="(Valores recomendados: 100-500ms)").pack(side=tk.LEFT, padx=5)
        
        # Número de clusters
        clusters_frame = ttk.Frame(params_frame)
        clusters_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(clusters_frame, text="Detecção automática de clusters", 
                        variable=self.cluster_mode_var, value="auto").pack(anchor="w", padx=5)
        
        manual_frame = ttk.Frame(clusters_frame)
        manual_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(manual_frame, text="Definir número de clusters manualmente:", 
                       variable=self.cluster_mode_var, value="manual").pack(side=tk.LEFT, padx=5)
        
        ttk.Entry(manual_frame, textvariable=self.k_clusters_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Botão para iniciar análise
        ttk.Button(frame, text="Iniciar Análise", command=self.run_analysis).pack(pady=20)
        
        # Progresso e status
        progress_frame = ttk.Frame(frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.analysis_progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="indeterminate")
        self.analysis_progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(pady=5)
        
        return frame
        
    def create_generation_frame(self):
        """Cria o frame para geração e configuração."""
        frame = ttk.Frame(self.content_frame, padding="10")
        
        # Título
        ttk.Label(frame, text="Configuração e Geração de Composições", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Criar duas colunas
        columns_frame = ttk.Frame(frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Coluna esquerda
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Coluna direita
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # === COLUNA ESQUERDA ===
        
        # Parâmetros da Composição
        comp_frame = ttk.LabelFrame(left_column, text="Parâmetros da Composição", padding="10")
        comp_frame.pack(fill=tk.X, pady=10)
        
        # Duração
        duration_frame = ttk.Frame(comp_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(duration_frame, text="Duração (segundos):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(duration_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Número de tracks
        tracks_frame = ttk.Frame(comp_frame)
        tracks_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(tracks_frame, text="Número de Tracks:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(tracks_frame, textvariable=self.tracks_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Método de Síntese
        synthesis_frame = ttk.LabelFrame(left_column, text="Método de Síntese", padding="10")
        synthesis_frame.pack(fill=tk.X, pady=10)
        
        # Frame para Concatenativa
        concat_frame = ttk.Frame(synthesis_frame)
        concat_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(concat_frame, text="Concatenativa (com crossfade)", 
                       variable=self.synthesis_type_var, value="concatenative",
                       command=self.update_synthesis_params).pack(side=tk.LEFT, padx=5)
        
        # Frame para Granular
        granular_frame = ttk.Frame(synthesis_frame)
        granular_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(granular_frame, text="Granular", 
                       variable=self.synthesis_type_var, value="granular",
                       command=self.update_synthesis_params).pack(side=tk.LEFT, padx=5)
        
        # Frame para Espectral
        spectral_frame = ttk.Frame(synthesis_frame)
        spectral_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(spectral_frame, text="Espectral", 
                       variable=self.synthesis_type_var, value="spectral",
                       command=self.update_synthesis_params).pack(side=tk.LEFT, padx=5)
        
        # Parâmetros específicos de cada síntese
        self.synthesis_params_frame = ttk.LabelFrame(left_column, text="Parâmetros da Síntese", padding="10")
        self.synthesis_params_frame.pack(fill=tk.X, pady=10)
        
        # Inicialmente configurado para concatenativa
        self.create_concatenative_params()
        
        # === COLUNA DIREITA ===
        
        # Modo de Duração
        duration_mode_frame = ttk.LabelFrame(right_column, text="Modo de Duração", padding="10")
        duration_mode_frame.pack(fill=tk.X, pady=10)
        
        ttk.Radiobutton(duration_mode_frame, text="Fixa (baseada no tamanho da janela)", 
                       variable=self.duration_mode_var, value="fixed").pack(anchor="w", padx=5, pady=2)
        
        ttk.Radiobutton(duration_mode_frame, text="Baseada na duração média de cada cluster", 
                       variable=self.duration_mode_var, value="cluster_mean").pack(anchor="w", padx=5, pady=2)
        
        ttk.Radiobutton(duration_mode_frame, text="Baseada em sequências consecutivas do mesmo estado", 
                       variable=self.duration_mode_var, value="sequence").pack(anchor="w", padx=5, pady=2)
        
        # Configurações adicionais
        add_config_frame = ttk.LabelFrame(right_column, text="Configurações Adicionais", padding="10")
        add_config_frame.pack(fill=tk.X, pady=10)
        
        # Normalização
        norm_frame = ttk.Frame(add_config_frame)
        norm_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(norm_frame, text="Normalizar áudio final", 
                       variable=self.normalize_var).pack(anchor="w", padx=5)
        
        # Método de ajuste de duração
        adjust_frame = ttk.Frame(add_config_frame)
        adjust_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(adjust_frame, text="Método de ajuste de duração:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(adjust_frame, textvariable=self.adjust_method_var, 
                    values=["stretch", "loop"], width=10).pack(side=tk.LEFT, padx=5)
        
        # Botão para gerar as tracks
        ttk.Button(right_column, text="Gerar Composição", command=self.generate_tracks).pack(pady=20)
        
        # Progresso e status
        progress_frame = ttk.Frame(right_column)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.generation_progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="indeterminate")
        self.generation_progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.generation_status_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.generation_status_var).pack(pady=5)
        
        return frame
    
    def create_player_frame(self):
        """Cria o frame de player de áudio."""
        frame = ttk.Frame(self.content_frame, padding="10")
        
        # Título
        ttk.Label(frame, text="Reprodução de Áudio", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Criar duas colunas
        columns_frame = ttk.Frame(frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Coluna esquerda - Lista de tracks
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)
        
        # Coluna direita - Player e informações
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        
        # === COLUNA ESQUERDA ===
        # Frame de seleção de áudio
        selection_frame = ttk.LabelFrame(left_column, text="Selecione o áudio para reprodução")
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Lista de áudios disponíveis com rótulo
        ttk.Label(selection_frame, text="Faixas disponíveis:").pack(anchor="w", padx=5, pady=5)
        
        tracks_frame = ttk.Frame(selection_frame)
        tracks_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.track_list = tk.Listbox(tracks_frame, height=15, font=("Arial", 10))
        self.track_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.track_list.bind('<<ListboxSelect>>', lambda event: self.load_selected_audio())
        
        scrollbar = ttk.Scrollbar(tracks_frame, orient="vertical", command=self.track_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.track_list.config(yscrollcommand=scrollbar.set)
        
        # Botões de ação
        action_frame = ttk.Frame(selection_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Carregar Áudio", command=self.load_selected_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Atualizar Lista", command=self.update_audio_list).pack(side=tk.LEFT, padx=5)
        
        # === COLUNA DIREITA ===
        # Informações da pasta de saída
        output_frame = ttk.LabelFrame(right_column, text="Pasta de Saída")
        output_frame.pack(fill=tk.X, pady=10)
        
        self.output_path_var = tk.StringVar(value="Nenhuma pasta selecionada")
        ttk.Label(output_frame, textvariable=self.output_path_var, wraplength=600).pack(pady=5)
        
        ttk.Button(output_frame, text="Abrir Pasta", command=self.open_output_folder).pack(pady=5)
        
        # Player de áudio
        player_frame = ttk.LabelFrame(right_column, text="Controles de Reprodução")
        player_frame.pack(fill=tk.X, pady=10)
        
        # Informações de reprodução
        info_frame = ttk.Frame(player_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(info_frame, text="Reproduzindo:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.playing_label = ttk.Label(info_frame, text="Nenhum arquivo", font=("Arial", 12))
        self.playing_label.pack(side=tk.LEFT, padx=5)
        
        # Controles do player
        controls_frame = ttk.Frame(player_frame)
        controls_frame.pack(fill=tk.X, pady=20)
        
        # Botões maiores e com ícones
        style = ttk.Style()
        style.configure('Player.TButton', font=('Arial', 14))
        
        play_btn = ttk.Button(controls_frame, text="▶ Play", command=self.play_audio, style='Player.TButton', width=10)
        play_btn.pack(side=tk.LEFT, padx=20)
        
        pause_btn = ttk.Button(controls_frame, text="⏸ Pause", command=self.pause_audio, style='Player.TButton', width=10)
        pause_btn.pack(side=tk.LEFT, padx=20)
        
        stop_btn = ttk.Button(controls_frame, text="⏹ Stop", command=self.stop_audio, style='Player.TButton', width=10)
        stop_btn.pack(side=tk.LEFT, padx=20)
        
        # Status do player
        self.player_status = ttk.Label(player_frame, text="", font=("Arial", 10, "italic"))
        self.player_status.pack(pady=10)
        
        return frame
        
    def create_visualization_frame(self):
        """Cria o frame para visualizações e análises avançadas."""
        frame = ttk.Frame(self.content_frame, padding="10")
        
        # Título
        ttk.Label(frame, text="Visualizações e Análises", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Container para abas de visualização
        viz_container = ttk.Notebook(frame)
        viz_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Aba 1: Visão Geral dos Clusters
        tab_clusters = ttk.Frame(viz_container)
        viz_container.add(tab_clusters, text="Visão Geral dos Clusters")
        
        # Aba 2: Análise das Tracks
        tab_tracks = ttk.Frame(viz_container)
        viz_container.add(tab_tracks, text="Análise das Tracks")
        
        # Aba 3: Matriz de Transição
        tab_transition = ttk.Frame(viz_container)
        viz_container.add(tab_transition, text="Matriz de Transição")
        
        # Aba 4: Espectrogramas
        tab_spectrograms = ttk.Frame(viz_container)
        viz_container.add(tab_spectrograms, text="Espectrogramas")
        
        # Container para o gráfico na aba de clusters
        self.cluster_plot_frame = ttk.Frame(tab_clusters)
        self.cluster_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Container para o gráfico na aba de tracks
        self.track_plot_frame = ttk.Frame(tab_tracks)
        self.track_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Container para o gráfico na aba de matriz de transição
        self.transition_plot_frame = ttk.Frame(tab_transition)
        self.transition_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Container para o gráfico na aba de espectrogramas
        self.spectro_plot_frame = ttk.Frame(tab_spectrograms)
        self.spectro_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controles de seleção para visualizações
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Dropdown para selecionar a track a visualizar
        track_selector_frame = ttk.Frame(control_frame)
        track_selector_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(track_selector_frame, text="Selecionar Track:").pack(side=tk.LEFT, padx=5)
        self.track_viz_var = tk.StringVar()
        self.track_viz_combo = ttk.Combobox(track_selector_frame, textvariable=self.track_viz_var, width=15)
        self.track_viz_combo.pack(side=tk.LEFT, padx=5)
        self.track_viz_combo.bind("<<ComboboxSelected>>", self.update_visualizations)
        
        # Botão para atualizar visualizações
        ttk.Button(control_frame, text="Atualizar Visualizações", 
                  command=self.update_visualizations).pack(side=tk.RIGHT, padx=20)
        
        return frame
    
    def create_concatenative_params(self):
        """Cria campos para parâmetros da síntese concatenativa."""
        # Limpa o frame
        for widget in self.synthesis_params_frame.winfo_children():
            widget.destroy()
            
        # Crossfade
        crossfade_frame = ttk.Frame(self.synthesis_params_frame)
        crossfade_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(crossfade_frame, text="Duração do crossfade (segundos):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(crossfade_frame, textvariable=self.crossfade_duration_var, width=10).pack(side=tk.LEFT, padx=5)
    
    def create_granular_params(self):
        """Cria campos para parâmetros da síntese granular."""
        # Limpa o frame
        for widget in self.synthesis_params_frame.winfo_children():
            widget.destroy()
            
        # Layout em grid para melhor organização
        params_grid = ttk.Frame(self.synthesis_params_frame)
        params_grid.pack(fill=tk.X, pady=5)
        
        # Tamanho do grão
        ttk.Label(params_grid, text="Tamanho do grão (segundos):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_grid, textvariable=self.grain_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Densidade
        ttk.Label(params_grid, text="Densidade (grãos/segundo):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_grid, textvariable=self.density_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Pitch shift
        ttk.Label(params_grid, text="Mudança de pitch (semitons):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_grid, textvariable=self.pitch_shift_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Position jitter
        ttk.Label(params_grid, text="Variação da posição (0-1):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_grid, textvariable=self.position_jitter_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Duration jitter
        ttk.Label(params_grid, text="Variação da duração (0-1):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_grid, textvariable=self.duration_jitter_var, width=10).grid(row=1, column=3, padx=5, pady=5)
    
    def create_spectral_params(self):
        """Cria campos para parâmetros da síntese espectral."""
        # Limpa o frame
        for widget in self.synthesis_params_frame.winfo_children():
            widget.destroy()
            
        # Preserve transients
        transients_frame = ttk.Frame(self.synthesis_params_frame)
        transients_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(transients_frame, text="Preservar transientes", 
                       variable=self.preserve_transients_var).pack(anchor="w", padx=5)
        
        # Spectral stretch
        stretch_frame = ttk.Frame(self.synthesis_params_frame)
        stretch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stretch_frame, text="Fator de esticamento espectral:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(stretch_frame, textvariable=self.spectral_stretch_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # FFT size
        fft_frame = ttk.Frame(self.synthesis_params_frame)
        fft_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fft_frame, text="Tamanho da FFT:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(fft_frame, textvariable=self.fft_size_var, width=10).pack(side=tk.LEFT, padx=5)
    
    def update_synthesis_params(self):
        """Atualiza os campos de parâmetros com base no tipo de síntese selecionado."""
        synthesis_type = self.synthesis_type_var.get()
        
        if synthesis_type == "concatenative":
            self.create_concatenative_params()
        elif synthesis_type == "granular":
            self.create_granular_params()
        elif synthesis_type == "spectral":
            self.create_spectral_params()
    
    def add_files(self):
        """Adiciona arquivos de áudio à seleção."""
        files = filedialog.askopenfilenames(
            title="Selecione os arquivos de áudio",
            filetypes=[("Arquivos de Áudio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif")]
        )
        
        if not files:
            return
            
        for file in files:
            if file not in self.audio_files:
                self.audio_files.append(file)
                self.files_listbox.insert(tk.END, os.path.basename(file))
                
        self.status_var.set(f"{len(self.audio_files)} arquivos selecionados")
        
    def clear_files(self):
        """Limpa a seleção de arquivos."""
        self.audio_files = []
        self.files_listbox.delete(0, tk.END)
        self.status_var.set("Seleção de arquivos limpa")
        
    def run_analysis(self):
        """Executa a análise nos arquivos selecionados."""
        if not self.audio_files:
            messagebox.showwarning("Aviso", "Nenhum arquivo selecionado para análise.")
            return
            
        try:
            # Configuração de parâmetros
            window_length_ms = float(self.window_length_var.get())
            cluster_mode = self.cluster_mode_var.get()
            k = None
            
            if cluster_mode == "manual":
                k = int(self.k_clusters_var.get())
            
            # Iniciar análise em thread separada
            self.status_var.set("Iniciando análise...")
            self.analysis_progress.start()
            
            analysis_thread = threading.Thread(
                target=self._run_analysis_task,
                args=(window_length_ms, k)
            )
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Erro de Valor", f"Parâmetro inválido: {str(e)}")
            self.status_var.set("Erro: Verifique os valores dos parâmetros")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar análise: {str(e)}")
            self.status_var.set(f"Erro: {str(e)}")
            
    def _run_analysis_task(self, window_length_ms, k):
        """Executa a análise em uma thread separada."""
        try:
            # Atualiza o status
            self.root.after(0, lambda: self.status_var.set("Inicializando analisador..."))
            
            # Cria o analisador
            self.analyzer = MultiAudioAnalyzer(window_length_ms=window_length_ms)
            
            # Carrega os arquivos
            self.root.after(0, lambda: self.status_var.set("Carregando arquivos..."))
            self.analyzer.load_audio_files(self.audio_files)
            
            # Executa a análise
            self.root.after(0, lambda: self.status_var.set("Executando análise de características..."))
            self.analyzer.analyze_all_files(k=k)
            
            # Atualiza a interface
            self.root.after(0, lambda: self._analysis_completed())
            
        except Exception as e:
            # Captura a mensagem de erro em uma variável local
            error_message = str(e)
            # Reporta o erro usando a variável local
            self.root.after(0, lambda msg=error_message: self._analysis_failed(msg))
            
    def _analysis_completed(self):
        """Callback para quando a análise é concluída com sucesso."""
        self.analysis_progress.stop()
        self.status_var.set("Análise concluída com sucesso!")
        
        # Habilita a navegação para a próxima etapa
        self.go_to_next_step()
        
        messagebox.showinfo("Sucesso", "Análise dos arquivos concluída com sucesso!")
        
    def _analysis_failed(self, error_msg):
        """Callback para quando a análise falha."""
        self.analysis_progress.stop()
        self.status_var.set(f"Erro durante a análise: {error_msg}")
        messagebox.showerror("Erro na Análise", f"A análise falhou: {error_msg}")
        
    def generate_tracks(self):
        """Gera as tracks baseado nas configurações definidas."""
        if not self.analyzer or not self.analyzer.kmeans:
            messagebox.showwarning("Aviso", "Execute a análise de arquivos primeiro.")
            return
            
        try:
            # Coleta parâmetros de geração
            duration_seconds = float(self.duration_var.get())
            num_tracks = int(self.tracks_var.get())
            
            # Tipo de síntese
            synthesis_type_str = self.synthesis_type_var.get()
            synthesis_type = {
                "concatenative": SynthesisType.CONCATENATIVE,
                "granular": SynthesisType.GRANULAR,
                "spectral": SynthesisType.SPECTRAL
            }[synthesis_type_str]
            
            # Modo de duração
            duration_mode_str = self.duration_mode_var.get()
            duration_mode = {
                "fixed": DurationMode.FIXED,
                "cluster_mean": DurationMode.CLUSTER_MEAN,
                "sequence": DurationMode.SEQUENCE_LENGTH
            }[duration_mode_str]
            
            # Parâmetros específicos da síntese
            synthesis_params = self._get_synthesis_params(synthesis_type_str)
            
            # Iniciar geração em thread separada
            self.generation_status_var.set("Inicializando geração...")
            self.generation_progress.start()
            
            generation_thread = threading.Thread(
                target=self._run_generation_task,
                args=(num_tracks, duration_seconds, synthesis_type, synthesis_params, duration_mode)
            )
            generation_thread.daemon = True
            generation_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Erro de Valor", f"Parâmetro inválido: {str(e)}")
            self.generation_status_var.set("Erro: Verifique os valores dos parâmetros")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar geração: {str(e)}")
            self.generation_status_var.set(f"Erro: {str(e)}")
    
    def _get_synthesis_params(self, synthesis_type):
        """Coleta os parâmetros específicos do tipo de síntese."""
        params = {}
        
        if synthesis_type == "concatenative":
            params['crossfade_duration'] = float(self.crossfade_duration_var.get())
            
        elif synthesis_type == "granular":
            params['grain_size'] = float(self.grain_size_var.get())
            params['density'] = float(self.density_var.get())
            params['pitch_shift'] = float(self.pitch_shift_var.get())
            params['position_jitter'] = float(self.position_jitter_var.get())
            params['duration_jitter'] = float(self.duration_jitter_var.get())
            
        elif synthesis_type == "spectral":
            params['preserve_transients'] = self.preserve_transients_var.get()
            params['spectral_stretch'] = float(self.spectral_stretch_var.get())
            params['fft_size'] = int(self.fft_size_var.get())
            
        return params
        
    def _run_generation_task(
        self, 
        num_tracks, 
        duration_seconds, 
        synthesis_type,
        synthesis_params,
        duration_mode
    ):
        """Executa a geração de tracks em uma thread separada."""
        try:
            # Atualiza o status
            self.root.after(0, lambda: self.generation_status_var.set("Inicializando gerador..."))
            
            # Cria o gerador
            self.generator = MarkovTrackGenerator(self.analyzer)
            
            # Define o modo de duração
            self.root.after(0, lambda: self.generation_status_var.set(f"Configurando modo de duração: {duration_mode.value}..."))
            self.generator.set_duration_mode(duration_mode)
            
            # Gera as tracks
            self.root.after(0, lambda: self.generation_status_var.set(f"Gerando {num_tracks} tracks..."))
            self.generator.generate_tracks(
                num_tracks=num_tracks,
                duration_seconds=duration_seconds,
                synthesis_type=synthesis_type,
                synthesis_params=synthesis_params
            )
            
            # Exporta as tracks
            self.root.after(0, lambda: self.generation_status_var.set("Exportando arquivos..."))
            # Usar uuid em vez de datetime
            unique_id = str(uuid.uuid4())[:8]
            output_folder = f'output_multitrack_{unique_id}'
            self.generator.export_tracks(output_folder)
            self.output_folder = output_folder
            
            # Atualiza a interface
            self.root.after(0, lambda: self._generation_completed())
            
        except Exception as e:
            # Captura a mensagem de erro em uma variável local
            error_message = str(e)
            # Reporta o erro usando a variável local
            self.root.after(0, lambda msg=error_message: self._generation_failed(msg))
            
    def _generation_completed(self):
        """Callback para quando a geração é concluída com sucesso."""
        # Pare o indicador de progresso
        self.generation_progress.stop()
        self.generation_status_var.set("Geração concluída com sucesso!")
        
        # Atualiza informações de saída
        if self.output_folder:
            self.output_path_var.set(os.path.abspath(self.output_folder))
        
        # Habilita a navegação para a próxima etapa
        self.next_button.config(state=tk.NORMAL)
        
        # Exibe mensagem de sucesso - DESACOPLADO da lógica de navegação
        messagebox.showinfo("Sucesso", "Geração das tracks concluída com sucesso! Você pode agora reproduzir os arquivos e visualizar análises.")
        
        # Após a confirmação, use uma abordagem alternativa para mostrar o player
        self.root.after(100, self._show_player_safe)

    def _show_player_safe(self):
        """Método seguro para exibir o player após a confirmação da geração."""
        print("Tentando exibir o player com método seguro...")
        
        # Primeiro, tente o método standard
        try:
            print("Tentativa 1: Usando go_to_next_step...")
            self.current_step = 2  # Configura para a etapa antes do player
            self.go_to_next_step()  # Avança para o player
            
            # Verifica se funcionou
            if self._is_frame_visible("step3_player"):
                print("Sucesso! Player exibido com método standard")
                return
        except Exception as e:
            print(f"Erro na tentativa 1: {str(e)}")
        
        # Se falhou, tente mostrar diretamente
        try:
            print("Tentativa 2: Mostrando frame diretamente...")
            self.show_frame("step3_player")
            self.current_step = 3
            self.update_navigation_state()
            
            # Verifica se funcionou
            if self._is_frame_visible("step3_player"):
                print("Sucesso! Player exibido mostrando diretamente")
                self.update_audio_list()  # Certifique-se de atualizar a lista de áudio
                return
        except Exception as e:
            print(f"Erro na tentativa 2: {str(e)}")
        
        # Se ainda falhou, tente a reconstrução completa
        try:
            print("Tentativa 3: Reconstruindo completamente a interface...")
            if self.reconstruct_player_interface():
                print("Sucesso! Player exibido com reconstrução")
                return
        except Exception as e:
            print(f"Erro na tentativa 3: {str(e)}")
        
        # Última tentativa: Reprogramar toda a interface
        print("Tentativa 4: Última chance - reprogramando a navegação...")
        self.root.after(500, self._force_navigation_to_player)

    def _force_navigation_to_player(self):
        """Método de último recurso para forçar a navegação para o player."""
        try:
            # Reseta a estrutura de navegação
            self.current_step = 3
            
            # Esconde TODOS os widgets da janela principal
            for widget in self.root.winfo_children():
                widget.pack_forget()
            
            # Recria a estrutura principal da interface
            self.main_container = ttk.Frame(self.root)
            self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Recria o container de conteúdo
            self.content_frame = ttk.Frame(self.main_container)
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Recria os frames para cada etapa
            self.initialize_frames()
            
            # Recria a barra de navegação
            self.create_navigation_bar()
            
            # Mostra o frame do player
            self.show_frame("step3_player")
            
            # Atualiza a lista de áudio
            self.root.after(500, self.update_audio_list)
            
            # Força atualização da interface
            self.root.update_idletasks()
        except Exception as e:
            print(f"Falha na última tentativa: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro Crítico", 
                            "Não foi possível exibir a interface do player. " 
                            "Por favor, reinicie o aplicativo.")
        
    def _generation_failed(self, error_msg):
        """Callback para quando a geração falha."""
        self.generation_progress.stop()
        self.generation_status_var.set(f"Erro durante a geração: {error_msg}")
        messagebox.showerror("Erro na Geração", f"A geração falhou: {error_msg}")

    def _handle_success_message(self):
        """Manipula o comportamento após a confirmação da mensagem de sucesso."""
        # Primeiro verifica se o frame do player já está visível
        if self._is_frame_visible("step3_player"):
            print("Frame do player já está visível. Atualizando lista de áudio...")
            self.update_audio_list()
        else:
            print("Navegando para o frame do player...")
            self.current_step = 2  # Configura para etapa anterior ao player
            self.go_to_next_step()  # Avança para o player        
    
    def update_audio_list(self):
        """Atualiza a lista de arquivos de áudio disponíveis com verificações robustas."""
        print("Iniciando atualização da lista de áudio...")
        
        # Verificação de segurança - confirma que estamos no frame do player
        current_frame_name = f"step{self.current_step}_" + ["analysis", "generation", "player", "visualization"][self.current_step-1]
        if current_frame_name != "step3_player":
            print(f"ALERTA: Tentando atualizar lista de áudio mas o frame atual é {current_frame_name}")
        
        # Verifica se temos a lista de tracks
        if not hasattr(self, 'track_list') or self.track_list is None:
            print("track_list não existe ou é None! Verificando frame do player...")
            
            # Verifica se o frame do player existe
            if "step3_player" not in self.frames or self.frames["step3_player"] is None:
                print("ERRO: Frame do player não existe ou é None!")
                return
            
            # Tenta localizar a track_list dentro do frame do player
            try:
                found = False
                for child in self.frames["step3_player"].winfo_children():
                    if isinstance(child, ttk.LabelFrame) and "seleção de áudio" in child.cget("text").lower():
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Frame):
                                for widget in subchild.winfo_children():
                                    if isinstance(widget, tk.Listbox):
                                        self.track_list = widget
                                        found = True
                                        print("Encontrada track_list dentro do frame do player!")
                                        break
                                if found:
                                    break
                        if found:
                            break
                
                if not found:
                    print("ERRO: Não foi possível encontrar track_list, não é possível atualizar lista de áudio!")
                    return
            except Exception as e:
                print(f"Erro ao procurar track_list: {str(e)}")
                return
        
        # Limpa a lista atual
        self.track_list.delete(0, tk.END)
        
        # Verifica se temos pasta de saída
        if not self.output_folder:
            print("Nenhuma pasta de saída disponível para atualizar a lista de áudio")
            return
        
        try:
            print(f"Atualizando lista de áudio da pasta: {self.output_folder}")
            
            # Verifica se a pasta existe
            if not os.path.exists(self.output_folder):
                print(f"ERRO: Pasta de saída não encontrada: {self.output_folder}")
                return
                
            # Adiciona o mix final
            mix_path = os.path.join(self.output_folder, "final_mix.wav")
            if os.path.exists(mix_path):
                self.track_list.insert(tk.END, "Mix Final")
                print(f"Mix final adicionado: {mix_path}")
            else:
                print(f"Mix final não encontrado: {mix_path}")
                
            # Adiciona as tracks individuais
            try:
                track_folders = [d for d in os.listdir(self.output_folder) 
                            if os.path.isdir(os.path.join(self.output_folder, d)) 
                            and d.startswith('track_')]
                
                track_folders.sort(key=lambda x: int(x.split('_')[1]) 
                                if '_' in x and x.split('_')[1].isdigit() else 0)
                
                for folder in track_folders:
                    track_path = os.path.join(self.output_folder, folder, "audio.wav")
                    if os.path.exists(track_path):
                        self.track_list.insert(tk.END, folder)
                        print(f"Track adicionada: {track_path}")
                    else:
                        print(f"Arquivo de track não encontrado: {track_path}")
            except Exception as folder_error:
                print(f"Erro ao listar pastas de tracks: {str(folder_error)}")
                    
            # Se há itens na lista, seleciona o primeiro automaticamente
            if self.track_list.size() > 0:
                self.track_list.selection_set(0)
                print(f"Selecionado primeiro item da lista: {self.track_list.get(0)}")
                
                # Carrega automaticamente o primeiro áudio
                self.root.after(100, self.load_selected_audio)
                
            # Atualiza também a lista de tracks para visualização
            self._update_track_visualization_list()
            
            # Força atualização da interface
            self.root.update_idletasks()
            
            print("Lista de áudio atualizada com sucesso!")
                
        except Exception as e:
            print(f"Erro ao atualizar lista de áudio: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_track_visualization_list(self):
        """Atualiza a lista de tracks disponíveis para visualização."""
        if not self.output_folder:
            return
        
        try:
            # Lista de valores para o combobox
            track_options = ["Mix Final"]
            
            # Adiciona as tracks individuais
            track_folders = [d for d in os.listdir(self.output_folder) if d.startswith('track_')]
            track_folders.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
            
            track_options.extend(track_folders)
            
            # Atualiza o combobox
            self.track_viz_combo['values'] = track_options
            
            # Seleciona o primeiro item se disponível
            if track_options:
                self.track_viz_var.set(track_options[0])
                
        except Exception as e:
            print(f"Erro ao atualizar lista de visualização: {str(e)}")
    
    def load_selected_audio(self):
        """Carrega o áudio selecionado na lista com verificações adicionais."""
        selection = self.track_list.curselection()
        if not selection:
            print("Nenhum item selecionado na lista de áudio")
            return
                
        item_text = self.track_list.get(selection[0])
        print(f"Item selecionado: {item_text}")
        
        if not self.output_folder:
            print("Pasta de saída não disponível")
            messagebox.showwarning("Aviso", "Nenhuma pasta de saída disponível.")
            return
                
        try:
            if item_text == "Mix Final":
                audio_path = os.path.join(self.output_folder, "final_mix.wav")
            else:
                audio_path = os.path.join(self.output_folder, item_text, "audio.wav")
                    
            print(f"Tentando carregar: {audio_path}")
            
            # Verifica se o arquivo existe e pode ser lido
            if not os.path.exists(audio_path):
                print(f"Arquivo não encontrado: {audio_path}")
                messagebox.showerror("Erro", f"Arquivo não encontrado: {audio_path}")
                return
                
            if not os.access(audio_path, os.R_OK):
                print(f"Sem permissão para ler o arquivo: {audio_path}")
                messagebox.showerror("Erro", f"Sem permissão para ler o arquivo: {audio_path}")
                return
                    
            # Tenta carregar o áudio
            if self.audio_player.load_audio(audio_path):
                self.playing_label.config(text=item_text)
                self.player_status.config(text="Pronto para reprodução")
                print(f"Áudio carregado com sucesso: {audio_path}")
            else:
                print(f"Falha ao carregar áudio: {audio_path}")
                messagebox.showerror("Erro", f"Não foi possível carregar o áudio: {audio_path}")
                
        except Exception as e:
            print(f"Erro ao carregar áudio: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro ao carregar áudio: {str(e)}")

    def play_audio(self):
        """Inicia reprodução de áudio."""
        self.audio_player.play()
        self.player_status.config(text="Reproduzindo...")
        
    def pause_audio(self):
        """Pausa reprodução de áudio."""
        self.audio_player.pause()
        self.player_status.config(text="Pausado")
        
    def stop_audio(self):
        """Para reprodução de áudio."""
        self.audio_player.stop()
        self.player_status.config(text="Parado")
        
    def _update_player_state(self):
        """Atualiza o estado do player periodicamente."""
        try:
            # Verifica se terminou de tocar
            if pygame.mixer.get_init() and not pygame.mixer.music.get_busy() and self.audio_player.is_playing():
                self.audio_player.playing = False
                self.player_status.config(text="Reprodução concluída")
                
            # Agenda próxima verificação
            self.root.after(100, self._update_player_state)
        except Exception as e:
            print(f"Erro ao atualizar estado do player: {str(e)}")
            # Mesmo com erro, agenda próxima verificação
            self.root.after(100, self._update_player_state)
        
    def open_output_folder(self):
        """Abre a pasta de saída no explorador de arquivos."""
        if not self.output_folder:
            messagebox.showwarning("Aviso", "Nenhuma pasta de saída disponível")
            return
            
        try:
            if os.name == 'nt':  # Windows
                os.startfile(self.output_folder)
            elif os.name == 'posix':  # macOS ou Linux
                os.system(f'xdg-open "{self.output_folder}"')
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir pasta: {str(e)}")
    
    def update_visualizations(self, event=None):
        """Atualiza as visualizações com base na track selecionada."""
        if not self.output_folder or not self.track_viz_var.get():
            return
                
        try:
            track_name = self.track_viz_var.get()
            
            # Limpa visualizações anteriores
            for widget in self.cluster_plot_frame.winfo_children():
                widget.destroy()
            for widget in self.track_plot_frame.winfo_children():
                widget.destroy()
            for widget in self.transition_plot_frame.winfo_children():
                widget.destroy()
            for widget in self.spectro_plot_frame.winfo_children():
                widget.destroy()
                    
            # Carrega e exibe visualizações
            if track_name == "Mix Final":
                self._load_mix_visualizations()
            else:
                self._load_track_visualizations(track_name)
            
            # Força atualização da interface
            self.root.update_idletasks()
                    
        except Exception as e:
            print(f"Erro ao atualizar visualizações: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro ao carregar visualizações: {str(e)}")
    
    def _load_mix_visualizations(self):
        """Carrega visualizações para o mix final."""
        # Caminho para a pasta de análise do mix
        mix_analysis_folder = os.path.join(self.output_folder, "mix_analysis")
        
        if not os.path.exists(mix_analysis_folder):
            messagebox.showinfo("Informação", "Análises do mix final não disponíveis.")
            return
            
        try:
            # Visualização de clusters - usa arquivo existente
            combined_analysis_path = os.path.join(mix_analysis_folder, "combined_analysis.png")
            if os.path.exists(combined_analysis_path):
                # Criar figura de matplotlib
                img = plt.imread(combined_analysis_path)
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                
                # Exibir na interface
                canvas = FigureCanvasTkAgg(fig, master=self.cluster_plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Adiciona toolbar de navegação
                toolbar = NavigationToolbar2Tk(canvas, self.cluster_plot_frame)
                toolbar.update()
                
            # Carrega as informações do summary.txt
            summary_path = os.path.join(mix_analysis_folder, "summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read()
                    
                # Exibe informações na visualização de tracks
                text_widget = tk.Text(self.track_plot_frame, wrap=tk.WORD, height=20, width=80)
                text_widget.insert(tk.END, summary_text)
                text_widget.config(state=tk.DISABLED)  # Torna o texto não editável
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Adiciona scrollbar
                scrollbar = ttk.Scrollbar(self.track_plot_frame, command=text_widget.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                text_widget.config(yscrollcommand=scrollbar.set)
                
            # Cria espectrograma do mix final
            mix_path = os.path.join(self.output_folder, "final_mix.wav")
            if os.path.exists(mix_path):
                self._create_spectrogram(mix_path, self.spectro_plot_frame, "Mix Final")
                
        except Exception as e:
            print(f"Erro ao carregar visualizações do mix: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_track_visualizations(self, track_name):
        """Carrega visualizações para uma track específica."""
        # Caminho para a pasta da track e sua análise
        track_folder = os.path.join(self.output_folder, track_name)
        analysis_folder = os.path.join(track_folder, "analysis")
        
        if not os.path.exists(analysis_folder):
            messagebox.showinfo("Informação", f"Análises da {track_name} não disponíveis.")
            return
            
        try:
            # Visualização de análise - usa arquivo existente
            analysis_img_path = os.path.join(analysis_folder, "analysis.png")
            if os.path.exists(analysis_img_path):
                # Criar figura de matplotlib
                img = plt.imread(analysis_img_path)
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                
                # Exibir na interface
                canvas = FigureCanvasTkAgg(fig, master=self.cluster_plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Adiciona toolbar de navegação
                toolbar = NavigationToolbar2Tk(canvas, self.cluster_plot_frame)
                toolbar.update()
                
            # Carrega e exibe a matriz de transição
            transition_matrix_path = os.path.join(analysis_folder, "transition_matrix.csv")
            if os.path.exists(transition_matrix_path):
                # Carrega a matriz de transição
                try:
                    transition_df = pd.read_csv(transition_matrix_path, index_col=0)
                    
                    # Cria uma visualização heatmap da matriz
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(transition_df, annot=True, fmt='.2f', cmap='Blues', ax=ax)
                    ax.set_title(f'Matriz de Transição - {track_name}')
                    ax.set_xlabel('Para Estado')
                    ax.set_ylabel('De Estado')
                    
                    # Exibir na interface
                    canvas = FigureCanvasTkAgg(fig, master=self.transition_plot_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Adiciona toolbar de navegação
                    toolbar = NavigationToolbar2Tk(canvas, self.transition_plot_frame)
                    toolbar.update()
                except Exception as e:
                    print(f"Erro ao carregar matriz de transição: {str(e)}")
            
            # Carrega e exibe estatísticas
            stats_path = os.path.join(analysis_folder, "statistics.txt")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats_text = f.read()
                    
                # Exibe estatísticas na visualização de tracks
                text_widget = tk.Text(self.track_plot_frame, wrap=tk.WORD, height=20, width=80)
                text_widget.insert(tk.END, stats_text)
                text_widget.config(state=tk.DISABLED)  # Torna o texto não editável
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Adiciona scrollbar
                scrollbar = ttk.Scrollbar(self.track_plot_frame, command=text_widget.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                text_widget.config(yscrollcommand=scrollbar.set)
                
            # Cria espectrograma da track
            audio_path = os.path.join(track_folder, "audio.wav")
            if os.path.exists(audio_path):
                self._create_spectrogram(audio_path, self.spectro_plot_frame, track_name)
                
        except Exception as e:
            print(f"Erro ao carregar visualizações da track: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_spectrogram(self, audio_path, parent_frame, title):
        """Cria e exibe um espectrograma para o arquivo de áudio."""
        try:
            import librosa
            import librosa.display
            
            # Carrega o áudio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calcula o espectrograma
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(10, 6))
            img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
            ax.set_title(f'Espectrograma - {title}')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
            # Exibir na interface
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Adiciona toolbar de navegação
            toolbar = NavigationToolbar2Tk(canvas, parent_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Erro ao criar espectrograma: {str(e)}")
            # Exibe uma mensagem no frame em caso de erro
            label = ttk.Label(parent_frame, text=f"Erro ao gerar espectrograma: {str(e)}")
            label.pack(pady=20)

# Função principal
# Função principal corrigida
def main():
    """Função principal para iniciar a aplicação."""
    try:
        # Garante que as importações necessárias estão disponíveis no escopo da função
        import tkinter as tk_local
        from tkinter import messagebox as messagebox_local
        
        # Verifica se o pygame mixer está disponível
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            print("Pygame mixer inicializado com sucesso")
        
        # Cria a janela principal - Usando a importação local
        root = tk_local.Tk()
        root.title("Sistema de Composição Musical Markoviano")
        
        # Configura o tamanho e posição da janela
        window_width = 1200
        window_height = 800
        
        # Obtém as dimensões da tela
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calcula a posição para centralizar a janela
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Define a geometria da janela
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(1000, 700)
        
        # Cria a aplicação
        app = MarkovGUIImproved(root)
        
        # Define uma função simples para lidar com o fechamento
        def on_closing():
            try:
                pygame.mixer.quit()
                pygame.quit()
                print("Pygame finalizado")
            except:
                pass
            root.destroy()
        
        # Configura o tratamento de fechamento da janela
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Inicia o loop principal
        root.mainloop()
        
    except Exception as e:
        print(f"Erro ao iniciar a aplicação: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Tenta mostrar uma mensagem de erro em uma janela - usando as importações locais
        try:
            import tkinter as tk_error
            from tkinter import messagebox as messagebox_error
            error_root = tk_error.Tk()
            error_root.withdraw()
            messagebox_error.showerror("Erro Fatal", f"Erro ao iniciar a aplicação: {str(e)}\n\n"
                                      "Verifique se todas as dependências estão instaladas.")
            error_root.destroy()
        except:
            pass  # Se falhar ao mostrar o erro gráfico, pelo menos teremos o traceback no console

if __name__ == "__main__":
    main()