# Contributing Guide / Guia de Contribuição

## English

### How to Contribute

We welcome contributions to the Markovian Musical Composition System! Here's how you can help:

#### Types of Contributions

1. **Bug Reports**: Found a bug? Please report it!
2. **Feature Requests**: Have an idea for improvement? Let us know!
3. **Code Contributions**: Submit pull requests for bug fixes or new features
4. **Documentation**: Help improve documentation and examples
5. **Testing**: Help test the system with different audio files and scenarios

#### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/markovian-musical-composition.git
   cd markovian-musical-composition
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

#### Development Guidelines

##### Code Style
- Follow PEP 8 for Python code formatting
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

##### Example of Good Code Style:
```python
def extract_spectral_features(audio_signal: np.ndarray, sr: int, 
                            window_size: int) -> Dict[str, float]:
    """
    Extract spectral features from audio signal.
    
    Args:
        audio_signal: Audio data as numpy array
        sr: Sample rate in Hz
        window_size: Window size in samples
        
    Returns:
        Dictionary containing spectral features
        
    Raises:
        ValueError: If audio signal is empty or invalid
    """
    if len(audio_signal) == 0:
        raise ValueError("Audio signal cannot be empty")
    
    # Implementation here...
    return features
```

##### Testing
- Test your changes with different audio file formats
- Test edge cases (empty files, very short files, etc.)
- Ensure backward compatibility
- Test both desktop and Streamlit interfaces

##### Documentation
- Update relevant documentation when adding features
- Add examples for new functionality
- Update API documentation if needed
- Write clear commit messages

#### Pull Request Process

1. **Before Submitting**
   - Ensure your code follows the style guidelines
   - Test thoroughly with different audio files
   - Update documentation if needed
   - Make sure all existing tests pass

2. **Submit Pull Request**
   - Use a clear, descriptive title
   - Provide detailed description of changes
   - Reference any related issues
   - Include screenshots/audio samples if relevant

3. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Tested with different audio formats
   - [ ] Tested desktop interface
   - [ ] Tested Streamlit interface
   - [ ] No regressions in existing functionality
   
   ## Related Issues
   Fixes #(issue number)
   ```

#### Areas Needing Contribution

##### High Priority
- **Performance Optimization**: Improve processing speed for large files
- **Audio Format Support**: Add support for more audio formats
- **Error Handling**: Better error messages and recovery
- **Mobile Interface**: Improve Streamlit mobile experience

##### Medium Priority
- **New Synthesis Methods**: Implement additional synthesis algorithms
- **Visualization**: Enhanced audio visualizations
- **Export Options**: More output formats and options
- **Batch Processing**: GUI for batch operations

##### Low Priority
- **Presets**: Predefined parameter sets for different styles
- **Audio Effects**: Built-in audio effects and filters
- **MIDI Support**: MIDI file analysis and generation
- **Plugin Architecture**: Support for external plugins

#### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g. Windows 10, macOS 12, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Interface: [Desktop/Streamlit]
- Audio File Format: [e.g. WAV, MP3]

**Additional Context**
Any other relevant information
```

#### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why would this feature be useful?

**Proposed Implementation**
How do you think this could be implemented?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

---

## Português

### Como Contribuir

Contribuições para o Sistema de Composição Musical Markoviano são bem-vindas! Aqui está como você pode ajudar:

#### Tipos de Contribuições

1. **Relatórios de Bug**: Encontrou um bug? Por favor, reporte!
2. **Solicitações de Funcionalidades**: Tem uma ideia de melhoria? Nos conte!
3. **Contribuições de Código**: Envie pull requests para correções ou novas funcionalidades
4. **Documentação**: Ajude a melhorar documentação e exemplos
5. **Testes**: Ajude a testar o sistema com diferentes arquivos e cenários

#### Começando

1. **Fazer Fork do Repositório**
   ```bash
   git clone https://github.com/seuusuario/markovian-musical-composition.git
   cd markovian-musical-composition
   ```

2. **Configurar Ambiente de Desenvolvimento**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # No Windows: dev_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Se disponível
   ```

3. **Criar uma Branch**
   ```bash
   git checkout -b feature/nome-da-sua-funcionalidade
   # ou
   git checkout -b bugfix/descricao-do-problema
   ```

#### Diretrizes de Desenvolvimento

##### Estilo de Código
- Siga PEP 8 para formatação de código Python
- Use nomes significativos para variáveis e funções
- Adicione docstrings a todas as funções e classes
- Mantenha funções focadas e modulares

##### Exemplo de Bom Estilo de Código:
```python
def extrair_caracteristicas_espectrais(sinal_audio: np.ndarray, sr: int, 
                                     tamanho_janela: int) -> Dict[str, float]:
    """
    Extrai características espectrais do sinal de áudio.
    
    Args:
        sinal_audio: Dados de áudio como array numpy
        sr: Taxa de amostragem em Hz
        tamanho_janela: Tamanho da janela em amostras
        
    Returns:
        Dicionário contendo características espectrais
        
    Raises:
        ValueError: Se o sinal de áudio estiver vazio ou inválido
    """
    if len(sinal_audio) == 0:
        raise ValueError("Sinal de áudio não pode estar vazio")
    
    # Implementação aqui...
    return caracteristicas
```

#### Áreas Precisando de Contribuição

##### Alta Prioridade
- **Otimização de Performance**: Melhorar velocidade de processamento para arquivos grandes
- **Suporte a Formatos de Áudio**: Adicionar suporte a mais formatos
- **Tratamento de Erros**: Melhores mensagens de erro e recuperação
- **Interface Mobile**: Melhorar experiência mobile do Streamlit

##### Média Prioridade
- **Novos Métodos de Síntese**: Implementar algoritmos de síntese adicionais
- **Visualização**: Visualizações de áudio aprimoradas
- **Opções de Exportação**: Mais formatos e opções de saída
- **Processamento em Lote**: GUI para operações em lote

##### Baixa Prioridade
- **Presets**: Conjuntos de parâmetros predefinidos para diferentes estilos
- **Efeitos de Áudio**: Efeitos e filtros de áudio integrados
- **Suporte MIDI**: Análise e geração de arquivos MIDI
- **Arquitetura de Plugins**: Suporte para plugins externos

#### Template de Relatório de Bug

```markdown
**Descrição do Bug**
Descrição clara do bug

**Passos para Reproduzir**
1. Vá para '...'
2. Clique em '....'
3. Veja o erro

**Comportamento Esperado**
O que você esperava que acontecesse

**Comportamento Atual**
O que realmente aconteceu

**Ambiente**
- SO: [ex. Windows 10, macOS 12, Ubuntu 20.04]
- Versão do Python: [ex. 3.9.7]
- Interface: [Desktop/Streamlit]
- Formato do Arquivo de Áudio: [ex. WAV, MP3]

**Contexto Adicional**
Qualquer outra informação relevante
```

#### Template de Solicitação de Funcionalidade

```markdown
**Descrição da Funcionalidade**
Descrição clara da funcionalidade proposta

**Caso de Uso**
Por que esta funcionalidade seria útil?

**Implementação Proposta**
Como você acha que isso poderia ser implementado?

**Alternativas Consideradas**
Outras abordagens que você considerou

**Contexto Adicional**
Qualquer outra informação relevante
```