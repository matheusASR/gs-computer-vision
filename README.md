# Alerta Córrego 🌊

## Integrantes

- João Pedro de Albuquerque Oliveira | 551579
- Pedro Augusto Carneiro Barone Bomfim | 99781
- Matheus Augusto Santos Rego | 551466

## Sobre o projeto

O Alerta Córrego é uma proposta de solução tecnológica voltada à prevenção de impactos causados por enchentes em comunidades localizadas próximas a córregos e regiões de risco na cidade de São Paulo.

A solução busca fornecer alertas antecipados aos moradores, permitindo que tenham mais tempo para se preparar e tomar medidas preventivas antes que uma inundação atinja suas residências.

## Problema

Moradores de regiões vulneráveis frequentemente percebem o avanço das enchentes quando a água já está atingindo suas ruas ou casas. A ausência de informações específicas e em tempo real sobre as condições locais dificulta a tomada de decisões preventivas e aumenta os prejuízos materiais e os riscos à população.

## Solução proposta

O Alerta Córrego propõe um sistema de monitoramento de regiões próximas a córregos, utilizando informações sobre o nível da água e análise visual para identificar situações de risco e auxiliar na emissão de alertas antecipados.

A solução busca tornar a informação mais acessível, localizada e útil para os moradores das regiões afetadas.

**Escopo atual:** foi implementado um protótipo de visão computacional para detectar réguas de nível e localizar visualmente a linha da água em imagens. A emissão automática de alertas e a medição física validada ainda não fazem parte das funcionalidades concluídas.

## Visão Computacional

O componente de visão computacional analisa imagens de áreas próximas a córregos para reconhecer réguas de medição (`staff_gauge`) e a posição visual da água (`water_level`). A implementação utiliza detecção de objetos com YOLO11n e processamento/visualização de imagens com OpenCV.

### Objetivos

- Identificar réguas de nível em imagens.
- Detectar a linha da água próxima às réguas.
- Avaliar quantitativamente o desempenho das detecções.
- Investigar uma conversão da posição da água em leitura física mediante calibração.
- Futuramente, complementar dados de monitoramento para apoiar alertas de inundação.

### Tecnologias utilizadas

- **Python 3.11** — linguagem de implementação.
- **Ultralytics YOLO11n** — treinamento e inferência do detector.
- **PyTorch** — execução do modelo de aprendizado profundo.
- **OpenCV** — leitura, recorte e anotação de imagens; protótipos de calibração.
- **NumPy** — operações numéricas e cálculo de erros.

### Dataset

Foi utilizado um conjunto de imagens anotadas exportado do **Roboflow (projeto Sungai, versão 1)**, em formato YOLO, com duas classes:

| ID | Classe | Descrição |
|---|---|---|
| 0 | `staff_gauge` | Régua de medição do nível da água. |
| 1 | `water_level` | Região anotada correspondente à linha/posição da água. |

| Partição | Imagens |
|---|---:|
| Treinamento (`train`) | 630 |
| Validação (`valid`) | 60 |
| Teste (`test`) | 30 |
| **Total** | **720** |

As anotações da classe `water_level` frequentemente são caixas horizontais muito estreitas. Isso torna a detecção sensível a pequenas diferenças de posição e ao critério de IoU. Os metadados da exportação indicam **CC BY 4.0**; antes de redistribuir imagens, é necessário conferir atribuição, procedência e eventuais direitos de terceiros, inclusive marcas d'água.

Local esperado do dataset:

```text
computer_vision/dataset/roboflow_export/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

### Treinamento

Foi treinado um detector **YOLO11n**, com os seguintes parâmetros principais:

| Parâmetro | Valor |
|---|---|
| Arquitetura | YOLO11n |
| Épocas | 30 |
| Resolução de treinamento (`imgsz`) | 416 |
| Lote (`batch`) | 4 |
| Ambiente utilizado | CPU |
| Pesos selecionados | `best.pt` |

Os pesos treinados são esperados em:

```text
computer_vision/results/training/alerta_corrego_yolo11n/weights/best.pt
```

### Resultados do modelo

**Validação (60 imagens, 131 instâncias; resolução 416):**

| Classe | Precisão | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| Todas | 0,585 | 0,544 | 0,516 | 0,292 |
| `staff_gauge` | 0,774 | 0,824 | 0,804 | 0,528 |
| `water_level` | 0,396 | 0,263 | 0,228 | 0,055 |

**Teste separado (30 imagens, 71 instâncias; resolução 416):**

| Classe | Precisão | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| Todas | 0,561 | 0,634 | 0,578 | 0,331 |
| `staff_gauge` | 0,709 | 0,856 | 0,881 | 0,589 |
| `water_level` | 0,413 | 0,412 | 0,275 | 0,073 |

Essas métricas mostram melhor desempenho na detecção de réguas do que na detecção da linha da água. Os valores acima correspondem às avaliações do modelo treinado; não devem ser confundidos com as métricas exploratórias de associação e localização descritas a seguir.

### Experimentos de pós-processamento

Foram comparadas duas formas de estimar a posição vertical da água:

1. **Heurística OpenCV:** busca uma mudança de intensidade dentro da caixa da régua.
2. **Detecção YOLO:** utiliza o centro vertical da caixa prevista para `water_level`.

Nos **28 casos comparáveis** da validação, o erro vertical médio normalizado foi de **12,22% da altura da imagem** com a heurística OpenCV e **0,58%** com a detecção YOLO. Essa comparação é condicionada aos casos em que houve previsão e utiliza associação geométrica aproximada; não mede a precisão do sistema em todos os cenários.

Também foram avaliados diferentes limiares mínimos de confiança, mantendo a resolução de inferência em 416:

| Confiança | Recall exploratório da água | Precisão exploratória da água | Erro vertical médio nos pares associados |
|---|---:|---:|---:|
| 0,10 | 68,42% | 45,88% | 0,41% |
| **0,15** | **63,16%** | **58,06%** | **0,55%** |
| 0,20 | 56,14% | 66,67% | 0,53% |
| 0,25 | 47,37% | 79,41% | 0,53% |

**Nota metodológica:** nessa análise, as caixas de água foram associadas às anotações com **IoU ≥ 0,10**, critério mais permissivo que o usado em métricas padrão de detecção. A precisão foi estimada pela razão entre associações válidas e previsões, com pareamento sem reutilização dentro de cada imagem. Esses números são exploratórios, não equivalem a mAP@50 e não constituem validação de segurança para alertas reais.

O limiar **0,15** foi escolhido provisoriamente para a demonstração por equilibrar cobertura e falsos positivos no conjunto de validação. A escolha não foi confirmada por uma avaliação final independente no conjunto de teste.

### Demonstração

O arquivo `computer_vision/src/demo.py` executa o modelo treinado sobre imagens do conjunto de validação, filtra detecções de réguas, associa geometricamente linhas de água às réguas e salva imagens anotadas:

- **Caixas verdes:** réguas detectadas.
- **Linhas azuis:** posições de água associadas.

Na execução realizada, o programa processou as **60 imagens de validação**. Houve pelo menos uma linha associada em **46 imagens** e nenhuma linha associada em **14 imagens**. Essa contagem descreve a saída do protótipo, **não** a quantidade de detecções comprovadamente corretas.

As regras de associação são heurísticas e podem falhar em imagens com várias réguas, caixas sobrepostas ou detecções incorretas.

### Calibração experimental

O módulo `computer_vision/src/interactive_calibration.py` permite selecionar manualmente dois pontos de referência na régua e estimar uma leitura por interpolação linear a partir da coordenada vertical prevista para a água.

Foi realizado um teste ilustrativo com referências `85,00` e `84,00`, resultando em uma leitura numérica próxima de **83,55**. Como a linha detectada estava fora do intervalo entre as referências, o cálculo foi uma **extrapolação**. Além disso, os traços exatos da escala e a interpretação física dessas cotas não foram validados.

**Portanto, 83,55 é um resultado experimental, não uma medição física comprovada em centímetros ou metros.** A calibração confiável exige marcas físicas verificadas, correspondência correta entre água e régua e definição da referência vertical utilizada.

### Estrutura principal do projeto

```text
gs-computer-vision/
├── README.md
└── computer_vision/
    ├── dataset/
    │   └── roboflow_export/
    │       ├── data.yaml
    │       ├── train/
    │       ├── valid/
    │       └── test/
    ├── src/
    │   ├── train.py
    │   ├── detect.py
    │   ├── evaluate.py
    │   ├── evaluate_resolution.py
    │   ├── estimate_water_level.py
    │   ├── evaluate_water_line.py
    │   ├── evaluate_yolo_water_level.py
    │   ├── evaluate_confidence.py
    │   ├── visualize_water_level.py
    │   ├── compare_water_predictions.py
    │   ├── inspect_gauge.py
    │   ├── calibrate_water_level.py
    │   ├── interactive_calibration.py
    │   └── demo.py
    └── results/
        ├── training/alerta_corrego_yolo11n/weights/best.pt
        ├── test_evaluation/
        ├── demo/
        └── interactive_calibration/
```

A árvore apresenta os arquivos principais utilizados no desenvolvimento; outros arquivos e diretórios auxiliares podem estar presentes.

### Como executar (Windows / PowerShell)

**Pré-requisitos:** Python 3.11, dataset disponível na estrutura indicada e pesos treinados em `best.pt`.

Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy
```

Para executar a demonstração em todas as imagens de validação:

```powershell
python computer_vision/src/demo.py
```

Para processar uma imagem específica, informe seu **nome exato** na pasta `valid/images`:

```powershell
python computer_vision/src/demo.py --image img_0004_jpg.rf.508bc9da35535ff606429e34120d0646.jpg
```

Os resultados são salvos em:

```text
computer_vision/results/demo/
```

Para repetir a comparação exploratória de limiares de confiança:

```powershell
python computer_vision/src/evaluate_confidence.py
```

Para abrir a calibração manual experimental (requer ambiente com interface gráfica):

```powershell
python computer_vision/src/interactive_calibration.py
```

**Observação:** instalar as dependências não substitui o arquivo `best.pt`, que precisa ser disponibilizado separadamente caso não esteja versionado no repositório. O conjunto de imagens também precisa estar presente localmente.

### Limitações e próximos passos

- A detecção de `water_level` apresenta cobertura limitada e pode produzir falsos positivos.
- O pareamento entre réguas e linhas de água utiliza regras geométricas aproximadas.
- As anotações do dataset podem não coincidir exatamente com o contato físico da água na régua.
- A leitura de altura física ainda depende de calibração validada, preferencialmente com referências verificáveis na imagem.
- Não foram definidos nem validados limiares hidrológicos para classificação de risco.
- Não há, nesta etapa, monitoramento contínuo ou emissão automática de alertas a moradores.
- Antes de uma aplicação real, são necessários novos dados representativos, avaliação independente, testes em condições ambientais variadas e integração com outras fontes de monitoramento.

## Impacto esperado

- Maior antecedência na identificação de situações de risco.
- Redução de danos materiais causados por enchentes.
- Maior acesso à informação por moradores de áreas vulneráveis.
- Apoio à prevenção e à tomada de decisões.

Esses impactos representam **objetivos futuros da solução proposta**, não resultados já demonstrados em campo.

## Status

**Protótipo de visão computacional implementado e avaliado experimentalmente.** O projeto geral permanece **em desenvolvimento**: a medição física confiável, a classificação de risco e a emissão de alertas ainda exigem validação e integração adicionais.
