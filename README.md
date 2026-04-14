# 🌽 Monitoramento Inteligente de Fitossanidade (Milho)

Sistema de Visão Computacional baseado em Deep Learning para detecção e classificação de patologias em folhas de milho, com integração automática para Business Intelligence (BI).

## 📝 Resumo
O projeto utiliza a arquitetura de rede neural **DenseNet121** para analisar imagens de folhas de milho e classificá-las entre saudáveis ou doentes. O diferencial do sistema é a aplicação de **Grad-CAM++**, que gera mapas de calor (máscaras) sobre a imagem original, explicando visualmente quais áreas da folha levaram o modelo àquela conclusão.

## ✨ Funcionalidades
- **Classificação Automática:** Identificação de múltiplas classes de doenças e estado saudável.
- **IA Explicável (XAI):** Geração de imagens anotadas com máscaras adaptativas que destacam as lesões foliares.
- **Fluxo de BI Automatizado:** Sincronização direta entre o processamento em Python e dashboards interativos.
- **Gestão Agrícola:** Suporte para segmentação de dados por **Fazenda** e **Talhão (Plot)**.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.12+
- **Deep Learning:** PyTorch & Torchvision (DenseNet121)
- **Visão Computacional:** OpenCV (Masking e Anotação)
- **Banco de Dados:** SQLite (Armazenamento local de inferências)
- **Integração e BI:** Google Sheets API & Google Looker Studio

## 🚀 Como Testar (Ambiente Google Colab)


### 1. - Abrir projeto em ferramenta para python opencv (google colab, por exemplo)

### 2. Preparação e Autenticação - criar cêlula com comando para autenticação: 
from google.colab import auth
auth.authenticate_user()

### 3. - Verificar instalação: 
!pip install gspread

### 4. - Clonar projeto:
!git clone ...

### 5. - Colocar imagens na pasta "input images"

### 6. - Para analisar as imagens por upload na pasta: 
!python corn_predictions/mvp/consumer.py --model_path corn_predictions/models/maize_densenet121_state_dict.pt --meta_path corn_predictions/models/maize_densenet121_meta.json --input_dir corn_predictions/input_images --run_once

### 7. - Pasta "annotated_images" é criada. Dentro dela é possível visualizar demarcação da doença na planta.

### 8. - Exportar csv e google sheets:
!python corn_predictions/mvp/export_results_to_csv.py --flatten_probs

### 9. - Arquivo "results.csv" é criado e dados são enviados ao googles sheets. 

### 10. - dados conectados ao painel em google looker studio.
