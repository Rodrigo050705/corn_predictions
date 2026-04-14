Projeto para identificação de plantas doentes e saudáveis

Resumo: uso de visão computacional para identificação e classificação de plantas saudáveis e doentes. Dados exportados para visualização em dashboards para análise.

Funcionalidades: análise de imagens, classificação e exportação de dados para painel interativo para análise de dados. Atualização realizada em tempo real.

Descrição: imagens são analisadas aplicando técnicas de visão computacional (como masking). Dados seguem o fluxo: arquivo csv -> google sheets -> google looker studio (painel).  

Tecnologias utilizadas:
- Python (OpenCV)
- Google sheets
- Google Looker Studio

Como testar:
1 - Abrir projeto em ferramenta para python opencv (google colab, por exemplo)

2 - criar cêlula com comando para autenticação: 
from google.colab import auth
auth.authenticate_user()

3 - verificar instalação: 
!pip install gspread

4 - Clonar projeto: !git clone ...

5 - Colocar imagens na pasta "input images"

6 - Para analisar as imagens por upload na pasta: 
!python corn_predictions/mvp/consumer.py --model_path corn_predictions/models/maize_densenet121_state_dict.pt --meta_path corn_predictions/models/maize_densenet121_meta.json --input_dir corn_predictions/input_images --run_once

7 - Pasta "annotated_images" é criada. Dentro dela é possível visualizar demarcação da doença na planta.

8 - Exportar csv e google sheets:
!python corn_predictions/mvp/export_results_to_csv.py --flatten_probs

9 - Arquivo "results.csv" é criado e dados são enviados ao googles sheets. 

10 - dados conectados ao painel em google looker studio.
