FROM python:3.12-slim

# Instala dependências do sistema necessárias para o OpenCV rodar no Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala as dependências do Python
# ⚠ AJUSTADO: Removido o prefixo 'corn_predictions/'
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gspread pandas  # Garante as ferramentas de sincronização

# Copia o restante do código do projeto
COPY . .

# Comando padrão
CMD ["python", "mvp/consumer.py", "--model_path", "models/maize_densenet121_state_dict.pt", "--meta_path", "models/maize_densenet121_meta.json"]
