FROM python:3.12-slim

# Instala dependências do sistema necessárias para o OpenCV rodar no Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala as dependências do Python
COPY corn_predictions/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gspread pandas  # Garante as ferramentas de sincronização

# Copia o restante do código do projeto
COPY . .

# Comando padrão (pode ser sobrescrito pelo docker-compose)
CMD ["python", "corn_predictions/mvp/consumer.py", "--model_path", "corn_predictions/models/maize_densenet121_state_dict.pt", "--meta_path", "corn_predictions/models/maize_densenet121_meta.json"]
