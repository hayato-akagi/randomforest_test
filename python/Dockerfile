FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY main.py .
COPY generate_dataset.py .
COPY train_model.py .

# Run data generation and training
CMD ["python", "main.py"]
