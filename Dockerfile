# 1. Use a lightweight Python 3.12 image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the source code and the model artifacts into the container
COPY src/ ./src/
COPY artifacts/model_v2/ ./artifacts/model_v2/
COPY configs/ ./configs/

# 5. Set the working directory to src, where our code lives
WORKDIR /app/src

# 6. Expose port 8000 to the outside world
EXPOSE 8000

# 7. The command to run when the container starts
CMD ["python", "-m", "uvicorn", "cortex.api:app", "--host", "0.0.0.0", "--port", "8000"]