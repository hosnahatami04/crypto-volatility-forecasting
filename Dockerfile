FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# torch's default PyPI wheel pulls in ~2GB of NVIDIA CUDA dependencies this
# project never uses (CPU-only walk-forward training). Installing torch
# from its CPU-only index first, then the rest of requirements.txt normally,
# keeps the image small and the build fast.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.eval.run_final_report"]
