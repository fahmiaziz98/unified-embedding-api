# # Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# # you will also find guides on how best to write your Dockerfile

# FROM python:3.10-slim

# RUN useradd -m -u 1000 user
# USER user
# ENV PATH="/home/user/.local/bin:$PATH"

# WORKDIR /app

# COPY --chown=user ./requirements.txt requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# COPY --chown=user . /app
# CMD ["python", "app.py"]

# Gunakan base image ringan tapi sudah mendukung PyTorch + Transformers
FROM huggingface/transformers-pytorch-cpu:latest

# Buat user non-root (wajib di Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy file requirements dan install dependency
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy seluruh source code
COPY --chown=user . /app

# Jalankan aplikasi
CMD ["python", "app.py"]
