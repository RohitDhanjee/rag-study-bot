FROM python:3.12-slim

# Run as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=0
WORKDIR $HOME/app

# Install dependencies first so code changes do not re-run this slow step
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Then copy the code; download the model and build the index at build time
COPY --chown=user . .
RUN python build_index.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
