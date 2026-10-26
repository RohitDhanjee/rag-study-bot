FROM python:3.12-slim

# HF Spaces app ko user 1000 ke naam se chalata hai
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=0
WORKDIR $HOME/app

# Pehle sirf libraries, taake code badalne par ye mehenga step dobara na chale
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Phir code, aur build ke waqt hi model download + index tayyar
COPY --chown=user . .
RUN python build_index.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]