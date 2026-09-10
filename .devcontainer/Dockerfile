FROM python:3.10-slim

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install common Python packages for quant dev
RUN pip install --upgrade pip setuptools wheel
RUN pip install black isort pytest pandas numpy jupyter jupyterlab sqlalchemy

CMD ["/bin/bash"]
