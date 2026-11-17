# Hugging Face Space (Docker SDK): the interactive article and its API.
# Build context: the folder written by `python space/assemble.py <dir>` (this file sits at its root).
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# the app writes here at run time (runs/ledger.jsonl, blog/index.html, space/_site): owned by the user
RUN mkdir -p $HOME/app
WORKDIR $HOME/app

COPY --chown=user:user space/requirements.txt space/requirements.txt
RUN pip install --no-cache-dir --no-deps -r space/requirements.txt

COPY --chown=user:user . .
RUN python space/build_page.py

EXPOSE 7860
CMD ["uvicorn", "space.app:app", "--host", "0.0.0.0", "--port", "7860"]
