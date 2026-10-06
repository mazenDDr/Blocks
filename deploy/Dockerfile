# Control service image (ADR 0074). Build from the repository root:  docker build -f deploy/Dockerfile -t project-void .
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 VOID_WORKBENCH=/data/workbench
WORKDIR /app
COPY python/requirements.txt python/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r python/requirements.txt
COPY pyproject.toml ./
COPY python python
COPY services services
COPY examples examples
COPY deploy/entrypoint.sh /usr/local/bin/void-entrypoint
RUN pip install --no-deps -e . && chmod 0755 /usr/local/bin/void-entrypoint && useradd --create-home --uid 10001 void && mkdir -p /data && chown void /data
USER void
VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["void-entrypoint"]
