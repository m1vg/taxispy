ARG ORIGINAL_WORKSPACE_PLATFORM=linux/amd64
FROM --platform=${ORIGINAL_WORKSPACE_PLATFORM} m1vg/taxispy AS original-workspace

FROM quay.io/jupyter/minimal-notebook:python-3.11

LABEL org.opencontainers.image.title="TaxisPy" \
      org.opencontainers.image.description="Jupyter environment for quantitative analysis of bacterial chemotaxis" \
      org.opencontainers.image.version="0.1.6.4"

USER root
RUN apt-get update && \
    apt-get install --yes --no-install-recommends libxml2 && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/taxispy-environment.yml
RUN mamba env update --name base --file /tmp/taxispy-environment.yml && \
    conda clean --all --force --yes && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    mkdir -p /Documents /opt/taxispy && \
    chown -R "${NB_UID}:${NB_GID}" /Documents /opt/taxispy

COPY --from=original-workspace --chown=${NB_UID}:${NB_GID} /Documents /Documents
COPY --chown=${NB_UID}:${NB_GID} taxispy /opt/taxispy/taxispy

ENV PYTHONPATH="/opt/taxispy:/opt/taxispy/taxispy:/Documents:/Documents/taxispy" \
    TAXISPY_VERSION="0.1.6.4"
WORKDIR /Documents
USER ${NB_UID}

EXPOSE 8888
CMD ["start-notebook.py", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/Documents"]
