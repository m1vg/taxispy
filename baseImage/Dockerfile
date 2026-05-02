FROM quay.io/jupyter/minimal-notebook:python-3.11

USER root

COPY environment.txt /tmp/taxispy-environment.txt
WORKDIR /Documents
RUN apt-get update && \
    apt-get install --yes --no-install-recommends libxml2 && \
    rm -rf /var/lib/apt/lists/* && \
    mamba install --yes --channel conda-forge --override-channels --file /tmp/taxispy-environment.txt && \
    python -m pip install --no-cache-dir scoop deap && \
    conda clean --all --force --yes && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    mkdir -p /Documents && \
    chown -R "${NB_UID}:${NB_GID}" /Documents

USER ${NB_UID}

EXPOSE 8888
CMD ["start-notebook.py", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/Documents"]
