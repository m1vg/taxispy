FROM jupyter/minimal-notebook
USER root
COPY . /Documents
WORKDIR /Documents
RUN conda config --add channels bioconda && \
    conda config --add channels conda-forge && \
    conda install --yes --file /Documents/environment.txt && \
    sudo apt-get update -y && sudo apt-get install ffmpeg -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    rm /Documents/environment.txt

EXPOSE 8888
CMD jupyter notebook --NotebookApp.token='' --allow-root 
