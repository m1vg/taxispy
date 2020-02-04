FROM m1vg/taxispy:base
COPY . /Documents
RUN sudo mv taxispy /opt/conda/lib/python3.6/site-packages && rm Dockerfile
EXPOSE 8888
CMD jupyter notebook --NotebookApp.token='' --allow-root 
