
# scipy/machine learning (tensorflow, pytorch)
#https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root
RUN apt-get -y install git 
# 3) install packages using notebook user
USER jovyan 
# RUN conda install -y environment dependencies seen on environment.yaml
# docker push
RUN pip install --no-cache-dir pandas numpy vak scipy torch tqdm librosa sox multipledispatch tabulate seaborn plotly colorednoise 

#RUN git clone https://github.com/EdmundoZamora/Methodology5.git

CMD ["/bin/bash"] 
#[ "python","-u", ".\run.py", "data", "features", "model", "evaluate" ]