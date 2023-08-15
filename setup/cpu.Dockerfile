FROM ubuntu:20.04
#FROM continuumio/miniconda3

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# get code repo
RUN apt-get update && \
	apt-get install -y git openssh-client wget vim htop tmux scons libboost-all-dev build-essential gcc-9 g++-9
# use this instead of cloning the repo (private right now)
RUN mkdir -p /workspace /workspace/chkpts
COPY . /workspace/hetero-accel
WORKDIR /workspace/hetero-accel
RUN mkdir -p /root/.ssh/
RUN mv github_balaskas /root/.ssh/

# install anaconda
ENV CONDA_DIR /opt/conda
RUN apt-get update && \
	apt-get install --yes libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 && \
	apt-get clean all
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p $CONDA_DIR && \
	rm ~/anaconda.sh
# set path to conda
ENV PATH $CONDA_DIR/bin:$PATH

# install and activate conda environment
# the dependencies of all comprising tools are included in this environment
RUN conda init bash
RUN conda update -y -n base -c defaults conda && \
	conda env create -f setup/environment.yml
# update the bashrc file to fix conda loading issue
COPY setup/bashrc /etc/bashrc
RUN cat /etc/bashrc >> /root/.bashrc

# ENTRYPOINT ["/bin/bash", "-i"]

