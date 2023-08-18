FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
	apt install -y git openssh-client wget curl pip vim htop tmux scons libboost-all-dev build-essential g++-9 \
			   scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" >> /etc/apt/sources.list
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" >> /etc/apt/sources.list
RUN apt update && apt install -y g++-5 gcc-5 g++-7 gcc-7

# use this instead of cloning the repo (private right now)
RUN mkdir -p /workspace /workspace/chkpts
COPY . /workspace/hetero-accel
WORKDIR /workspace/hetero-accel
RUN mkdir -p /root/.ssh/
RUN mv github_balaskas /root/.ssh/

# install anaconda, if needed
ENV CONDA_DIR /opt/conda
RUN apt update && \
	apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 && \
	apt clean all
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p $CONDA_DIR && \
	rm ~/anaconda.sh
# set path to conda
ENV PATH $CONDA_DIR/bin:$PATH
# initialize conda on new shells
RUN conda init
RUN echo "conda activate haccel" >> /root/.bashrc

# the dependencies of all comprising tools are included in this environment
RUN conda update -y -n base -c defaults conda
RUN conda create -n haccel python=3.9
SHELL ["conda", "run", "-n", "haccel", "/bin/bash", "--login", "-c"]
RUN pip install --upgrade pip && \
	pip install -r setup/requirements.txt

# install timeloop-accelergy
#RUN git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
WORKDIR accelergy-timeloop-infrastructure
RUN make pull
WORKDIR src/cacti
RUN make
WORKDIR ../accelergy
RUN pip install --upgrade pip && \
	pip install .
WORKDIR ../accelergy-aladdin-plug-in/
RUN pip install .
WORKDIR ../accelergy-cacti-plug-in/
RUN pip install .
RUN cp -r ../cacti /opt/conda/envs/haccel/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
WORKDIR ../accelergy-table-based-plug-ins/
RUN pip install .
WORKDIR ../timeloop/src/
RUN ln -s ../pat-public/src/pat .
WORKDIR ..
RUN scons -j4 --accelergy --static
RUN cp build/timeloop-* /opt/conda/envs/haccel/bin
WORKDIR ../../..
RUN git clone https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git && \
	accelergy && \
	accelergyTables
RUN pip install git+https://github.com/Fibertree-Project/fibertree jupyter
ENV PATH $PATH:/opt/conda/evns/haccel/bin

WORKDIR /workspace/hetero-accel
SHELL ["/bin/bash", "--login", "-c"]

# ENTRYPOINT ["/bin/bash", "-i"]
