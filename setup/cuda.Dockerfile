FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
	apt install -y git openssh-client wget curl pip vim htop tmux scons libboost-all-dev build-essential g++-9 \
			   scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev

# use this instead of cloning the repo (private right now)
RUN mkdir -p /workspace /workspace/chkpts
RUN mkdir -p /root/.ssh/ && \
	touch /root/.ssh/known_hosts && \
	ssh-keyscan github.com >> /root/.ssh/known_hosts
# TODO: maybe use this to copy private key into image
#RUN if [ -z "$DEVELOP_MODE" ]; then
#COPY sth 
WORKDIR /workspace
RUN --mount=type=secret,id=ssh_id,target=/root/.ssh/id_rsa \
	 git clone git@github.com:kompalas/hetero-accel.git
WORKDIR /workspace/hetero-accel

# install generalized assignment solver
RUN apt-get install apt-transport-https curl gnupg -y && \
	curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
	mv bazel-archive-keyring.gpg /usr/share/keyrings && \
	echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
	apt-get update && apt-get install bazel
RUN cd generalizedassignmentsolver && \
	bazel build -- //...

# install anaconda
ENV CONDA_DIR /opt/conda
RUN apt-get install --yes libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 && \
	apt-get clean all
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p $CONDA_DIR && \
	rm ~/anaconda.sh
# set path to conda
ENV PATH $CONDA_DIR/bin:$PATH

# the dependencies of all comprising tools are included in this environment
RUN conda init
RUN echo "conda activate haccel" >> /root/.bashrc
RUN conda update -y -n base -c defaults conda
RUN conda create -n haccel python=3.9
SHELL ["conda", "run", "-n", "haccel", "/bin/bash", "--login", "-c"]
RUN python3 -m pip install --upgrade pip && \
	python3 -m pip install -r setup/requirements.txt

# install timeloop-accelergy
RUN apt install -y scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev 
RUN --mount=type=secret,id=ssh_id,target=/root/.ssh/id_rsa \
	git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
WORKDIR accelergy-timeloop-infrastructure
RUN git submodule sync && \
	git submodule update --init && \
	sed -i '/git submodule/ c\' Makefile && \
	make pull
RUN cd src/cacti && \
	make
RUN cd src/accelergy && \
	pip install . 
RUN cd src/accelergy-aladdin-plug-in/ && \
	 pip install .
RUN cd src/accelergy-cacti-plug-in/ && \
	pip install . && \
	cp -r ../cacti /opt/conda/envs/haccel/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/
RUN cd src/accelergy-table-based-plug-ins/ && \
	pip install .
RUN cd src/timeloop/src/ && \
	ln -s ../pat-public/src/pat .
RUN cd src/timeloop && \
	scons -j4 --accelergy --static && \
	cp build/timeloop-* /opt/conda/envs/haccel/bin
# RUN git clone https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git && \
RUN accelergy && \
	accelergyTables -r /workspace/hetero-accel/data/ && \
	pip install git+https://github.com/Fibertree-Project/fibertree jupyter
ENV PATH $PATH:/opt/conda/evns/haccel/bin

WORKDIR /workspace/hetero-accel
RUN ln -s accelergy-timeloop-infrastructure/timeloop-accelergy-exercises/workspace/exercises/2020.ispass/timeloop+accelergy/ eyeriss.timeloop
SHELL ["/bin/bash", "--login", "-c"]
