FROM pytorch/pytorch

ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip install cmake
RUN pip install gym[atari,accept-rom-license]==0.21.0