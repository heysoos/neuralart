FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# install ffmpeg
RUN add-apt-repository -y ppa:savoury1/ffmpeg4 && \
    apt-get update -y && \
    apt-get install -y ffmpeg

RUN python -m pip install git+https://github.com/Po-Hsun-Su/pytorch-ssim.git

RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install \
    p_tqdm \
    tensorboard

RUN mkdir /notebooks/neuralart

# copy the content from local package folder to the docker package folder
COPY neuralart /notebooks/neuralart
COPY setup.py /notebooks