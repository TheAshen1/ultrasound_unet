FROM tensorflow/tensorflow:nightly
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip install opencv-python sklearn pandas argparse tqdm

WORKDIR /ultrasound_unet

COPY data.py .
COPY model.py .
COPY main.py .