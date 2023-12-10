FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-knugs

COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install -y git
RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--shm-size="8g""]
