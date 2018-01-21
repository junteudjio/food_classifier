FROM tensorflow/tensorflow:1.4.1

MAINTAINER Junior Teudjio "jun.teudjio@gmail.com"

########################################################################################################################
# Install dependencies
########################################################################################################################
RUN apt-get update && \
      apt-get -y install sudo
RUN sudo apt-get --assume-yes install python-dev python-tk  python-numpy  python3-dev python3-tk python3-numpy

WORKDIR /home/junior
COPY food_classifier/ /home/junior/food_classifier

WORKDIR /home/junior/food_classifier
RUN pip install -r requirements.txt

EXPOSE 8383
RUN ["chmod", "+x", "/home/junior/food_classifier/run__preprocess_and_train.sh"]
RUN ["chmod", "+x", "/home/junior/food_classifier/run__food_classifier_service.sh"]

ENTRYPOINT /home/junior/food_classifier/run__preprocess_and_train.sh





