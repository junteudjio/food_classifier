FROM tensorflow/tensorflow:1.4.1

MAINTAINER Junior Teudjio "jun.teudjio@gmail.com"

USER junior

WORDIR /home/junior
COPY food_classifier/ /home/junior/food_classifier
EXPOSE 8081

########################################################################################################################
# Install dependencies
########################################################################################################################
WORKDIR /home/junior/food_classifier
RUN pip install -r requirements.txt

ENTRYPOINT /home/junior/food_classifier/run_preprocess_and_train.sh





