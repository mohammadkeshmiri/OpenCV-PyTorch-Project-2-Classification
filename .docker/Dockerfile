FROM python-opencv-pytorch:1

RUN pip3 install notebook
RUN pip3 install pandas
RUN pip3 install -U scikit-learn
RUN pip3 install tensorboard

ENV QT_X11_NO_MITSHM=1

WORKDIR /root/Project2_Kaggle_Competition_Classification/src

CMD ["bash"]