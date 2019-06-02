FROM python:3.6

RUN mkdir /Workdir

WORKDIR /Workdir

RUN pip install gym \
    && pip install numpy

CMD ["/bin/bash"]
