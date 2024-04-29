FROM python:3.10

ENV PYSETUP_PATH="/opt/pysetup"
ENV VENV_PATH="/opt/pysetup/.venv"
ENV POETRY_HOME="/opt/poetry"

# pip install requirements
RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

CMD [ "bin/bash" ]

