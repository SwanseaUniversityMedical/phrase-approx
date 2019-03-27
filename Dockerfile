FROM python:3

WORKDIR /usr/src/app

COPY . .

RUN pip install --proxy=${http_proxy} -r requirements.txt

CMD [ "python", "./entrypoint.py" ]
