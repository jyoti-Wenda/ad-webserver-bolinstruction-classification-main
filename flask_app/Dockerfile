FROM python:3.8-slim

# layer caching for faster builds
COPY requirements.txt /
RUN pip install -r /requirements.txt

#COPY app.py /app.py
COPY files /flask_app/files
ADD . /flask_app
WORKDIR /flask_app
RUN chmod -R 777 /flask_app/files
RUN chown 1000:1000 -R /flask_app/files

#CMD gunicorn --bind=0.0.0.0:5000 --timeout=0 --certfile=ssl/certificateChain.crt --keyfile=ssl/private.key app:app
CMD gunicorn --bind=0.0.0.0:5000 --timeout=0 app:app

# CMD flask run --host=0.0.0.0 --cert=ssl/certificateChain.crt --key=ssl/private.key

# CMD gunicorn \
#  --bind 0.0.0.0:8008 \
#  --log-level DEBUG \
#  app:app
