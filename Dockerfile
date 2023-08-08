# Docker file, Image, Container
FROM python:3.8

ADD diabetes_logistic_regression.py .
ADD requirements.txt .
ADD diabetes.csv .

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "./diabetes_logistic_regression.py"]
 