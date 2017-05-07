# Making an application ready with Luigi and Airflow

## Introduction

This project aims to teach and compare *luigi* and *airflow* projects through a machine
learning workshop

## Execution environment

The complete execution environment for Airflow implies many servers : Redis, RabbitMQ,
PostgreSQL, an Airflow Scheduler and at least one Airflow worker. So, to make
it simpler, we use *docker-compose*.

**Prerequisites :** To set up your environment, follow instructions in the `docker` subdirectory.

## Source code explanations (branch `master`)
TODO

## Exercises : machine learning project

1. Go to luigi-airflow/love_matcher_exercises/preprocessing
    - Open raw_data_preprocessing.py and complete exercises 1.1 and 1.2
2. Go to luigi-airflow/love_matcher_exercises/feature_engineering
    - Open feature_engineering.py to see what are the new features added to dataset
3. Go to luigi-airflow/love_matcher_exercises/utils
    - Open split_train_test.py and complete exercise 2.1
4. Go to luigi-airflow/love_matcher_exercises/training
    - Open training.py and complete exercise 3.1
5. Go to luigi-airflow/love_matcher_exercises/evaluation
    - Open evaluation.py and complete exercises 4.1 and 4.2
6. Go to luigi-airflow/love_matcher_exercises/predictions
    - Open predictions.py and complete exercise 5.1

## Exercises : luigi
Useful docs :
http://luigi.readthedocs.io/en/stable/example_top_artists.html
http://luigi.readthedocs.io/en/stable/api/luigi.local_target.html

1. Go to luigi-airflow/love_matcher_exercises/run/luigi
    - Open main_luigi.py and complete all tasks 

