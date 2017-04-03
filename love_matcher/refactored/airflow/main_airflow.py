import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from love_matcher.refactored.airflow.task_classes import FeatureEngineeringTask, TuneTask, TrainTask

args = {
    'owner': 'santoine',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='love_matcher', default_args=args,
    schedule_interval=None)

feature_engineering = PythonOperator(
    task_id='love_matcher_feature_engineering',
    python_callable=FeatureEngineeringTask().run,
    dag=dag)

tune = PythonOperator(
    task_id='love_matcher_tune',
    python_callable=TuneTask().run,
    dag=dag)

train = PythonOperator(
    task_id='love_matcher_train',
    python_callable=TrainTask().run,
    dag=dag)

train.set_upstream(tune)
tune.set_upstream(feature_engineering)
