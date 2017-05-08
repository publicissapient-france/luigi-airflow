import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from love_matcher_exercises.run.airflow.task_classes import FeatureEngineeringTask, TrainTask, EvalTask, \
    PredictTask

model_type = "Decision_Tree"

args = {
    'owner': 'Sandoine',
    'start_date': airflow.utils.dates.days_ago(2),
}

dag = DAG(
    dag_id='love_matcher', default_args=args,
    schedule_interval=None)

feature_engineering = None
train = None
eval = None
predict = None

