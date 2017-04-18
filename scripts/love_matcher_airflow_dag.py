import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from love_matcher.refactored.airflow.task_classes import FeatureEngineeringTask, TuneTask, TrainTask, EvalTask, \
    PredictTask

model_type = "Decision_Tree"

args = {
    'owner': 'Sandoine',
    'start_date': airflow.utils.dates.days_ago(2),
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
    python_callable=TuneTask(model_type=model_type).run,
    dag=dag)

train = PythonOperator(
    task_id='love_matcher_train',
    python_callable=TrainTask(model_type=model_type).run,
    dag=dag)

eval = PythonOperator(
    task_id='love_matcher_eval',
    python_callable=EvalTask(model_type=model_type).run,
    dag=dag)

predict = PythonOperator(
    task_id='love_matcher_predict',
    python_callable=PredictTask(model_type=model_type).run,
    dag=dag)

predict.set_upstream(train)
train.set_upstream(tune)
eval.set_upstream(train)
tune.set_upstream(feature_engineering)
