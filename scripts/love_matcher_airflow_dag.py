import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from docs import *
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
tune.set_upstream(feature_engineering)

# if __name__ == '__main__':
#     start_date_time = datetime.datetime(2015, 6, 21)
#     end_date_time = datetime.datetime(2015, 6, 21)
#     ti = TaskInstance(tune, start_date_time)
#     ti.run(ignore_task_deps=True, ignore_ti_state=True, test_mode=True)
    #tune.run(start_date=start_date_time, end_date=end_date_time)

