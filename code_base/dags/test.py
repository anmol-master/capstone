from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def example_task():
    print("This is a sample task")

default_args = {
    'description':'A DAG to orchestrate data',
    'start_date': datetime(2025,9,15),
    'catchup': False,
}

dag = DAG(
    dag_id= 'test',
    default_args= default_args,
    schedule=timedelta(minutes=5)  
)

with dag:
    task1 = PythonOperator(
        task_id= 'sample_task',
        python_callable= example_task
    )