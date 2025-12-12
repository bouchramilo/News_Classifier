# dag
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

def _log_and_return(msg):
    print(msg)
    return msg



with DAG(
    "myFirstDag",
    description="DAG pour automatiser tout le pipeline : récupération des données, transformations, prédictions, stockage et visualisation.",
    start_date=datetime(2025,12,12),
    schedule_interval="*/1 * * * *",
    catchup=False,
    dagrun_timeout=timedelta(minutes=45),
) as dag : 
    task_1 = PythonOperator(
        task_id="task_1",
        python_callable=_log_and_return,
        op_kwargs={"msg": "Hello, Airflow!"}
    )

    task_2 = PythonOperator(
        task_id="task_2",
        python_callable=_log_and_return,
        op_kwargs={"msg": "Hello, Airflow!"}
    )

    task_1 >> task_2

