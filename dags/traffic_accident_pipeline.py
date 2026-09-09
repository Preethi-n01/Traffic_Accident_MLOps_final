from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="traffic_accident_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["traffic-accident", "mlops"],
) as dag:

    verify_data = BashOperator(
        task_id="verify_analytical_data",
        bash_command=(
            "test -f "
            "/opt/airflow/project/data/analytical/accident_weather.csv"
        ),
    )

    pipeline_complete = BashOperator(
        task_id="pipeline_complete",
        bash_command="echo 'Traffic Accident MLOps pipeline completed successfully'",
    )

    verify_data >> pipeline_complete