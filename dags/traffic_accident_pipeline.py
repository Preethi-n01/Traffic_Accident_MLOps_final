from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/mnt/c/Users/shaiza samreen/Traffic_Accident_MLOps"


with DAG(
    dag_id="traffic_accident_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["traffic-accident", "mlops"],
) as dag:

    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=(
            f"cd '{PROJECT_DIR}' && "
            "python src/ingestion/ingest_data.py"
        ),
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd '{PROJECT_DIR}' && "
            "python src/validation/validate_data.py"
        ),
    )

    transform_data = BashOperator(
        task_id="transform_data",
        bash_command=(
            f"cd '{PROJECT_DIR}' && "
            "python src/transformation/transform_data.py"
        ),
    )

    load_database = BashOperator(
        task_id="load_database",
        bash_command=(
            f"cd '{PROJECT_DIR}' && "
            "python src/load_to_database.py"
        ),
    )

    verify_data = BashOperator(
        task_id="verify_analytical_data",
        bash_command=(
            "test -f "
            f"'{PROJECT_DIR}/data/analytical/accident_weather.csv'"
        ),
    )

    pipeline_complete = BashOperator(
        task_id="pipeline_complete",
        bash_command=(
            "echo 'Traffic Accident MLOps pipeline completed successfully'"
        ),
    )


    ingest_data >> validate_data >> transform_data >> load_database >> verify_data >> pipeline_complete