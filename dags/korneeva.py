import datetime
from pathlib import Path
import pendulum
import os

import requests
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
import subprocess
import sys


def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


install_package("cianparser")

import pandas as pd

import cianparser


def request_to_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """

    moscow_parser = cianparser.CianParser(location="Москва")

    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)[["url", "floor", "floors_count", "rooms_count", "price"]]
    return df


@dag(
    dag_id="process_flats_korneeva",
    schedule="0 0 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
)
def ProcessFlats():
    create_flat_table = SQLExecuteQueryOperator(
        task_id="create_flat_table",
        conn_id="tutorial_pg_conn",
        sql="""
            CREATE TABLE IF NOT EXISTS flat_korneeva (
                url VARCHAR(512) NOT NULL,
                floor INT,
                floors_count INT,
                rooms_count INT,
                price DECIMAL(15, 2) NOT NULL
            );""",
    )

    @task
    def get_data(n_rooms):
        data = request_to_cian(n_rooms)
        # NOTE: configure this as appropriate for your airflow environment
        root = Path("/opt/airflow/dags/files/")
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / "downloaded_flats.csv"
        data.to_csv(data_path, index=False)

        postgres_hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
        conn = postgres_hook.get_conn()
        cur = conn.cursor()
        with open(data_path, "r") as file:
            cur.copy_expert(
                "COPY flat_korneeva FROM STDIN WITH CSV HEADER DELIMITER AS ',' QUOTE '\"'",
                file,
            )
        conn.commit()

    (
        [create_flat_table]
        >> get_data(n_rooms=1)
        >> get_data(n_rooms=2)
        >> get_data(n_rooms=3)
    )


dag = ProcessFlats()
