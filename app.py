import ast
import base64
import configparser
import io
import os
import os.path
from io import StringIO
from datetime import datetime

import pandas as pd
from flask import Flask, make_response, request, render_template, Response

from prediction_projects.model_utilities import connect_redshift, load_features_from_redshift
from prediction_projects.predict import create_predictions_df

from youtube_project.youtube_search import community_based_search
from youtube_project.youtube_scraping_functions import build_youtube

import vault


# INITIALIZE VAULT PROCESS #

def get_vault_secrets():
    # initialize Vault secrets retrieval process
    vaultClient = vault.VaultClient()

    # retrieve redshift secrets
    redshift_credentials = {"host": vaultClient.get("REDSHIFT_HOST"),
                            "port": vaultClient.get("REDSHIFT_PORT"),
                            "dbname": vaultClient.get("REDSHIFT_DBNAME"),
                            "user": vaultClient.get("REDSHIFT_USER"),
                            "password": vaultClient.get("REDSHIFT_PASSWORD")
                            }

    # retrieve YouTube secrets
    youtube_api = {"API_KEY": vaultClient.get("YOUTUBE_API")}

    return redshift_credentials, youtube_api


# INITIALIZE FLASK APP #

app = Flask(__name__)


# PRE-APPLICATION FUNCTIONS #

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


def load_config(model_names, model_types):
    # store config files dictionary
    config_objects = {}

    # populate dictionary with config files
    for m_name in model_names:
        config_objects[m_name] = []
        for m_type in model_types:
            # get config file name
            con_name = "config_" + m_type + "_" + m_name + ".ini"
            # initialize configparser object
            config = configparser.ConfigParser()
            # read from config file (location = this file dir / configs)
            config.read(os.path.join(os.path.dirname(__file__), 'configs', con_name))
            # store config file
            config_objects[m_name].append(config)

    return config_objects


# CREATE WEB APPLICATIONS ROUTES #

# create home html page
@app.route('/', methods=['GET'])
def home():
    return render_template('index_home.html')


# create acceptance_rate html page
@app.route('/acceptance_rate', methods=['GET'])
def acceptance_rate_page():
    return render_template('index_acceptance_rate.html')


# create raid_qod html page
@app.route('/raid_qod', methods=['GET'])
def raid_qod_page():
    return render_template('index_raid_qod.html')


# create YouTube html page
@app.route('/youtube', methods=['GET'])
def youtube_search_page():
    return render_template('index_youtube_search.html')


# health check route
@app.route("/health")
def health_check():
    print("/health request")
    status_code = Response(status=200)
    return status_code


# CREATE WEB APPLICATIONS FUNCTIONS #

# create predict_acceptance_rate flask app
@app.route('/predict_acceptance_rate', methods=["POST"])
def predict_acceptance_rate():
    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df = pd.read_csv(StringIO(result))

    # define model names/types to predict with
    models_name = ["acceptance_rate"]
    model_types = ["clf"]

    # load config file
    config_objects = load_config(models_name, model_types)

    # retrieve config
    for config in config_objects.values():
        clf_config = config[0]

    # retrieve redshift credentials
    if clf_config["Model_Location"]["location"] == "GCS":
        credentials = get_vault_secrets()[0]
    elif clf_config["Model_Location"]["location"] == "LOCAL":
        credentials = None
    else:
        raise ValueError("No location was specified in the config file")

    # connect redshift
    cur = connect_redshift(credentials=credentials)

    # add features to csv file
    df = load_features_from_redshift(df=df,
                                     cur=cur,
                                     config=clf_config
                                     )

    # predict
    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")
    df = create_predictions_df(df, clf_config)
    print(
        f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finshed at: \033[1m{datetime.now()}\033[0m")

    # return predictions
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


# create predict_raid_qod flask app
@app.route('/predict_raid_qod', methods=["POST"])
def predict_raid_qod():
    """
    Background:
        This is a prediction pipeline which predict 1st using the Tutorial trained model then
        using the Deposits trained model.

    Usage:
        To switch between different pipelines - change the values in the "model_types" list, e.g.,
        ["clf", "reg"] - predict using classifier and regressor hurdle models
        ["reg"] - predict using regressor single model

    Note:
        The current usage process is a quick and temporary solution until a better one will be implemented
        (such as python parameter, move "model_types" list to config etc.)
    """

    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df = pd.read_csv(StringIO(result))

    # RAID TUTORIALS PIPELINE #
    # define model names/types to predict with
    models_name = ["raid_tutorials"]
    model_types = ["clf", "reg"]  # model_types = ["clf", "reg"]
    # load config files
    config_objects = load_config(models_name, model_types)
    # initialize raid_tutorials pipeline
    for config in config_objects.values():
        # retrieve classifier and regressor config files
        if len(model_types) == 1:
            clf_config = None
            reg_config = config[0]
        elif len(model_types) == 2:
            clf_config = config[0]
            reg_config = config[1]
        # retrieve redshift credentials
        if clf_config["Model_Location"]["location"] == "GCS":
            credentials = get_vault_secrets()[0]
        elif clf_config["Model_Location"]["location"] == "LOCAL":
            credentials = None
        else:
            raise ValueError("No location was specified in the config file")
        # connect redshift
        cur = connect_redshift(credentials=credentials)
        # add features to csv file
        df = load_features_from_redshift(df=df,
                                         cur=cur,
                                         config=clf_config
                                         )
        # predict
        print(
            f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")
        df = create_predictions_df(df, clf_config, reg_config)
        print(
            f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finished at: \033[1m{datetime.now()}\033[0m")

    # RAID DEPOSITS PIPELINE #
    # define model names/types to predict with
    models_name = ["raid_deposits"]
    model_types = ["clf", "reg"]  # model_types = ["clf", "reg"]
    # load config files
    config_objects = load_config(models_name, model_types)
    # initialize raid_tutorials pipeline
    for config in config_objects.values():
        # retrieve classifier and regressor config files
        if len(model_types) == 1:
            clf_config = None
            reg_config = config[0]
        elif len(model_types) == 2:
            clf_config = config[0]
            reg_config = config[1]
        # predict
        print(
            f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - started at: \033[1m{datetime.now()}\033[0m")
        df = create_predictions_df(df, clf_config, reg_config)
        print(
            f"Prediction process for pipeline {', '.join(models_name)} ({', '.join(model_types)}) - finished at: \033[1m{datetime.now()}\033[0m")

    # return predictions
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


# create youtube_search flask app
@app.route('/youtube_search', methods=["POST"])
def youtube_search():
    # request csv file
    f = request.files['data_file']
    if not f:
        return "No file"

    # read data from csv file
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    stream.seek(0)
    result = transform(stream.read())

    # convert the data into df
    df_input = pd.read_csv(StringIO(result))

    # # initialize df to store results
    df_result = pd.DataFrame()

    # define model names/types to use with it
    models_name = ["search"]
    model_types = ["youtube"]

    # retrieve config file
    config_objects = load_config(models_name, model_types)

    # load config
    for config in config_objects.values():
        youtube_config = config[0]

    # retrieve YouTube API
    if youtube_config["Model_Location"]["location"] == "GCS":
        credentials = get_vault_secrets()[0]
        api_key = get_vault_secrets()[1]
    elif youtube_config["Model_Location"]["location"] == "LOCAL":
        credentials = None
        api_key = None
    else:
        raise ValueError("No location was specified in the config file")

    # construct YouTube API object
    youtube = build_youtube(api_key=api_key)

    # iterate over channels url input and retrieve recommendations (relative channels)

    print(f"Youtube search process for all channels - started at: \033[1m{datetime.now()}\033[0m")
    df_result = community_based_search(
        youtube=youtube,
        df_starting_point=df_input,
        n_recommendations=int(youtube_config["Data_Processing"]["n_recommendations"]),
        n_videos_per_request=int(youtube_config["Data_Processing"]["n_videos_per_request"]),
        n_comments_per_video=int(youtube_config["Data_Processing"]["n_comments_per_video"])
    )

    print(f"Youtube search process for all channels - finished at: \033[1m{datetime.now()}\033[0m")

    # prepare df to redshift (extract provider_id from given url --> so we can join bi_db.creators.provider_id)
    df_result["provider_id"] = df_result["Channel URL (Output)"].str.split('/').str[-1]

    # connect redshift
    cur = connect_redshift(credentials=credentials)

    # add features to csv file
    df = load_features_from_redshift(df=df_result,
                                     cur=cur,
                                     config=youtube_config
                                     )

    # rename and reorder columns
    df.rename(columns={'boss_channel_id': 'BOSS Channel ID (Output)'}, inplace=True)
    df = df[["Channel URL (Input)", "Channel URL (Output)", "BOSS Channel ID (Output)", "Score"]]

    # return predictions
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
