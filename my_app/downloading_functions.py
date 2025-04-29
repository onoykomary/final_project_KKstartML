import os
import sys
import hashlib
from sqlalchemy import create_engine
from datetime import datetime
from catboost import CatBoostClassifier
from loguru import logger
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

#разбиентие на группы для AB теста
def get_exp_group(user_id: int) -> str:
    salt = 'splitting_for_experiment'
    if int(hashlib.md5((salt + str(user_id)).encode()).hexdigest(), 16) % 2 == 0:
        return 'control'
    return 'test'

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1": 
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path_simple_cat = get_model_path("model\model_control") #обычный catboost
    model_path_nn_cat = get_model_path("model\model_test") #catboost + nnet 
    
    simple_cat = CatBoostClassifier()
    nn_cat = CatBoostClassifier()
    
    simple_cat.load_model(model_path_simple_cat)
    nn_cat.load_model(model_path_nn_cat)
    return simple_cat, nn_cat  

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DATABASE")
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    engine = create_engine(db_url)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():
    TABLE_CUSTOM_POST = os.getenv("TABLE_CUSTOM_POST")
    TABLE_CUSTOM_USERS = os.getenv("TABLE_CUSTOM_USERS")
    TABLE_CUSTOM_POST_FEATURES = os.getenv("TABLE_CUSTOM_POST_FEATURES")
    
    # колонки c айдишниками, где взамодействие = лайк 
    query_liked = f"""SELECT DISTINCT user_id, post_id
					FROM feed_data
					WHERE action = 'like'
    				"""
    #кастомный датасет POST
    query_post = f"""SELECT *
                FROM {TABLE_CUSTOM_POST};
            	"""
    #кастомный датасет USER
    query_users = f"""SELECT * 
                FROM {TABLE_CUSTOM_USERS};
             """
    # датасет с постами, т.к. я удалила нужную информацию для ответа о тексте и топиках из моего кастомного
    original_post = f"""SELECT *
                FROM public.post_text_df;
            	"""
    #векторизированные тексты        
    query_post_features = f"""SELECT *
                FROM {TABLE_CUSTOM_POST_FEATURES};
            	"""
             
    logger.info('downloading feed_data with like-filtering')
    data_liked = batch_load_sql(query_liked)
    
    logger.info('downloading original data text')
    data_posts_original = batch_load_sql(original_post)    
    
    logger.info('downloading custom post info')
    data_posts = batch_load_sql(query_post)
    
    logger.info('downloading custom user info')
    data_users = batch_load_sql(query_users)
    
    logger.info("downloading custom posts' fetures info")
    data_post_features = batch_load_sql(query_post_features)
    
    return [data_liked, data_users, data_posts, data_posts_original, data_post_features]

