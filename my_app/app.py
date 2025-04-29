import os
import sys
import hashlib

from fastapi import FastAPI
from my_app.response_schemas import PostGet, Response
from my_app.nn_answer import recommended_posts_nn_cat
from my_app.cat_answer import recommended_posts_simple_cat 
from my_app.downloading_functions import (get_exp_group, 
                                        load_models, 
                                        get_model_path,
                                        batch_load_sql,
                                        load_features)

from datetime import datetime
from catboost import CatBoostClassifier
from loguru import logger
import pandas as pd
import numpy as np


data_sql = load_features()
model_control = load_models()[0]
model_test = load_models()[1]

app = FastAPI()

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
		id: int, 
		time: datetime, 
		limit: int = 10) -> Response:

    
    logger.info(f'user group type defining')
    user_type = get_exp_group(id)
    
    if user_type == 'control': 
        logger.info('control group --- using simple cat')
        return recommended_posts_simple_cat(id, user_type, time, model_control, data_sql, limit)
    elif user_type == 'test': 
        logger.info('test group --- using NNet+cat')
        return recommended_posts_nn_cat(id, user_type, time, model_test, data_sql, limit)       
    else:
        logger.info('ValueError')
        raise ValueError('unknown group')
    
    
