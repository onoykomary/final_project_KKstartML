import os
import sys
import hashlib
from my_app.downloading_functions import get_exp_group, load_models

from typing import List
from my_app.response_schemas import PostGet, Response

from datetime import datetime
from catboost import CatBoostClassifier
from loguru import logger
import pandas as pd
import numpy as np


def recommended_posts_simple_cat(
		id: int, 
    exp_group: int, 
		time: datetime, 
    model: CatBoostClassifier,
    data_sql: tuple, 
		limit: int = 10) -> Response:

    
    logger.info(f'user {id}: downloading data for predictions')  
    data = {
            'user': data_sql[1], #custom user data
            'posts': data_sql[2], #custom post data
            'liked': data_sql[0], #liked posts to delete
            'texts': data_sql[3], #post info for answer
            'post_features': data_sql[4] #custom features, resulted from NNet
        }     
    
    #достаю нужного юзера по id и превращаю его данные в 2d array, чтобы сконкатить с post Features
    key_user = data['user'].loc[data['user'].user_id == id].drop('user_id', axis=1).values
    key_user = np.tile(key_user, (data['posts'].shape[0], 1))  
    key_user = pd.DataFrame(key_user, columns=data['user'].columns.to_list()[1:]) #без колонки user_id!
    
	
    logger.info('adding time features')
    key_user['hour'] = time.hour
    key_user['month'] = time.month
    key_user['day'] = time.day
    key_user['week_day'] = time.weekday()
    
    #соединяю информацию о юзере и постах
    final_data = pd.concat((data['posts'], key_user), axis=1).drop(['post_id'], axis=1)
    
    logger.info('model predicting')
    preds = model.predict_proba(final_data)[:, 1]
    final_data['predictions'] = preds
    
    #сортировка по предсказаниям, возвращаю колонку post_id
    logger.info('data processing')
    final_data['post_id'] = data['posts'].post_id
    final_data.sort_values(by='predictions', ascending=False, inplace=True)

    #нахожу post_id, которые уже пролайкал рассматриваемый нами key_user => из predictions это надо удалить
    logger.info('removing liked posts')
    post_ids = data['liked'].loc[data['liked'].user_id == id, 'post_id'].to_list()
    final_data = final_data[~final_data.post_id.isin(post_ids)]
    top_posts_idx = final_data.post_id.iloc[:limit].values # топ limit рекомендаций
    
    #преобразую в словарь
    logger.info('final data formation')
    recommended = [{'id': post_id, 
                'text': data['texts'].loc[data['texts'].post_id == post_id, 'text'].values[0],
                'topic': data['texts'].loc[data['texts'].post_id == post_id, 'topic'].values[0]} 
              for post_id in top_posts_idx]
    
    recomended_top = [PostGet(id=post_id, text=text, topic=topic) 
                      for post_id, text, topic 
                      in [(post['id'], post['text'], post['topic']) for post in recommended]]

    #валидация ответа
    return Response(exp_group=exp_group, recommendations=recomended_top)
  