#-*- coding: utf-8 -*-
from kuanke.user_space_api import *
import redis
import json
import time

# Redis 配置
REDIS_CONFIG = {
    "host": "YOUR_REDIS_HOST",
    "port": 6379,
    "password": "YOUR_REDIS_PASSWORD",
    "decode_responses": True
}  #这里 键入你的服务器 和端口 还有 密码


def send_signal_to_redis(stock_code, direction,  remark):
    try:
        conn = redis.Redis(**REDIS_CONFIG)
        msg = {
            "type": "order",
            "stock_code": stock_code,
            "quantity": direction,
            "remark": remark,
            "signal_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        result = conn.lpush("data_queue", json.dumps(msg, ensure_ascii=False))
        return result > 0
    except Exception as e:
        print(f"Redis 发送失败：{e}")
        return False

def order_zzy(security, quantity, style=None, pindex=0):
    send_signal_to_redis(security,  quantity, "order")
    return order(security, quantity, style=style, pindex=pindex)

def order_target_zzy(security, quantity, style=None, pindex=0):
    send_signal_to_redis(security,   quantity, "order_target")
    return order_target(security, quantity, style=style, pindex=pindex)

def order_value_zzy(security, quantity, style=None, pindex=0):
    send_signal_to_redis(security,  quantity, "order_value")
    return order_value(security, quantity, style=style, pindex=pindex)
def order_target_value_zzy(security, quantity, style=None, pindex=0):
    send_signal_to_redis(security,   quantity, "order_target_value")
    return order_target_value(security, quantity, style=style, pindex=pindex)


'''

from send_to_redis import (
order_zzy as order, order_target_zzy as order_target, order_value_zzy as order_value,
order_target_value_zzy as order_target_value
)

'''
