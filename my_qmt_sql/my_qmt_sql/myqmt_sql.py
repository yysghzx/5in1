#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码
# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入 

#####################只需要修改自己的用户名和密码，别的无需更改#######################################
from kuanke.user_space_api import *
import pymssql
from typing import Optional, Union

class MyTrade():
    
    def __init__(self): 
        
        self.conn = pymssql.connect('YOUR_SQL_HOST', 'YOUR_SQL_USER', 'YOUR_SQL_PASSWORD', 'YOUR_SQL_DATABASE')  # 建立连接

        
    def update(self,code,quantity,types):
    
        # 创建数据库连接
        conn = self.conn

    # 生成游标对象 cursor
        cursor = conn.cursor() # 创建一个游标对象python里的sql语句都要通过cursor来执行
           
        value1=''

        if code[-4:]=='XSHE':
            code1=code[:-4]+"SZ"
        else:
            code1=code[:-4]+"SH"
        name1=get_security_info(code).display_name


        now = datetime.datetime.now()
        now1=now.strftime('%Y-%m-%d %H:%M:%S')
        #now = context.current_dt.strftime('%Y-%m-%d %H:%M:%S')
        value1=value1 +"('"+ name1 +"','"+ code1 +"','"+ types+"','"+ str(quantity)+"','"+now1+"','未分类')"


        str_sql1="insert into trade (name,code,type,num,date,fenlei)  VALUES "+value1

        if value1!='':
            cursor.execute(str_sql1)  # 执行sql语句
            conn.commit()  # 执行update操作时需要写这个，否则就会更新不成功
            #row = cursor.fetchone()  # 读取查询结果

            cursor.close()
            conn.close()   


    

def order_zzy(security: str, quantity: int,style = None,pindex=0):  #按股数下单.
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order')
    _order = order(security, quantity, style=style,pindex=pindex)  
    return _order

def order_target_zzy(security: str, quantity: int,style = None,pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_target')
    _order = order_target(security, quantity,style=style,pindex=pindex)
    return _order

def order_value_zzy(security: str, quantity: int,style = None,pindex=0):  #按价值下单
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_value')
    _order = order_value(security, quantity, style=style,pindex=pindex)
    return _order

def order_target_value_zzy(security: str, quantity: int,style = None,pindex=0):  #目标价值下单
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_target_value')
    _order = order_target_value(security, quantity, style=style,pindex=pindex)
    return _order
