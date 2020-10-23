#!/usr/bin/env
# coding:utf-8
import logging
import logging.handlers
import os

log_info = dict(
    # 日志保留路径,默认保留在项目跟目录下的logs文件
    log_dir='',
    # 日志级别,默认INFO级别,ERROR,WARNING,WARN,INFO,DEBUG
    log_level='DEBUG',
    # 每天生成几个日志文件,默认每天生成1个
    log_interval=1,
    # #日志保留多少天,默认保留7天的日志
    log_backupCount=7,
)


class Logging():
    __instance=None
    # ERROR,WARNING,WARN,INFO,DEBUG
    __logger_level_dic={
        'ERROR':logging.ERROR,
        'WARNING':logging.WARNING,
        'WARN':logging.WARN,
        'INFO':logging.INFO,
        'DEBUG':logging.DEBUG,
    }

    def __init__(self):
        self.__logger = logging.getLogger()
        # 日志文件名
        self.__filename = 'train.log'
        __log_info = log_info
        self.__log_dir = __log_info['log_dir']
        self.__log_level = __log_info['log_level']
        log_backupCount = __log_info['log_backupCount']
        log_interval = __log_info['log_interval']

        if log_backupCount is None or not isinstance(log_backupCount,int):
            log_backupCount = 7
        elif log_backupCount<0:
            log_backupCount = 7

        if log_interval is None or not isinstance(log_interval,int):
            log_interval = 1
        elif log_interval<0:
            log_interval = 7

        if self.__log_level == None or self.__log_level =='':
            self.__level = logging.INFO
        else:
            if self.__log_level.upper() not in self.__logger_level_dic.keys():
                self.__level = logging.INFO
            else:
                self.__level = self.__logger_level_dic[self.__log_level.upper()]


        if self.__log_dir == None or self.__log_dir =='':
            project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            self.__log_dir = os.path.join(project_root_path,'logs')
        if not os.path.exists(self.__log_dir):
            os.makedirs(self.__log_dir)
        # 创建一个handler，用于写入日志文件 (每天生成1个，保留10天的日志)
        fh = logging.handlers.TimedRotatingFileHandler(os.path.join(self.__log_dir,self.__filename), 'D', log_interval,log_backupCount)
        fh.suffix = "%Y%m%d-%H%M.log"
        fh.setLevel(self.__level)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(self.__level)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.__logger.setLevel(self.__level)
        # 给logger添加handler
        self.__logger.addHandler(fh)
        self.__logger.addHandler(ch)

    @classmethod
    def getLogger(cls)-> logging.Logger:
        if not cls.__instance:
            cls.__instance = Logging()
        return cls.__instance.__logger

if __name__ == '__main__':
    logger = Logging.getLogger()
    for i in range(100):
        logger.debug('aa')
        logger.error('bb')
        logger.info('cc')
        # time.sleep(1)
