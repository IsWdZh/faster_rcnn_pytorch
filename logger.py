import logging
import os
import time


class Log():
    def __init__(self):
        self.now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.INFO_format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"
        self.DATEFMT = "[%Y-%m-%d %H:%M:%S]"
        self.logpath = os.path.join(os.getcwd(), "output/log")
        self.log_filename = os.path.join(self.logpath, self.now_time + ".log")
        self.logger = logging.getLogger("train")
        logging.basicConfig(level=logging.INFO,
                            format=self.INFO_format,
                            datefmt=self.DATEFMT,
                            handlers=[logging.FileHandler(self.log_filename),
                                      logging.StreamHandler()])

    def _info(self, message):
        self.logger.info(message)

    def _debug(self, message):
        self.logger.debug(message)

    def _warning(self, message):
        self.logger.warning(message)

    def _error(self, message):
        self.logger.error(message)

class Logger():
    def __init__(self):
        self.log = Log()

    def info(self, message):
        self.log._info(message)

    def debug(self, message):
        self.log._debug(message)

    def warning(self, message):
        self.log._warning(message)

    def error(self, message):
        self.log._error(message)






