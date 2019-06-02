import logging
import os
import time

class Logger():
    def __init__(self):
    # def initialize(self):
        self.logger = logging.getLogger("train")
        self.now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.INFO_format = "%(asctime)s %(name)s %(levelname)s %(message)s" # 8s靠右,-6s靠左
        self.DATEFMT = "[%Y-%m-%d %H:%M:%S]"
        self.logpath = os.path.join(os.getcwd(), "output/log")
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)
        self.log_filename = os.path.join(self.logpath, self.now_time + ".log")

        logging.basicConfig(level=logging.INFO,
                            format=self.INFO_format,
                            datefmt=self.DATEFMT,
                            handlers=[logging.FileHandler(self.log_filename),
                                      logging.StreamHandler()])

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    # @staticmethod
    def get_logger(self):
        if not self.logger.handlers:
            # file
            file_handler = logging.FileHandler(self.log_filename)
            file_handler.setFormatter(self.INFO_format)

            # console
            console_handler = logging.StreamHandler()
            console_handler.formatter = self.INFO_format

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self.logger.setLevel(logging.INFO)

        return self.logger


logger = Logger()

def get_logger():
    return logger



