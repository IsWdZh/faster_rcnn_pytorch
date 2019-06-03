import logging
import os
import ctypes
import time

FOREGROUND_WHITE = 0x0007
FOREGROUND_BLUE = 0x01 # text color contains blue.
FOREGROUND_GREEN= 0x02 # text color contains green.
FOREGROUND_RED  = 0x04 # text color contains red.
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

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

        # self.STD_OUTPUT_HANDLE= -11
        # self.std_out_handle = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)

        logging.basicConfig(level=logging.INFO,
                            format=self.INFO_format,
                            datefmt=self.DATEFMT,
                            handlers=[logging.FileHandler(self.log_filename),
                                      logging.StreamHandler()])

    # def set_color(self, color):
    #     handle=self.std_out_handle
    #     bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    #     return bool

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message, color=FOREGROUND_YELLOW):
        # self.set_color(color)
        self.logger.warning(message)
        # self.set_color(FOREGROUND_WHITE)

    def error(self, message, color=FOREGROUND_RED):
        # self.set_color(color)
        self.logger.error(message)
        # self.set_color(FOREGROUND_WHITE)

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



