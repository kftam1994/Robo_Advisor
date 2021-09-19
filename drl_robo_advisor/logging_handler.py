import logging
from pathlib import Path
from datetime import datetime

class LoggingHandler:
    """

    LoggingHandler to create logger


    Methods
    -------
    get_logger()
        Return the existing logger
    shut_down_logging()
        Shutdown the existing logger after program complete
    exception_hook(exc_type, exc_value, exc_traceback)
        Record exception to existing logger
    log_subprocess_output(logger,pipe)
        Log the output from command line call

    """
    def __init__(self,runs_path):
        """
        
        Parameters
        ----------
        runs_path : pathlib.Path
            path of runs folder

        """
        now = datetime.now()
        self._logger_filename_suffix = now.strftime("%Y%m%d_%H%M%S")

        logger_filename = f'log_{self._logger_filename_suffix}.log'
        self._logger_filename = logger_filename

        logger_path = Path(runs_path,self._logger_filename)
	
        logging.basicConfig()

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(logger_path, mode="a")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
		
        self._logger = logger

    @property
    def logger_filename(self):
        """

        Returns
        -------
        logger_filename : str
            file name of logger

        """
        return self._logger_filename

    @property
    def logger_filename_suffix(self):
        """

        Returns
        -------
        logger_filename_suffix : str
            suffix which is the string of datetime in the logger's file name

        """
        return self._logger_filename_suffix

    def get_logger(self):
        """
        
        Returns
        -------
        logger : Logger
            Return the existing logger
        
        """
        return self._logger

    def shut_down_logging(self):
        """
        
        Shutdown the existing logger
        
        """
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        logging.shutdown()

    def exception_hook(self,exc_type, exc_value, exc_traceback):
        """

        Record the exception in log
        Reference: https://stackoverflow.com/questions/1508467/log-exception-with-traceback

        Parameters
        ----------
        exc_type : str

        exc_value : str

        exc_traceback : traceback

        """
        self._logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    @staticmethod
    def log_subprocess_output(logger,pipe):
        """

        Log the output from command line call

        Parameters
        ----------
        pipe : subprocess.PIPE
            log subprocess output

        """
        for line in iter(pipe.readline, b''): # b'\n'-separated lines
            logger.info('got line from subprocess: %r', line)