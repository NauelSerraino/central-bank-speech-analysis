import logging
import os
import time
import psutil

class LoggerManager:
    def __init__(self, name: str, log_file: str = "pipeline.log", level: int = logging.INFO):
        """
        Initializes the LoggerManager with a configured logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(log_path, mode='a')
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def log_resource_usage(self, function_name: str, start_time: float) -> None:
        """
        Logs memory, CPU, and execution time.

        Args:
            function_name (str): Name of the executed function.
            start_time (float): Start time in seconds.
        """
        end_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        execution_time = end_time - start_time

        self.logger.info(
            f"{function_name} executed in {execution_time:.4f} sec | CPU: {cpu_usage}% | Memory: {memory_usage}%"
        )
