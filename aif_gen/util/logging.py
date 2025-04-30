import logging
import pathlib
from typing import List, Optional, Union


def setup_basic_logging(
    log_file_path: Optional[Union[str, pathlib.Path]] = None,
    log_file_logging_level: int = logging.DEBUG,
    stream_logging_level: int = logging.INFO,
    dependancy_logging_level: int = logging.WARNING,
) -> None:
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_logging_level)
    stream_handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s %(message)s')
    )
    handlers.append(stream_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(filename=log_file_path, mode='a')
        file_handler.setLevel(log_file_logging_level)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(name)s - %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
        handlers=handlers,
    )

    if log_file_path is not None:
        # warning if the log file has size over 500MB
        log_file_path = pathlib.Path(log_file_path)
        if log_file_path.exists() and log_file_path.stat().st_size > 500 * 1024 * 1024:
            logging.warning(
                f'Log file {log_file_path} is over 500MB. Consider rotating or deleting it.'
            )

    # Disable verbose third-party loggers
    dependancy_loggers = ['httpx', 'httpcore', 'openai', 'elastic_transport']
    for logger_name in dependancy_loggers:
        logging.getLogger(logger_name).setLevel(dependancy_logging_level)
