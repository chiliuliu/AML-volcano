import os
import logging


def init_logging(PATH, name_experiment):
    """
    Initialising loggin
    :param PATH, str: Base path
    :param name_experiment, str: name of experiment, folder name
    :return: logger
    """
    # create folder for experiment
    os.makedirs(os.path.join(PATH, name_experiment), exist_ok=False)
    # Setting up the log file
    logfile = os.path.join(PATH, name_experiment, name_experiment + "_log_file.log")

    logger = logging.getLogger("phcp")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "- %(asctime)s [%(levelname)s] -- " "[- %(process)d - %(name)s]%(message)s"
    )

    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Logger initialised...")

    return logger
