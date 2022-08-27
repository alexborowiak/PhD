import logging
import sys
LOG_FORMAT = "%(message)s"
logging.basicConfig(format=LOG_FORMAT, filemode='w')
logger = logging.getLogger()

# Making the log message appear as a print statements rather than in the jupyter cells
logger.handlers[0].stream = sys.stdout