import logging
import sys

def setup_logging():
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)03d >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
