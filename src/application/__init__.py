import asyncio

from helps.logger import AppLogger

from .api import start as webapi
from .cleaner import start as cleaner

logger = AppLogger.get_logger(__name__)


async def launch() -> None:
    logger.info("start application")
    tasks = [webapi(), cleaner()]
    await asyncio.gather(*tasks)
    logger.info("end application")
