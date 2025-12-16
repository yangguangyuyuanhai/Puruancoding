import asyncio
import os
from datetime import datetime
from pathlib import Path

import helps.vairables as app_vars
from helps.logger import AppLogger

logger = AppLogger.get_logger(__name__)


async def stat_cache_folder_create_at():
    cache_p_dir = Path(app_vars.APP_CACHE_IMAGE_DIR)
    await asyncio.to_thread(os.makedirs, cache_p_dir, exist_ok=True)
    sub_dirs = [d for d in cache_p_dir.iterdir() if d.is_dir()]
    logger.debug(f"cache image dir: {sub_dirs}")
    return sub_dirs


async def clean(sub_dirs: list[Path]):
    now = datetime.now()
    for sub_dir in sub_dirs:
        folder_created_at = datetime.fromtimestamp(sub_dir.stat().st_ctime)
        delta = (now - folder_created_at).total_seconds() / 60
        if delta > app_vars.APP_CACHE_CLEAN_DURATION:
            await asyncio.to_thread(os.rmdir, sub_dir)
            logger.info(f"clean dir: {sub_dir}")


async def start():
    while True:
        logger.info("start cleaner")
        sub_dirs = await stat_cache_folder_create_at()
        await clean(sub_dirs=sub_dirs)
        await asyncio.sleep(1)
