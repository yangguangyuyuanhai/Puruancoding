import argparse
import asyncio

from application import launch
from helps.logger import AppLogger

logger = AppLogger.get_logger(__name__)


async def prepare_logger(level: str):
    AppLogger.set_logger_level(level)
    AppLogger.init_logger_config()


async def main(_args: argparse.Namespace):
    level = _args.log_level
    await prepare_logger(level)
    await launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beeble is for post handling the AI recogonization"
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        dest="log_level",
        default="info",
        help="set the log level",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
