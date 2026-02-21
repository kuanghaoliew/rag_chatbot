"""Sync a local langfuse.log file to a Langfuse server."""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils.langfuse_sync import LangfuseOTLPClient, LangfuseSyncService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse, os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Sync local log file to Langfuse")
    parser.add_argument("log_file", type=Path, help="Path to langfuse.log")
    parser.add_argument("--langfuse-host", default=os.getenv("LANGFUSE_HOST"))
    parser.add_argument("--langfuse-public-key", default=os.getenv("LANGFUSE_PUBLIC_KEY"))
    parser.add_argument("--langfuse-secret-key", default=os.getenv("LANGFUSE_SECRET_KEY"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--clear", action="store_true", help="Archive log file after successful sync")
    args = parser.parse_args()

    if not all([args.langfuse_host, args.langfuse_public_key, args.langfuse_secret_key]):
        logger.error("Langfuse host, public key, and secret key are all required")
        return 1

    if not args.log_file.exists():
        logger.error("File not found: %s", args.log_file)
        return 1

    client = LangfuseOTLPClient(
        langfuse_host=args.langfuse_host,
        public_key=args.langfuse_public_key,
        secret_key=args.langfuse_secret_key,
    )
    service = LangfuseSyncService(langfuse_client=client, batch_size=args.batch_size)

    total, successful = service.sync_log_file(args.log_file)
    logger.info("Synced %d/%d spans", successful, total)

    # Archive log file if --clear flag is set and sync was successful
    if args.clear and successful > 0:
        archive_dir = args.log_file.parent / "langfuse_synced"
        archive_dir.mkdir(exist_ok=True)
        archive_name = f"langfuse_synced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        archive_path = archive_dir / archive_name
        args.log_file.rename(archive_path)
        logger.info("Log file archived to %s", archive_path)

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())