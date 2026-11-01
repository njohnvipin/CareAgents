"""Load fictional demo data: `python -m careagents.seed`. Refuses to run in staging/prod."""

import asyncio
from importlib import resources

import asyncpg

from careagents.config import get_settings


async def seed() -> None:
    settings = get_settings()
    if settings.env in ("staging", "prod"):
        raise SystemExit("refusing to load demo data outside dev/test")
    con = await asyncpg.connect(settings.database_url)
    try:
        await con.execute(resources.files("careagents.seeds").joinpath("demo.sql").read_text())
    finally:
        await con.close()


if __name__ == "__main__":
    asyncio.run(seed())
    print("seeded demo data")
