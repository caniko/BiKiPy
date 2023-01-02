import os
from glob import iglob
from pathlib import Path

from bikipy.ingress.utils.io import BIKIPY_SETTINGS_FILE_NAME
from bikipy.ingress.utils.settings.update import update_settings

# for f in iglob("**/**/*.toml"):
#     os.remove(f)

for category_dir in os.listdir():
    category_dir = Path(category_dir)
    if category_dir.name == "live" or category_dir.is_file():
        continue

    for project_dir in category_dir.iterdir():
        if (project_dir / BIKIPY_SETTINGS_FILE_NAME).exists():
            with open(project_dir / BIKIPY_SETTINGS_FILE_NAME, "r") as infile:
                if not infile.readlines():
                    continue
        print(project_dir)
        update_settings(project_dir, delete_outdated=True)
