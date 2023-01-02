import os
from pathlib import Path

from bikipy.ingress.utils.io import BIKIPY_SETTINGS_FILE_NAME
from bikipy.ingress.utils.settings.update import update_settings

for category_dir in os.listdir():
    if category_dir == "live":
        continue

    for project_dir in Path(category_dir).iterdir():
        if (project_dir / BIKIPY_SETTINGS_FILE_NAME).exists():
            continue
        print(project_dir)
        update_settings(project_dir, delete_outdated=True)
