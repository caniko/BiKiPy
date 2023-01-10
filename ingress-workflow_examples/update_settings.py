import os
from pathlib import Path

from bikipy.ingress.utils.settings.update import update_settings

for category_dir in os.listdir():
    category_dir = Path(category_dir)
    if category_dir.name == "live" or category_dir.is_file():
        continue

    for project_dir in category_dir.iterdir():
        update_settings(project_dir, delete_outdated=True)
