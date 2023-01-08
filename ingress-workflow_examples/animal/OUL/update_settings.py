import os
from glob import iglob
from pathlib import Path

from bikipy.ingress.utils.io import BIKIPY_SETTINGS_FILE_NAME
from bikipy.ingress.utils.settings.update import update_settings

update_settings(".", delete_outdated=True)
