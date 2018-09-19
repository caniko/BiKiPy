import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_DIR = os.path.normpath(os.path.join(BASE_DIR, "videos"))

print(CSV_DIR)
