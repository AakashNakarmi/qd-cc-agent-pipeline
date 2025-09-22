import json
import os

# Load environment variables from local.settings.json if available
# def load_local_settings():
#     try:
#         with open("local.settings.json") as f:
#             settings = json.load(f)
#             values = settings.get("Values", {})
#             for key, value in values.items():
#                 os.environ.setdefault(key, value)
#     except FileNotFoundError:
#         # local.settings.json not found, continue with system env vars
#         pass

# load_local_settings()


BLOB_CONNECTION_STRING = os.environ.get('BLOB_CONNECTION_STRING')
AZURE_SQL_SERVER = os.environ.get('AZURE_SQL_SERVER')
AZURE_SQL_DATABASE = os.environ.get('AZURE_SQL_DATABASE')
AZURE_SQL_USERNAME = os.environ.get('AZURE_SQL_USERNAME')
AZURE_SQL_PASSWORD = os.environ.get('AZURE_SQL_PASSWORD')
AZURE_OPENAI_ENDPOINT= os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
