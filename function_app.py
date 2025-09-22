import azure.functions as func
import logging

from services.filehandler import process_excel_file
from utils.file_utils import is_excel_file

app = func.FunctionApp()

@app.blob_trigger(
    arg_name="_blob", 
    # path="docs/{name}", 
    path="docs/{project_oriented_country}/{project_start_year}/{project}/{name}", 
    connection="BLOB_CONNECTION_STRING"
)
def ExcelBlobProcessor(_blob: func.InputStream):
    if not is_excel_file(_blob.name):
        logging.info(f"Skipping file (not Excel): {_blob.name}")
        return
    
    parts = _blob.name.split("/")
    
    try:
        blob_details = {
            "project_oriented_country": parts[1],
            "project_start_year": parts[2],
            "project_folder_name": parts[3],
            "name": parts[4],
            "file_path": _blob.name
        }
        
        result = process_excel_file(_blob, blob_details)
        if result.success:
            logging.info(f"Processing succeeded: {result.message}")
        else:
            logging.error(f"Processing failed: {result.message}")

    except Exception as e:
        logging.exception(f"Unexpected error in blob trigger for {_blob.name}: {e}")

