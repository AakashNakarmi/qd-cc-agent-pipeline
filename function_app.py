import azure.functions as func

# from filehandler import process_excel_file
from processors.excel_to_table_processor import process_excel_file
from utils.file_utils import is_excel_file
import logging
import os

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="mycontainer/{name}",
                  connection="saqddev01_STORAGE")
def blob_trigger(myblob: func.InputStream):
    logging.info(f"Processing blob: {myblob.name}, Size: {myblob.length} bytes")
    
    if not is_excel_file(myblob.name):
        logging.info(f"Skipping non-Excel file: {myblob.name}")
        return
    
    try:
        result = process_excel_file(myblob)
        logging.warning(f"Result: {result}")
        
        if result.success:
            logging.info(f"✅ {result.message}")
        else:
            logging.error(f"❌ {result.message}")
            
    except Exception as e:
        logging.error(f"Error in blob trigger: {str(e)}")