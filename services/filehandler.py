import logging
from models.processing_models import ProcessingResult
from processors.excel_to_table_processor import EnhancedExcelToTableProcessor


def process_excel_file(_blob,blob_details):
    """
    Enhanced process function for multiple sheets with OpenAI mapping
    """
    logging.info("Processing blob", extra={"blob_name": blob_details['name'], "blob_size": _blob.length})

    try:
        file_content = _blob.read()
        logging.info("Initializing processor", extra={"processor": EnhancedExcelToTableProcessor.__name__})
        project_excel_processor = EnhancedExcelToTableProcessor()

        logging.info("Processing and storing project info", extra={"file_name": blob_details['name']})
        excel_result = project_excel_processor.process_and_store_project_info(
            file_content=file_content,
            filename=blob_details['name'],
            blob_details=blob_details
        )
        logging.info(f"###################### Ended processing file: {blob_details['name']} ######################")
        return excel_result
    except Exception as e:
        logging.exception("Error processing blob", extra={"error": str(e)})
        return ProcessingResult(
            success=False,
            message=f"Error processing blob: {str(e)}",
            records_processed=0
        )
