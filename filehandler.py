import logging
import os

from models.processing_models import ProcessingResult
from processors.excel_to_table_processor import EnhancedExcelToTableProcessor


def process_excel_file(myblob):
    """
    Enhanced process function for multiple sheets with OpenAI mapping
    """
    try:
        # Read blob content
        file_content = myblob.read()
        
        project_excel_processor = EnhancedExcelToTableProcessor()
        
        project_result, excel_result = project_excel_processor.process_and_store_project_info(
            file_content=file_content,
            filename=myblob.name
        )
        
        # Log results
        if project_result and excel_result.success:
            logging.info(f"Processing successful: {excel_result.message}")

        else:
            logging.error(f"Processing failed: {excel_result.message}")
        
        return excel_result
        
    except Exception as e:
        logging.error(f"Error in process_excel_file: {str(e)}")
        return ProcessingResult(
            success=False,
            message=f"Error processing blob: {str(e)}",
            records_processed=0
        )