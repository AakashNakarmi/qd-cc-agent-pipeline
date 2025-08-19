# import logging
# import azure.functions as func
# from processors.excel_to_table_processor import excel_processor, process_excel_file2

# def process_excel_file(myblob: func.InputStream):
#     """
#     Process the Excel file. Add your Excel processing logic here.
#     """
#     try:
#         # Read blob content
#         result = process_excel_file2(myblob)
        
#         # Process Excel and store in Azure Table
#         # result = excel_processor.process_excel_to_table(
#         #     file_content=file_content,
#         #     filename=myblob.name,
#         #     max_records=5  # Top 5 records
#         # )
        
#         logging.info(f"Successfully processed Excel file: {myblob.name}")
        
        
#     except Exception as e:
#         logging.error(f"Error processing Excel file {myblob.name}: {str(e)}")
        
#     return result