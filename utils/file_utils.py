
import os


def is_excel_file(filename: str) -> bool:
    """
    Check if the file is an Excel file based on its extension.
    Supports both .xlsx and .xls formats.
    """
    if not filename:
        return False
    
    excel_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
    file_extension = os.path.splitext(filename.lower())[1]
    return file_extension in excel_extensions