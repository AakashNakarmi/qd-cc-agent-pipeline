# processors/excel_to_table_processor.py
import time
import uuid
import pandas as pd
import logging
import uuid
from io import BytesIO
from typing import List, Dict, Any, Optional
from models.open_ai_services import OpenAIService
from services.azure_sql_service import AzureSQLService
from models.processing_models import BOQSchema, ProcessingResult, Section
from utils.boq_utils import merge_boq_descriptions_advanced




class EnhancedExcelToTableProcessor:
    def __init__(self):
        self.service_openai = OpenAIService()
        self.table_service = AzureSQLService()
        self.boq_schema = BOQSchema()



    def process_and_store_project_info(self, file_content: bytes, filename: str, blob_details: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Complete workflow: Extract project info from first sheet and store in Azure Table
        
        Args:
            file_content: Excel file content as bytes
            filename: Name of the Excel file
            blob_details: Details extracted from the blob path

        Returns:
            Dict containing stored project information with project_id, or None if failed
        """
        try:
            project_info={}
            # Extract project information from first sheet
            # project_info = self.process_first_sheet_for_project_info(file_content, filename) #skip
            project_info['project_oriented_country'] = blob_details.get('project_oriented_country')
            project_info['project_start_year'] = blob_details.get('project_start_year') 
            project_info['project_folder_name'] = blob_details.get('project_folder_name')
            project_info['file_path'] =  blob_details.get('file_path')
            project_id = f"{project_info['project_oriented_country']}_{project_info['project_start_year']}_{project_info['project_folder_name']}"
            project_info['project_id'] = project_id
            if not project_info:
                logging.error("Could not extract project information from first sheet")
                return None
            
            # Store the project information
            
            if not self.table_service._is_project_id_present(project_id):
                project_result = self.store_project_with_retry(project_info, filename)
            else:
               logging.info(f"Skipping project insert: Project with ID '{project_id}' already exists in the database.")
            
            excel_result = self.process_excel_to_table(
                file_content = file_content,
                filename = filename,
                project_id = project_id
            )
            
            if excel_result.success:
                return excel_result
            else:
                logging.error("Failed to store project information")
                return None
                
        except Exception as e:
            logging.error(f"Error in process_and_store_project_info: {str(e)}")
            return None
        
    #######################################################    
    ##### EXTRACTION OF PROJECT INFO FROM FIRST SHEET #####
    ##### RELATED FUNCTIONS ###############################
    #######################################################    
    
    def process_first_sheet_for_project_info(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        try:
            # Read Excel file
            excel_data = BytesIO(file_content)
            excel_file = pd.ExcelFile(excel_data)
            
            if not excel_file.sheet_names:
                logging.error("No sheets found in Excel file")
                return None
            
            # Get the first sheet
            first_sheet_name = excel_file.sheet_names[0]
            logging.info(f"Processing first sheet '{first_sheet_name}' for project information")
            
            # Read the first sheet without assuming header location
            df_raw = pd.read_excel(excel_file, sheet_name=first_sheet_name, header=None)
            
            if df_raw.empty:
                logging.error(f"First sheet '{first_sheet_name}' is empty")
                return None
            
            # Convert first 15 rows to string representation for OpenAI analysis
            # Usually project info is at the top of the sheet
            preview_rows = min(15, len(df_raw))
            sheet_preview = df_raw.head(preview_rows).to_string(index=True, na_rep='')
            
            logging.info(f"First sheet '{first_sheet_name}' preview for project info extraction:\n{sheet_preview}")
            
            # Use OpenAI to extract project information
            print("\n--- Parameters passed---")
            print(f"sheet_preview (first 200 chars):\n{sheet_preview[:200]}") 
            print(f"sheet_name: {first_sheet_name}")
            print(f"filename: {filename}")
            print("------------------------------------------------------------\n")
            project_info = self.service_openai._extract_project_info_with_openai(sheet_preview, first_sheet_name, filename)
            print(project_info)
            
            if project_info:
                logging.info(f"Successfully extracted project information: {project_info}")
                return project_info
            else:
                logging.warning("Could not extract project information from first sheet")
                return None
                
        except Exception as e:
            logging.error(f"Error processing first sheet for project info: {str(e)}")
            return None

    def store_project_with_retry(self, project_info: Dict[str, Any], filename: str, max_retries: int = 3) -> Optional[str]:
        """
        Store project information with retry logic
        
        Args:
            project_info: Dictionary containing project information
            filename: Source filename
            max_retries: Maximum number of retry attempts
        
        Returns:
            str: Project ID if successful, None otherwise
        """
        project_id = None
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generate project ID if not provided
                if not project_info.get('project_id'):
                    project_info['project_id'] = str(uuid.uuid4())
                
                project_id = project_info['project_id']
                
                # Attempt to store the project
                success = self.table_service.insert_project_record(project_info, filename)
                
                if success:
                    logging.info(f"Successfully stored project '{project_info['project_name']}' with ID: {project_id}")
                    return project_id
                else:
                    retry_count += 1
                    logging.warning(f"Failed to store project, retry {retry_count}/{max_retries}")
                    
            except Exception as e:
                retry_count += 1
                logging.error(f"Error storing project, retry {retry_count}/{max_retries}: {str(e)}")
        
        logging.error(f"Failed to store project after {max_retries} retries")
        return None

    
    ############################################################
    ##### EXTRACTION OF SECTION AND BOQ INFO FROM ALL SHEETS ###
    ##### RELATED FUNCTIONS ####################################
    ############################################################
    
    def process_excel_to_table(self, file_content: bytes, filename: str, project_id: str) -> ProcessingResult:
        """Process Excel file with multiple sheets and store records using OpenAI mapping"""
        try:
            # Read Excel file
            excel_data = BytesIO(file_content)
            
            # Get all sheet names
            excel_file = pd.ExcelFile(excel_data)
            sheet_names = excel_file.sheet_names
            
            logging.info(f"Found {len(sheet_names)} sheets in {filename}: {sheet_names}")
            
            all_processed_records = []
            all_sections = []
            sheet_results = []
            total_records_processed = 0
            # project_id = filename.replace('.xlsx', '').replace('.xls', '')
            
            for sheet_index, sheet_name in enumerate(sheet_names):
                try:
                    # Process each sheet
                    sheet_result = self._process_single_sheet(
                        excel_file, sheet_name, filename, sheet_index, project_id
                    )
                    
                    sheet_results.append({
                        'sheet_name': sheet_name,
                        'success': sheet_result['success'],
                        'message': sheet_result['message'],
                        'records_processed': sheet_result['records_processed']
                    })
                    
                    if sheet_result['success']:
                        # Add section information if available
                        if sheet_result.get('section'):
                            all_sections.append(sheet_result['section'])
                            
                        # Add BOQ records if available
                        if sheet_result.get('records'):
                            all_processed_records.extend(sheet_result['records'])
                            total_records_processed += sheet_result['records_processed']
                        
                    
                except Exception as e:
                    logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
                    sheet_results.append({
                        'sheet_name': sheet_name,
                        'success': False,
                        'message': f"Error: {str(e)}",
                        'records_processed': 0
                    })
            
            # Store all valid records and sections in Azure Table (BATCH PROCESSING)
            storage_success = True
            
            # Store section information
            if all_sections:
                section_success = self._store_sections_with_retry(all_sections, filename)
                if not section_success:
                    storage_success = False
              
            # Store BOQ records
            if all_processed_records:
                boq_success = self._store_records_with_retry(all_processed_records, filename)
                if not boq_success:
                    storage_success = False
            
            
            if storage_success and (all_processed_records or all_sections):
                return ProcessingResult(
                    success=True,
                    message=f"Successfully processed {total_records_processed} BOQ records and {len(all_sections)} sections from {len(sheet_names)} sheets",
                    records_processed=total_records_processed,
                    data={
                        "records": all_processed_records, 
                        "sections": all_sections
                    },
                    sheet_results=sheet_results
                )
            elif not storage_success:
                return ProcessingResult(
                    success=False,
                    message="Failed to store some records/sections in Azure Table",
                    records_processed=total_records_processed,
                    sheet_results=sheet_results
                )
            else:
                return ProcessingResult(
                    success=False,
                    message="No valid BOQ data or sections found in any sheet",
                    records_processed=0,
                    sheet_results=sheet_results
                )
            
        except Exception as e:
            logging.error(f"Error processing Excel file: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Error processing Excel file: {str(e)}",
                records_processed=0
            )
    
    def _store_sections_with_retry(self, sections: List[Section], filename: str, max_retries: int = 3) -> bool:
        """Store section information with retry logic"""
        try:
            # Convert Section dataclass objects to dictionaries for storage
            section_records = []
            for section in sections:
                section_dict = {
                    # Remove PartitionKey and RowKey from here - let insert_section_records handle them
                    'section_id': section.section_id,
                    'section_name': section.section_name,
                    'project_id': section.project_id,
                    'cost': section.cost,
                    'discipline': section.discipline,
                    'source_file': filename
                }
                section_records.append(section_dict)
            
            # Try to store all sections at once
            success = self.table_service.insert_section_records(section_records)
            
            if success:
                logging.info(f"Successfully stored all {len(sections)} sections in batch")
                return True
            
            # If batch insert fails, try individual section insertion with retry
            logging.warning("Batch section insert failed, trying individual section insertion")
            successful_inserts = 0
            
            for i, section_record in enumerate(section_records):
                retry_count = 0
                record_inserted = False
                
                while retry_count < max_retries and not record_inserted:
                    try:
                        single_success = self.table_service.insert_section_records([section_record])
                        if single_success:
                            successful_inserts += 1
                            record_inserted = True
                            logging.debug(f"Successfully inserted section {i+1}/{len(section_records)}")
                        else:
                            retry_count += 1
                            logging.warning(f"Failed to insert section {i+1}, retry {retry_count}/{max_retries}")
                            
                            # Add exponential backoff
                            time.sleep(2 ** retry_count)
                    
                    except Exception as e:
                        retry_count += 1
                        logging.error(f"Error inserting section {i+1}, retry {retry_count}/{max_retries}: {str(e)}")
                        
                        # Add exponential backoff
                        if retry_count < max_retries:
                            time.sleep(2 ** retry_count)
                
                if not record_inserted:
                    logging.error(f"Failed to insert section {i+1} after {max_retries} retries: {section_record}")
            
            logging.info(f"Individual section insertion completed: {successful_inserts}/{len(section_records)} sections inserted")
            return successful_inserts > 0
            
        except Exception as e:
            logging.error(f"Error in _store_sections_with_retry: {str(e)}")
            return False
    
    def _store_records_with_retry(self, records: List[Dict[str, Any]], filename: str, max_retries: int = 3) -> bool:
        """Store records with retry logic and batch processing"""
        try:
            # First, try to store all records at once
            # success = self.table_service.insert_excel_records(records, filename)
            success = self.table_service.insert_boq_records(records, filename)
            
            if success:
                logging.info(f"Successfully stored all {len(records)} records in batch")
                return True
            
            # If batch insert fails, try individual record insertion with retry
            logging.warning("Batch insert failed, trying individual record insertion")
            successful_inserts = 0
            
            for i, record in enumerate(records):
                retry_count = 0
                record_inserted = False
                
                while retry_count < max_retries and not record_inserted:
                    try:
                        # Try to insert single record
                        single_record_success = self.table_service.insert_boq_records([record], filename)
                        if single_record_success:
                            successful_inserts += 1
                            record_inserted = True
                            logging.debug(f"Successfully inserted record {i+1}/{len(records)}")
                        else:
                            retry_count += 1
                            logging.warning(f"Failed to insert record {i+1}, retry {retry_count}/{max_retries}")
                    
                    except Exception as e:
                        retry_count += 1
                        logging.error(f"Error inserting record {i+1}, retry {retry_count}/{max_retries}: {str(e)}")
                
                if not record_inserted:
                    logging.error(f"Failed to insert record {i+1} after {max_retries} retries: {record}")
            
            logging.info(f"Individual insertion completed: {successful_inserts}/{len(records)} records inserted")
            return successful_inserts > 0
            
        except Exception as e:
            logging.error(f"Error in _store_records_with_retry: {str(e)}")
            return False
    
    def _process_single_sheet(self, excel_file: pd.ExcelFile, sheet_name: str, filename: str, 
                            sheet_index: int, project_id: str) -> Dict[str, Any]:
        """Process a single Excel sheet for both BOQ items and section information"""
        try:
            sheet_id = str(uuid.uuid4())
            # Read the sheet without assuming header location
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            if df_raw.empty:
                return {
                    'success': False,
                    'message': f"Sheet '{sheet_name}' is empty",
                    'records_processed': 0
                }
            
            # Convert first 20 rows to string representation for OpenAI analysis
            preview_rows = min(20, len(df_raw))
            sheet_preview = df_raw.head(preview_rows).to_string(index=True, na_rep='')
            
            logging.info(f"Sheet '{sheet_name}' preview (first {preview_rows} rows):\n{sheet_preview}")
            
            # Use OpenAI to analyze the entire sheet content and find both table structure and section info
            analysis_result = self.service_openai._analyze_sheet_with_openai(sheet_preview, sheet_name, sheet_index, project_id, sheet_id)
            
            if not analysis_result or not analysis_result.get('boq_table_found'):
                return {
                    'success': False,
                    'message': f"Could not analyze sheet '{sheet_name}'",
                    'records_processed': 0
                }
            
            table_analysis = analysis_result.get('table_analysis')
            section_info = analysis_result.get('section_info')
            
            result_data = {
                'success': True,
                'message': f"Successfully processed sheet '{sheet_name}'",
                'records_processed': 0,
                'records': [],
                'section': None
            }
            
            # Process BOQ table if found
            if table_analysis:
                header_row = table_analysis.get('header_row')
                column_mapping = table_analysis.get('column_mapping')
                
                if header_row is not None and column_mapping:
                    # Check if required fields are present
                    required_mapped = self._validate_required_fields(column_mapping)
                    if required_mapped:
                        
                        column_indices = self._get_column_indices(df_raw, header_row, column_mapping)
                    
                        if column_indices:
                            # Re-read the sheet with the correct header row
                            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
                            
                            # Filter out empty rows
                            df = df.dropna(how='all')
                            
                            if not df.empty:
                                # Process BOQ records using column indices
                                preprocessed_records = self._extract_boq_records_by_indices(
                                    df, column_indices, sheet_name, project_id, section_id=sheet_id
                                )
                                
                                processed_records = merge_boq_descriptions_advanced(preprocessed_records)
                                
                                if processed_records:
                                    result_data['success'] = True
                                    result_data['records'] = processed_records
                                    result_data['records_processed'] = len(processed_records)
                                    result_data['message'] = f"Successfully processed {len(processed_records)} BOQ records from sheet '{sheet_name}'"
                                    
                                    logging.info(f"Successfully extracted {len(processed_records)} BOQ records from sheet '{sheet_name}'")
                        
                
                # Process section information if found
                        if section_info:
                            try:
                                section = Section(
                                    section_id=section_info['section_id'],
                                    section_name=section_info['section_name'], 
                                    project_id=section_info['project_id'],
                                    cost=section_info.get('cost'),
                                    discipline=section_info.get('discipline')
                                )
                                result_data['section'] = section
                                result_data['message'] += f", Section: {section.section_name})"
                            except Exception as e:
                                logging.error(f"Error creating Section object: {str(e)}")
                                result_data['message'] += ", Section: error creating)"
                        else:
                            result_data['message'] += ")"
                    
                return result_data
                
        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
            return {
                'success': False,
                'message': f"Error processing sheet '{sheet_name}': {str(e)}",
                'records_processed': 0
            }

    def _get_column_indices(self, df_raw: pd.DataFrame, header_row: int, 
                       column_mapping: Dict[str, str]) -> Dict[str, int]:
        """Get column indices for all mapped columns"""
        try:
            # Get the header row data
            if header_row >= len(df_raw):
                logging.error(f"Header row {header_row} is beyond sheet length {len(df_raw)}")
                return {}
            
            header_data = df_raw.iloc[header_row]
            column_indices = {}
            
            logging.info(f"Header row {header_row} data: {header_data.tolist()}")
            
            # For each mapped column, find its index
            for actual_column_name, schema_field in column_mapping.items():
                found_index = None
                
                # Try exact match first
                for col_idx, header_value in enumerate(header_data):
                    if pd.isna(header_value):
                        continue
                        
                    header_str = str(header_value).strip()
                    if header_str == actual_column_name:
                        found_index = col_idx
                        break
                
                # If exact match not found, try fuzzy matching
                if found_index is None:
                    for col_idx, header_value in enumerate(header_data):
                        if pd.isna(header_value):
                            continue
                            
                        header_str = str(header_value).strip().lower()
                        actual_lower = actual_column_name.strip().lower()
                        
                        # Check if they're similar (removing spaces, punctuation)
                        header_clean = ''.join(c for c in header_str if c.isalnum())
                        actual_clean = ''.join(c for c in actual_lower if c.isalnum())
                        
                        if header_clean == actual_clean:
                            found_index = col_idx
                            break
                
                if found_index is not None:
                    column_indices[schema_field] = found_index
                    logging.info(f"Mapped '{actual_column_name}' -> '{schema_field}' at column index {found_index}")
                else:
                    logging.warning(f"Could not find column '{actual_column_name}' in header row")
            
            logging.info(f"Final column indices mapping: {column_indices}")
            return column_indices
            
        except Exception as e:
            logging.error(f"Error getting column indices: {str(e)}")
            return {}

    def _extract_boq_records_by_indices(self, df: pd.DataFrame, column_indices: Dict[str, int], 
                                    sheet_name: str, project_id: str, section_id: str) -> List[Dict[str, Any]]:
        """Extract BOQ records using column indices for better performance"""
        try:
            boq_records = []
            
            # Get required field indices
            quantity_idx = column_indices.get('Quantity')
            unit_rate_idx = column_indices.get('UnitRate')
            boq_description_idx = column_indices.get('Description')
            
            if quantity_idx is None or unit_rate_idx is None:
                logging.error("Required field indices not found")
                return []
            
            logging.info(f"Using column indices: {column_indices}")
            
            # Convert DataFrame to numpy array for faster access
            data_array = df.values
            
            for row_idx, row_data in enumerate(data_array):
                try:
                    # Check if we have enough columns
                    if len(row_data) <= max(quantity_idx, unit_rate_idx):
                        continue
                    
                    # Get required values using indices
                    description_value = row_data[boq_description_idx] if boq_description_idx < len(row_data) else None
                    
                    # Skip if required values are missing or invalid
                    # if pd.isna(quantity_value) or pd.isna(unit_rate_value):
                    #     continue
                    if pd.isna(description_value):
                        continue
                    
                    # Create BOQ record
                    boq_record = {
                        'ItemID': str(uuid.uuid4()),
                        'ProjectID': project_id,
                        'SectionID': section_id,  # Use sheet name as section ID
                        'SheetName': sheet_name,
                        'SourceRow': row_idx + 1,
                    }
                    
                    # Add ALL other mapped fields using indices
                    for schema_field, col_idx in column_indices.items():
                        if col_idx < len(row_data):
                            value = row_data[col_idx]
                            
                            if not pd.isna(value) and str(value).strip():
                                if schema_field == 'TotalCost':
                                    # Try to convert to float for cost fields
                                    try:
                                        boq_record[schema_field] = float(str(value).replace(',', ''))
                                    except (ValueError, TypeError):
                                        boq_record[schema_field] = str(value).strip()
                                else:
                                    boq_record[schema_field] = str(value).strip()
                    
                    boq_records.append(boq_record)
                    
                except Exception as row_error:
                    logging.error(f"Error processing row {row_idx + 1} in sheet '{sheet_name}': {str(row_error)}")
                    continue
            
            logging.info(f"Successfully extracted {len(boq_records)} valid BOQ records from sheet '{sheet_name}'")
            return boq_records
            
        except Exception as e:
            logging.error(f"Error extracting BOQ records from sheet '{sheet_name}': {str(e)}")
            return []


    def _validate_required_fields(self, column_mapping: Dict[str, str]) -> bool:
        """Validate that required fields are present in the mapping"""
        mapped_fields = set(column_mapping.values())
        required_fields = {'Quantity', 'UnitRate'}
        
        has_required = required_fields.issubset(mapped_fields)
        logging.info(f"Required fields check - Has quantity and unit_rate: {has_required}")
        return has_required
    
    def _map_and_clean_records(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                              sheet_name: str, filename: str) -> List[Dict[str, Any]]:
        """Map DataFrame columns to BOQ schema and clean the records"""
        # Get top N records
        # logging.warning("Mapping and cleaning records from DataFrame")
        # top_records = df.head(max_records)
        
        # Create mapped records
        mapped_records = []
        
        for index, row in df.iterrows():
            # Start with base record structure
            mapped_record = {
                'ItemID': str(uuid.uuid4()),  # Generate unique ID
                'ProjectID': filename.replace('.xlsx', '').replace('.xls', ''),  # Use filename as project ID
                'SheetName': sheet_name,
                'SourceRow': index + 1
            }
            
            # logging.warning("Mapped record structure initialized: %s", mapped_record)
            
            # Map the columns according to the OpenAI mapping
            for original_header, schema_field in column_mapping.items():
                # logging.warning(f"Processing header '{original_header}' mapped to schema field '{schema_field}'")
                if original_header in df.columns:
                    value = row[original_header]
                    
                    # Clean up the value
                    if pd.isna(value):
                        mapped_record[schema_field] = None

                    else:
                        # Handle text fields
                        mapped_record[schema_field] = str(value).strip() if value is not None else ""
                        
            mapped_records.append(mapped_record)
        
        logging.info(f"Mapped {len(mapped_records)} valid records from sheet '{sheet_name}'")
        return mapped_records