# processors/excel_to_table_processor.py
import json
import os
import time
import uuid
import pandas as pd
import logging
import uuid
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime
from services.azure_sql_service import AzureSQLService
from services.azure_table_service import AzureTableService
from models.processing_models import BOQSchema, ProcessingResult, Section
from services.openai_service import AzureOpenAIService
from utils.boq_utils import merge_boq_descriptions_advanced


class EnhancedExcelToTableProcessor:
    def __init__(self):
        self.table_service = AzureSQLService()
        # self.table_service = AzureTableService()
        self.openai_service = AzureOpenAIService()
        self.boq_schema = BOQSchema()
    
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
            project_info = self._extract_project_info_with_openai(sheet_preview, first_sheet_name, filename)
            
            if project_info:
                logging.info(f"Successfully extracted project information: {project_info}")
                return project_info
            else:
                logging.warning("Could not extract project information from first sheet")
                return None
                
        except Exception as e:
            logging.error(f"Error processing first sheet for project info: {str(e)}")
            return None

    def _extract_project_info_with_openai(self, sheet_preview: str, sheet_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Use OpenAI to extract project information from the sheet content
        
        Args:
            sheet_preview: String representation of sheet content
            sheet_name: Name of the sheet being processed
            filename: Original filename
        
        Returns:
            Dict containing extracted project information or None if extraction fails
        """
        try:
            system_message = """You are an expert in construction project document analysis and BOQ (Bill of Quantities) data extraction.
    Your task is to analyze Excel sheet content and extract project information including project name and description.

    Look for:
    1. Project names (often at the top of documents, may include words like "Project", "Construction", "Building", etc.)
    2. Project descriptions (detailed text describing what the project is about)
    3. Any other relevant project details like category, dates, or cost information

    Return only a valid JSON object with the extracted information."""

            user_message = f"""
    I have an Excel sheet named "{sheet_name}" from file "{filename}" with the following content (showing first 15 rows):

    {sheet_preview}

    Please analyze this content and extract project information. Focus on finding:

    1. PROJECT NAME (REQUIRED) - Look for:
    - Text that appears to be a project title or name
    - May contain words like "Project", "Construction", "Building", "Development"
    - Often appears in the first few rows
    - Could be in headers, titles, or standalone cells

    2. PROJECT DESCRIPTION (REQUIRED) - Look for:
    - Detailed text describing the project scope, location, or purpose
    - May be in a dedicated description field or scattered across multiple cells
    - Could include location details, project type, or scope information

    3. OPTIONAL FIELDS (if clearly visible):
    - Total cost or budget information
    - Project dates
    - Project category or type

    EXTRACTION RULES:
    1. MUST extract project_name and project_description - these are mandatory
    2. If no clear project name is found, try to construct one from available information
    3. If no clear description is found, combine relevant descriptive text
    4. Use the filename as fallback for project name if nothing else is found
    5. Be intelligent about combining related information from multiple cells
    6. Ignore purely tabular BOQ data (quantities, rates, amounts) - focus on project metadata

    Return ONLY a JSON object in this exact format:
    {{
    "project_name": "extracted or constructed project name",
    "project_description": "extracted or constructed project description", 
    "total_cost": null or numeric_value,
    "project_date": null or "YYYY-MM-DD",
    "project_category": null or "category_string"
    }}

    Example response:
    {{
    "project_name": "Residential Complex Phase 1 Construction",
    "project_description": "Construction of 50-unit residential complex with amenities including parking, landscaping, and community facilities in Downtown Area",
    "total_cost": 2500000.0,
    "project_date": null,
    "project_category": "Residential Construction"
    }}

    If no meaningful project information can be found, use the filename and make reasonable assumptions:
    {{
    "project_name": "{filename.replace('.xlsx', '').replace('.xls', '')}",
    "project_description": "Construction project as per BOQ specifications",
    "total_cost": null,
    "project_date": null, 
    "project_category": "Construction"
    }}
    """

            response = self.openai_service.simple_chat(
                user_message=user_message,
                system_message=system_message,
                temperature=0.2  # Low temperature for consistent extraction
            )
            
            # Parse the JSON response
            try:
                # Clean the response to extract JSON
                response_clean = response.strip()
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:-3]
                elif response_clean.startswith('```'):
                    response_clean = response_clean[3:-3]
                
                project_info = json.loads(response_clean)
                
                logging.info(f"OpenAI project extraction for '{sheet_name}': {project_info}")
                
                # Validate the response structure - ensure required fields exist
                if 'project_name' in project_info and 'project_description' in project_info:
                    # Ensure project_name and project_description are not empty
                    if not project_info['project_name'] or not project_info['project_description']:
                        logging.warning("Empty project_name or project_description, using fallbacks")
                        if not project_info['project_name']:
                            project_info['project_name'] = filename.replace('.xlsx', '').replace('.xls', '')
                        if not project_info['project_description']:
                            project_info['project_description'] = "Construction project as per BOQ specifications"
                    
                    return project_info
                else:
                    logging.error(f"Invalid response structure from OpenAI - missing required fields: {project_info}")
                    return None
                    
            except json.JSONDecodeError as je:
                logging.error(f"Failed to parse OpenAI response as JSON: {response}")
                logging.error(f"JSON Error: {str(je)}")
                return None
                
        except Exception as e:
            logging.error(f"Error extracting project info with OpenAI: {str(e)}")
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

    def process_and_store_project_info(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """
        Complete workflow: Extract project info from first sheet and store in Azure Table
        
        Args:
            file_content: Excel file content as bytes
            filename: Name of the Excel file
        
        Returns:
            Dict containing stored project information with project_id, or None if failed
        """
        try:
            project_id = str(uuid.uuid4())
            # Extract project information from first sheet
            project_info = self.process_first_sheet_for_project_info(file_content, filename)
            project_info['project_id'] = project_id
            
            if not project_info:
                logging.error("Could not extract project information from first sheet")
                return None
            
            # Store the project information
            project_result = self.store_project_with_retry(project_info, filename)
            
            excel_result = self.process_excel_to_table(
                file_content = file_content,
                filename = filename,
                project_id = project_id
            )
            
            if project_result and excel_result.success:
                # Return the complete project information including the assigned ID
                # stored_project_info = project_info.copy()
                # stored_project_info['project_id'] = project_id
                # stored_project_info['source_filename'] = filename
                
                # logging.info(f"Successfully processed and stored project: {stored_project_info['project_name']}")
                # return stored_project_info
                return project_result, excel_result
            else:
                logging.error("Failed to store project information")
                return None, None
                
        except Exception as e:
            logging.error(f"Error in process_and_store_project_info: {str(e)}")
            return None, None
    
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
            
            for sheet_index, sheet_name in enumerate(sheet_names[:3]):
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
                        # Add BOQ records if available
                        if sheet_result.get('records'):
                            all_processed_records.extend(sheet_result['records'])
                            total_records_processed += sheet_result['records_processed']
                        
                        # Add section information if available
                        if sheet_result.get('section'):
                            all_sections.append(sheet_result['section'])
                    
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
    
    # def _store_sections_with_retry(self, sections: List[Section], filename: str, max_retries: int = 3) -> bool:
    #     """Store section information with retry logic"""
    #     try:
    #         # Convert Section dataclass objects to dictionaries for storage
    #         section_records = []
    #         for section in sections:
    #             section_dict = {
    #                 'PartitionKey': f"{section.project_id}",
    #                 'RowKey': f"{section.section_id}",
    #                 'section_id': section.section_id,
    #                 'section_name': section.section_name,
    #                 'project_id': section.project_id,
    #                 'cost': section.cost,
    #                 'discipline': section.discipline,
    #                 'source_file': filename
    #             }
    #             section_records.append(section_dict)
            
    #         # Try to store all sections at once (assuming you have a method for this)
    #         # You might need to create a new method in AzureTableService for sections
    #         success = self.table_service.insert_section_records(section_records)
            
    #         if success:
    #             logging.info(f"Successfully stored all {len(sections)} sections in batch")
    #             return True
            
    #         # If batch insert fails, try individual section insertion with retry
    #         logging.warning("Batch section insert failed, trying individual section insertion")
    #         successful_inserts = 0
            
    #         for i, section_record in enumerate(section_records):
    #             retry_count = 0
    #             record_inserted = False
                
    #             while retry_count < max_retries and not record_inserted:
    #                 try:
    #                     single_success = self.table_service.insert_section_records([section_record])
    #                     if single_success:
    #                         successful_inserts += 1
    #                         record_inserted = True
    #                         logging.debug(f"Successfully inserted section {i+1}/{len(section_records)}")
    #                     else:
    #                         retry_count += 1
    #                         logging.warning(f"Failed to insert section {i+1}, retry {retry_count}/{max_retries}")
                    
    #                 except Exception as e:
    #                     retry_count += 1
    #                     logging.error(f"Error inserting section {i+1}, retry {retry_count}/{max_retries}: {str(e)}")
                
    #             if not record_inserted:
    #                 logging.error(f"Failed to insert section {i+1} after {max_retries} retries: {section_record}")
            
    #         logging.info(f"Individual section insertion completed: {successful_inserts}/{len(section_records)} sections inserted")
    #         return successful_inserts > 0
            
    #     except Exception as e:
    #         logging.error(f"Error in _store_sections_with_retry: {str(e)}")
    #         return False
    
    
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
            analysis_result = self._analyze_sheet_with_openai(sheet_preview, sheet_name, sheet_index, project_id, sheet_id)
            
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
                        # Re-read the sheet with the correct header row
                        # df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
                        
                        # # Clean column names
                        # df.columns = df.columns.astype(str).str.strip()
                        
                        # # Process BOQ records with the mapping
                        # processed_records = self._map_and_clean_records(df, column_mapping, sheet_name, filename)
                        
                        # result_data['records'] = processed_records
                        # result_data['records_processed'] = len(processed_records)
                        # result_data['message'] += f" (BOQ: {len(processed_records)} records"
                        
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
    
#     def _analyze_sheet_with_openai(self, sheet_preview: str, sheet_name: str, 
#           sheet_index, project_id: str) -> Optional[Dict[str, Any]]:
#         """Use OpenAI to analyze sheet for both BOQ table structure and section information"""
#         try:
#             # Prepare the schema information
#             required_fields_info = json.dumps(self.boq_schema.REQUIRED_FIELDS, indent=2)
#             optional_fields_info = json.dumps(self.boq_schema.OPTIONAL_FIELDS, indent=2)
            
#             system_message = """You are an expert in construction BOQ (Bill of Quantities) data analysis and Excel data extraction.
# Your task is to analyze Excel sheet content and identify:
# 1. BOQ table structure (header row and column mapping)
# 2. Section information (section name, costs, discipline)
# 3. Extract metadata about the sheet/section

# Return only a valid JSON object with the analysis results."""
            
#             user_message = f"""
# I have an Excel sheet named "{sheet_name}" (sheet index: {sheet_index}) from project "{project_id}" with the following content:

# {sheet_preview}

# Please analyze this sheet content and provide BOTH:

# 1. **BOQ TABLE ANALYSIS** (if a BOQ table exists):
#    - Identify the row number where data table headers are located
#    - Map headers to BOQ schema fields (quantity and unit_rate are mandatory)

# 2. **SECTION INFORMATION EXTRACTION**:
#    - Extract section name (often the sheet name or a title in the sheet)
#    - Identify any total costs or budget amounts
#    - Determine the discipline/trade (electrical, mechanical, civil, etc.)
#    - Generate a section_id (use sheet_index or derive from content)

# BOQ SCHEMA:
# REQUIRED FIELDS: {required_fields_info}
# OPTIONAL FIELDS: {optional_fields_info}

# SECTION SCHEMA:
# - section_id: int (required) - use sheet index or generate from content
# - section_name: str (required) - extract from sheet name or content
# - project_id: str (required) - use "{project_id}"
# - cost: float (optional) - any total/budget amount found
# - discipline: str (optional) - construction discipline/trade

# Return ONLY a JSON object in this format:
# {{
#   "boq_table_found":"true|false",
#   "table_analysis": {{
#     "header_row": <row_number_or_null>,
#     "column_mapping": {{}},
#     "confidence": "high|medium|low|none"
#   }},
#   "section_info": {{
#     "section_id": {sheet_index},
#     "section_name": "<extracted_section_name>",
#     "project_id": "{project_id}",
#     "cost": <total_cost_or_null>,
#     "discipline": "<discipline_or_null>"
#   }}
# }}

# IMPORTANT:
# - If no BOQ table is found, set boq_table_found as false. table_analysis values and section_info values to null/empty
# - Use sheet name as fallback for section_name
# - Look for keywords like "total", "sum", "budget" for cost extraction
# - Look for discipline keywords like "electrical", "mechanical", "civil", "plumbing", etc.
# """
            
#             response = self.openai_service.simple_chat(
#                 user_message=user_message,
#                 system_message=system_message,
#                 temperature=0.1
#             )
            
#             # Parse the JSON response
#             try:
#                 response_clean = response.strip()
#                 if response_clean.startswith('```json'):
#                     response_clean = response_clean[7:-3]
#                 elif response_clean.startswith('```'):
#                     response_clean = response_clean[3:-3]
                
#                 analysis_result = json.loads(response_clean)
                
#                 logging.info(f"OpenAI analysis for '{sheet_name}': {analysis_result}")
                
#                 # Validate and clean the response
#                 # if 'section_info' not in analysis_result:
#                 #     # Create default section info if missing
#                 #     analysis_result['section_info'] = {
#                 #         'section_id': sheet_index,
#                 #         'section_name': sheet_name,
#                 #         'project_id': project_id,
#                 #         'cost': None,
#                 #         'discipline': None
#                 #     }
                
#                 # Ensure section_info has required fields
#                 # section_info = analysis_result['section_info']
#                 # if 'section_id' not in section_info:
#                 #     section_info['section_id'] = sheet_index
#                 # if 'section_name' not in section_info or not section_info['section_name']:
#                 #     section_info['section_name'] = sheet_name
#                 # if 'project_id' not in section_info:
#                 #     section_info['project_id'] = project_id
                
#                 return analysis_result
                
#             except json.JSONDecodeError as je:
#                 logging.error(f"Failed to parse OpenAI response as JSON: {response}")
#                 logging.error(f"JSON Error: {str(je)}")
#                 # Return default structure with section info
#                 return {
#                     'table_analysis': None,
#                     'section_info': {
#                         'section_id': sheet_index,
#                         'section_name': sheet_name,
#                         'project_id': project_id,
#                         'cost': None,
#                         'discipline': None
#                     }
#                 }
                
#         except Exception as e:
#             logging.error(f"Error analyzing sheet with OpenAI: {str(e)}")
#             return {
#                 'table_analysis': None,
#                 'section_info': {
#                     'section_id': sheet_index,
#                     'section_name': sheet_name,
#                     'project_id': project_id,
#                     'cost': None,
#                     'discipline': None
#                 }
#             }
    
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
                    quantity_value = row_data[quantity_idx] if quantity_idx < len(row_data) else None
                    unit_rate_value = row_data[unit_rate_idx] if unit_rate_idx < len(row_data) else None
                    description_value = row_data[boq_description_idx] if boq_description_idx < len(row_data) else None
                    
                    # Skip if required values are missing or invalid
                    # if pd.isna(quantity_value) or pd.isna(unit_rate_value):
                    #     continue
                    if pd.isna(description_value):
                        continue
                    
                    # Try to convert to numeric values
                    # try:
                    #     quantity = float(str(quantity_value).replace(',', ''))
                    #     unit_rate = float(str(unit_rate_value).replace(',', ''))
                    # except (ValueError, TypeError):
                    #     logging.warning(f"Invalid numeric values in row {row_idx + 1}: quantity={quantity_value}, unit_rate={unit_rate_value}")
                    #     continue
                    
                    # Skip zero or negative values
                    # if quantity <= 0 or unit_rate <= 0:
                    #     continue
                    
                    # Create BOQ record
                    boq_record = {
                        'ItemID': str(uuid.uuid4()),
                        'ProjectID': project_id,
                        'SectionID': section_id,  # Use sheet name as section ID
                        'SheetName': sheet_name,
                        'SourceRow': row_idx + 1,
                        'Quantity': quantity_value,
                        'UnitRate': unit_rate_value,
                        # 'TotalCost': quantity_value  # Calculate total
                    }
                    
                    # Add ALL other mapped fields using indices
                    for schema_field, col_idx in column_indices.items():
                        if schema_field not in ['Quantity', 'UnitRate'] and col_idx < len(row_data):
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

    
    def _analyze_sheet_with_openai(self, sheet_preview: str, sheet_name: str, 
                              sheet_index: int, project_id: str, sheet_id: str ) -> Optional[Dict[str, Any]]:
            """Use OpenAI to analyze sheet for both BOQ table structure and section information"""
            try:
                # Prepare the schema information
                required_fields_info = json.dumps(self.boq_schema.REQUIRED_FIELDS, indent=2)
                optional_fields_info = json.dumps(self.boq_schema.OPTIONAL_FIELDS, indent=2)
                
                system_message = """You are an expert in construction BOQ (Bill of Quantities) data analysis and Excel data extraction.
        Your task is to analyze Excel sheet content and identify:
        1. BOQ table structure (header row and column mapping)
        2. Section information (section name, costs, discipline)
        3. Extract metadata about the sheet/section

        Return only a valid JSON object with the analysis results."""
                
                user_message = f"""
        I have an Excel sheet named "{sheet_name}" (sheet index: {sheet_index}) from project "{project_id}" with the following content:

        {sheet_preview}

        Please analyze this sheet content and provide BOTH:

        1. **BOQ TABLE ANALYSIS** (if a BOQ table exists):
        - Identify the exact row number (0-based index) where the table headers are located
        - Map the column headers to BOQ schema fields
        - Focus on finding Quantity and UnitRate columns (these are mandatory)

        2. **SECTION INFORMATION EXTRACTION**:
        - Extract section name (often the sheet name or a title in the sheet)
        - Identify any total costs or budget amounts
        - Determine the discipline/trade (electrical, mechanical, civil, etc.)
        - Use sheet_index as section_id

        BOQ SCHEMA:
        REQUIRED FIELDS: {required_fields_info}
        OPTIONAL FIELDS: {optional_fields_info}

        SECTION SCHEMA:
        - section_id: str (required) - use {sheet_id}
        - section_name: str (required) - extract from sheet name or content
        - project_id: str (required) - use "{project_id}"
        - cost: float (optional) - any total/budget amount found
        - discipline: str (optional) - construction discipline/trade

        Return ONLY a JSON object in this exact format:
        {{
        "boq_table_found": true|false,
        "table_analysis": {{
            "header_row": <row_number_or_null>,
            "column_mapping": {{
            "<actual_column_name>": "<schema_field_name>",
            ...
            }},
            "confidence": "high|medium|low"
        }},
        "section_info": {{
            "section_id": {sheet_id},
            "section_name": "<extracted_section_name>",
            "project_id": "{project_id}",
            "cost": <total_cost_or_null>,
            "discipline": "<discipline_or_null>"
        }}
        }}

        CRITICAL REQUIREMENTS:
        - header_row must be the exact 0-based row index where column headers are located
        - column_mapping keys must be the EXACT column names from the Excel sheet
        - column_mapping values must be from the BOQ schema field names
        - If no BOQ table is found, set boq_table_found to false and table_analysis to null
        - Always provide section_info using sheet name as fallback for section_name
        - Look for cost-related keywords: "total", "sum", "budget", "amount", "cost"
        - Look for discipline keywords: "electrical", "mechanical", "civil", "plumbing", "hvac", etc.
        """
                
                response = self.openai_service.simple_chat(
                    user_message=user_message,
                    system_message=system_message,
                    temperature=0.1
                )
                
                # Parse the JSON response
                try:
                    response_clean = response.strip()
                    if response_clean.startswith('```json'):
                        response_clean = response_clean[7:-3]
                    elif response_clean.startswith('```'):
                        response_clean = response_clean[3:-3]
                    
                    analysis_result = json.loads(response_clean)
                    
                    # Validate the response structure
                    if 'boq_table_found' not in analysis_result:
                        analysis_result['boq_table_found'] = False
                    
                    if 'section_info' not in analysis_result:
                        analysis_result['section_info'] = {
                            'section_id': sheet_id,
                            'section_name': sheet_name,
                            'project_id': project_id,
                            'cost': None,
                            'discipline': None
                        }
                    
                    # Ensure section_info has required fields
                    section_info = analysis_result['section_info']
                    section_info.setdefault('section_id', sheet_id)
                    section_info.setdefault('section_name', sheet_name)
                    section_info.setdefault('project_id', project_id)
                    
                    logging.info(f"OpenAI analysis for '{sheet_name}': {analysis_result}")
                    return analysis_result
                    
                except json.JSONDecodeError as je:
                    logging.error(f"Failed to parse OpenAI response as JSON: {response}")
                    logging.error(f"JSON Error: {str(je)}")
                    # Return default structure with section info
                    return {
                        'boq_table_found': False,
                        'table_analysis': None,
                        'section_info': {
                            'section_id': sheet_id,
                            'section_name': sheet_name,
                            'project_id': project_id,
                            'cost': None,
                            'discipline': None
                        }
                    }
                    
            except Exception as e:
                logging.error(f"Error analyzing sheet with OpenAI: {str(e)}")
                return {
                    'boq_table_found': False,
                    'table_analysis': None,
                    'section_info': {
                        'section_id': sheet_id,
                        'section_name': sheet_name,
                        'project_id': project_id,
                        'cost': None,
                        'discipline': None
                    }
                }
    
    
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



    # def link_boq_to_project(self, boq_records: List[Dict[str, Any]], project_id: str) -> List[Dict[str, Any]]:
    #     """
    #     Link BOQ records to a project by adding project_id to each record
        
    #     Args:
    #         boq_records: List of BOQ record dictionaries
    #         project_id: The project ID to link to
        
    #     Returns:
    #         Updated BOQ records with project_id added
    #     """
    #     try:
    #         updated_records = []
    #         for record in boq_records:
    #             updated_record = record.copy()
    #             updated_record['LinkedProjectID'] = project_id
    #             updated_records.append(updated_record)
            
    #         logging.info(f"Linked {len(updated_records)} BOQ records to project {project_id}")
    #         return updated_records
            
    #     except Exception as e:
    #         logging.error(f"Error linking BOQ records to project: {str(e)}")
    #         return boq_records  # Return original records if linking fails

    # Updated main processing function example
    # def process_excel_with_complete_workflow(self, file_content: bytes, filename: str, max_records: int = 30) -> ProcessingResult:
    #     """
    #     Complete workflow: Process project info and BOQ data, linking them together
        
    #     Args:
    #         file_content: Excel file content as bytes
    #         filename: Name of the Excel file
    #         max_records: Maximum BOQ records to process per sheet
        
    #     Returns:
    #         ProcessingResult with both project and BOQ information
    #     """
    #     try:
    #         # Step 1: Extract and store project information
    #         stored_project = self.process_and_store_project_info(file_content, filename)
            
    #         if not stored_project:
    #             logging.warning("Could not extract/store project info, continuing with BOQ processing only")
    #             project_id = None
    #         else:
    #             project_id = stored_project['project_id']
    #             logging.info(f"Project stored successfully with ID: {project_id}")
            
    #         # Step 2: Process BOQ data
    #         boq_result = self.process_excel_to_table(file_content, filename, max_records)
            
    #         # Step 3: Link BOQ records to project if both are successful
    #         if project_id and boq_result.success and boq_result.data and boq_result.data.get('records'):
    #             linked_records = self.link_boq_to_project(boq_result.data['records'], project_id)
                
    #             # Update the stored BOQ records with project linkage
    #             # You might want to update the records in the table here
    #             boq_result.data['records'] = linked_records
            
    #         # Enhance the result with project information
    #         if stored_project:
    #             if not boq_result.data:
    #                 boq_result.data = {}
    #             boq_result.data['project_info'] = stored_project
            
    #         # Update success message
    #         if boq_result.success and stored_project:
    #             boq_result.message = f"Successfully processed project '{stored_project['project_name']}' and {boq_result.records_processed} BOQ records"
    #         elif stored_project:
    #             boq_result.message = f"Successfully processed project '{stored_project['project_name']}' but BOQ processing failed"
            
    #         return boq_result
            
    #     except Exception as e:
    #         logging.error(f"Error in complete workflow: {str(e)}")
    #         return ProcessingResult(
    #             success=False,
    #             message=f"Error in complete workflow: {str(e)}",
    #             records_processed=0
    #         )
    
    
        # def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Retrieve a project record by ID
        
    #     Args:
    #         project_id: The project ID to retrieve
        
    #     Returns:
    #         Dict containing project data or None if not found
    #     """
    #     try:
    #         # Query for the project across all partitions
    #         filter_query = f"ProjectID eq '{project_id}'"
    #         entities = list(self.table_client.query_entities(query_filter=filter_query))
            
    #         if entities:
    #             entity = entities[0]  # Should be only one with unique project ID
    #             return {
    #                 'project_id': entity.get('ProjectID'),
    #                 'project_name': entity.get('ProjectName'),
    #                 'project_description': entity.get('ProjectDescription'),
    #                 'total_cost': entity.get('TotalCost'),
    #                 'project_date': entity.get('ProjectDate'),
    #                 'project_category': entity.get('ProjectCategory'),
    #                 'source_filename': entity.get('SourceFilename'),
    #                 'created_date': entity.get('CreatedDate'),
    #                 'last_modified': entity.get('LastModified'),
    #                 'extraction_confidence': entity.get('ExtractionConfidence'),
    #                 'extraction_notes': entity.get('ExtractionNotes')
    #             }
    #         return None
            
    #     except Exception as e:
    #         logging.error(f"Error retrieving project by ID {project_id}: {str(e)}")
    #         return None

    # def update_project_record(self, project_id: str, updated_data: Dict[str, Any]) -> bool:
    #     """
    #     Update an existing project record
        
    #     Args:
    #         project_id: The project ID to update
    #         updated_data: Dictionary containing updated project information
        
    #     Returns:
    #         bool: True if successful, False otherwise
    #     """
    #     try:
    #         from datetime import datetime
            
    #         # First, get the existing entity to preserve PartitionKey and RowKey
    #         existing_project = self.get_project_by_id(project_id)
    #         if not existing_project:
    #             logging.error(f"Project with ID {project_id} not found for update")
    #             return False
            
    #         # Query to get the full entity with metadata
    #         filter_query = f"ProjectID eq '{project_id}'"
    #         entities = list(self.table_client.query_entities(query_filter=filter_query))
            
    #         if not entities:
    #             return False
            
    #         entity = entities[0]
            
    #         # Update the entity with new data
    #         if 'project_name' in updated_data:
    #             entity['ProjectName'] = updated_data['project_name']
    #         if 'project_description' in updated_data:
    #             entity['ProjectDescription'] = updated_data['project_description']
    #         if 'total_cost' in updated_data:
    #             entity['TotalCost'] = updated_data['total_cost']
    #         if 'project_category' in updated_data:
    #             entity['ProjectCategory'] = updated_data['project_category']
    #         if 'project_date' in updated_data:
    #             if isinstance(updated_data['project_date'], datetime):
    #                 entity['ProjectDate'] = updated_data['project_date']
    #             else:
    #                 try:
    #                     entity['ProjectDate'] = datetime.strptime(str(updated_data['project_date']), '%Y-%m-%d')
    #                 except (ValueError, TypeError):
    #                     logging.warning(f"Could not parse updated project date: {updated_data.get('project_date')}")
            
    #         # Update metadata
    #         entity['LastModified'] = datetime.utcnow()
            
    #         # Update in table
    #         self.table_client.update_entity(entity=entity, mode='replace')
    #         logging.info(f"Successfully updated project record with ID: {project_id}")
    #         return True
            
    #     except Exception as e:
    #         logging.error(f"Error updating project record {project_id}: {str(e)}")
    #         return False
