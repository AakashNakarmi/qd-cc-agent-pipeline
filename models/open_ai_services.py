
from typing import Dict, Any, Optional
import json
import logging

from models.processing_models import BOQSchema
from services.openai_service import AzureOpenAIService


class OpenAIService():
    """Custom exception for OpenAI service errors"""
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        self.boq_schema = BOQSchema()

    def _extract_project_info_with_openai(
        self, sheet_preview: str, sheet_name: str, filename: str
    ) -> Optional[Dict[str, Any]]:

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
"total_cost": 0,
"project_category": null or "category_string"
}}

Example response:
{{
"project_name": "Residential Complex Phase 1 Construction",
"project_description": "Construction of 50-unit residential complex with amenities including parking, landscaping, and community facilities in Downtown Area",
"total_cost": 0,
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
        - Extract section name (often the sheet name or a title in the sheet). For section_name, check for bill numbers. For example, BILL NO.2 - ROAD E19 WORKS, then section_name is ROAD E19 WORKS.
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
            "section_name": "<extracted_section_name>_{sheet_name}",
            "project_id": "{project_id}",
            "cost": <null>,
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
    