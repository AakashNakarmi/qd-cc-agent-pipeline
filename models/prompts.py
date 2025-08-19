system_message = """You are an expert in construction BOQ (Bill of Quantities) data analysis and Excel data extraction.
Your task is to analyze Excel sheet content and identify:
1. Where the actual data table begins (header row number)
2. Map column headers to a standardized BOQ schema
3. Ensure quantity and unit_rate columns are identified (mandatory for BOQ processing)

Return only a valid JSON object with the analysis results."""
            
user_message = """
I have an Excel sheet named "{sheet_name}" with the following content (showing first 20 rows):

{sheet_preview}

Please analyze this sheet content and:

1. IDENTIFY the row number where the actual data table headers are located (this might not be row 0)
2. MAP the identified headers to the BOQ schema fields below

BOQ SCHEMA:

REQUIRED FIELDS (must be present):
{required_fields_info}

OPTIONAL FIELDS:
{optional_fields_info}

ANALYSIS RULES:
1. Look for a row that contains column headers for a BOQ/costing table
2. MUST identify 'quantity' and 'unit_rate' fields - these are mandatory
3. The header row typically contains terms like: item, description, quantity, rate, amount, etc.
4. Ignore rows with company names, titles, or other non-tabular content
5. Map each identified header to the most appropriate schema field
6. Use exact schema field names as values (e.g., 'quantity', 'unit_rate', 'description')

Return ONLY a JSON object in this exact format:
{{
  "header_row": <row_number>,
  "column_mapping": {{
    "original_header_1": "schema_field_1",
    "original_header_2": "schema_field_2"
  }},
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of why this row was identified as headers"
}}

Example response:
{{
  "header_row": 3,
  "column_mapping": {{
    "Item No": "item_number",
    "Description": "description", 
    "Qty": "quantity",
    "Rate": "unit_rate",
    "Amount": "total_cost"
  }},
  "confidence": "high",
  "reasoning": "Row 3 contains clear BOQ column headers with quantity and rate fields"
}}
"""

