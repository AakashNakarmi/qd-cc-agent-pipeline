import math
from typing import Any

import pandas as pd


def merge_boq_descriptions_advanced(processed_records):
    """
    Advanced version that handles continuous merging for multiple valid records
    after a group of invalid ones.
    
    Args:
        processed_records (list): List of dictionaries containing BOQ records
        
    Returns:
        list: Processed records with merged descriptions
    """
    if not processed_records:
        return processed_records
    
    def is_valid_item_number(item_number):
        """Check if ItemNumber starts with an alphabet (a-z or A-Z)"""
        if not item_number or item_number == '' or str(item_number).strip() == '':
            return False
        return str(item_number)[0].isalpha()
    
    def clean_description(description):
        """Clean and normalize description text"""
        if not description:
            return ""
        return str(description).strip()
    
    result = []
    accumulated_description = ""
    
    i = 0
    while i < len(processed_records):
        record = processed_records[i]
        item_number = record.get('ItemNumber', '')
        description = clean_description(record.get('Description', ''))
        
        if is_valid_item_number(item_number):
            # This is a valid record
            if accumulated_description:
                # We have accumulated description to merge
                merged_description = f"{accumulated_description} {description}".strip()
                record_copy = record.copy()
                record_copy['Description'] = merged_description
                result.append(record_copy)
                
                # Look ahead to see if there are more consecutive valid records
                # that should also get the accumulated description
                j = i + 1
                while j < len(processed_records):
                    next_record = processed_records[j]
                    next_item_number = next_record.get('ItemNumber', '')
                    next_description = clean_description(next_record.get('Description', ''))
                    
                    if is_valid_item_number(next_item_number):
                        # Another valid record - merge accumulated description
                        next_merged_description = f"{accumulated_description} {next_description}".strip()
                        next_record_copy = next_record.copy()
                        next_record_copy['Description'] = next_merged_description
                        result.append(next_record_copy)
                        j += 1
                    else:
                        # Found invalid record, break the chain
                        break
                
                # Reset accumulated description and update index
                accumulated_description = ""
                i = j - 1  # j-1 because the loop will increment i
            else:
                # No accumulated description, just add the record
                result.append(record)
        else:
            # Invalid ItemNumber - accumulate description
            if description:
                if accumulated_description:
                    accumulated_description = f"{accumulated_description} {description}"
                else:
                    accumulated_description = description
        
        i += 1
    
    return result


def sanitize_key(key: str) -> str:
    """Sanitize string to be used as PartitionKey or RowKey"""
    # Remove invalid characters for Azure Table keys
    invalid_chars = ['/', '\\', '#', '?', '\t', '\n', '\r']
    sanitized = key
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Ensure it's not empty and not too long (max 1KB)
    if not sanitized:
        sanitized = "default"
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
        
    return sanitized
    
def is_valid_total_cost(value: Any) -> bool:
    """
    Check if the total_cost value is valid (numeric and not NaN)
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    if value is None:
        return False
    
    # Check if it's a string
    if isinstance(value, str):
        # Try to convert string to number, but reject if it contains non-numeric characters
        try:
            numeric_value = float(value.strip())
            return not math.isnan(numeric_value) and math.isfinite(numeric_value)
        except (ValueError, TypeError):
            return False
    
    # Check if it's a numeric type
    if isinstance(value, (int, float)):
        return not math.isnan(float(value)) and math.isfinite(float(value))
    
    # Check for pandas NaN
    try:
        if pd.isna(value):
            return False
    except:
        pass
    
    return False