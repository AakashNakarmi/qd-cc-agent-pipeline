import os
from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ResourceExistsError, AzureError
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
from azure.data.tables import TableEntity

import re

class AzureTableService:
    def __init__(self):
        self.connection_string = os.environ["saqddev01_STORAGE"]
        self.table_service_client = TableServiceClient.from_connection_string(self.connection_string)
        self.project_table_name = os.environ.get('PROJECT_TABLE_NAME', 'Projects')
        self.section_table_name = os.environ.get('SECTION_TABLE_NAME', 'Sections')
        self.boq_table_name = os.environ.get('BOQ_TABLE_NAME', 'BOQItems')
        self.project_table_client = self._get_or_create_table(self.project_table_name)
        self.section_table_client = self._get_or_create_table(self.section_table_name)
        self.boq_table_client = self._get_or_create_table(self.boq_table_name)
        
        # Reserved property names that cannot be used
        self.reserved_properties = {
            'PartitionKey', 'RowKey', 'Timestamp', 'ETag', 
            'partitionkey', 'rowkey', 'timestamp', 'etag'
        }
    
    def _get_or_create_table(self, table_name) -> TableClient:
        """Create table if it doesn't exist and return table client"""
        try:
            # Try to create the table
            self.table_service_client.create_table_if_not_exists(table_name=table_name)
            logging.info(f"Table '{table_name}' is ready")
        except AzureError as e:
            logging.error(f"Error creating/accessing table: {str(e)}")
            raise
        
        return self.table_service_client.get_table_client(table_name=table_name)

    def insert_project_record(self, project_data: Dict[str, Any], filename: str) -> bool:
        """
        Insert a project record into Azure Table Storage
        
        Args:
            project_data: Dictionary containing project information
            filename: Source filename for tracking
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:            
            # Create a unique project ID if not provided
            project_id = str(uuid.uuid4())
            # project_id = project_data.get('project_id') or str(uuid.uuid4())
            
            # Prepare the entity for Azure Table
            # PartitionKey: Use a consistent partition strategy (e.g., by year or category)
            # RowKey: Use unique project ID
            partition_key = project_data.get('project_category', 'General')
            if not partition_key:
                partition_key = 'General'
            
            # Create table entity
            entity = TableEntity()
            entity['PartitionKey'] = partition_key
            entity['RowKey'] = project_id
            entity['ProjectID'] = project_id
            entity['ProjectName'] = project_data.get('project_name', '')
            entity['ProjectDescription'] = project_data.get('project_description', '')
            entity['TotalCost'] = project_data.get('total_cost')
            entity['ProjectCategory'] = project_data.get('project_category')
            entity['SourceFilename'] = filename
            entity['CreatedDate'] = datetime.utcnow()
            entity['LastModified'] = datetime.utcnow()
            
            # Handle project_date if provided
            if project_data.get('project_date'):
                if isinstance(project_data['project_date'], datetime):
                    entity['ProjectDate'] = project_data['project_date']
                else:
                    # Try to parse string date
                    try:
                        entity['ProjectDate'] = datetime.strptime(str(project_data['project_date']), '%Y-%m-%d')
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse project date: {project_data.get('project_date')}")
                        entity['ProjectDate'] = None
            
            # Add extraction metadata
            # entity['ExtractionConfidence'] = project_data.get('confidence', 'unknown')
            # entity['ExtractionNotes'] = project_data.get('extraction_notes', '')
            
            # Insert into table
            self.project_table_client.create_entity(entity=entity)
            logging.info(f"Successfully inserted project record with ID: {project_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error inserting project record: {str(e)}")
            return False
    
    def insert_section_records(self, section_records: List[Dict[str, Any]]) -> bool:
        """
        Insert section records into Azure Table Storage
        
        Args:
            section_records: List of dictionaries containing section information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not section_records:
                logging.warning("No section records to insert")
                return True
            
            successful_inserts = 0
            
            for record in section_records:
                try:
                    # Create table entity
                    entity = TableEntity()
                    
                    # Set partition and row keys
                    project_id = record.get('project_id', 'Unknown')
                    section_id = record.get('section_id', 0)
                    
                    entity['PartitionKey'] = f"123"
                    entity['RowKey'] = f"Section_{section_id}"
                    
                    # Set section data
                    entity['SectionID'] = section_id
                    entity['SectionName'] = record.get('section_name', '')
                    entity['ProjectID'] = project_id
                    entity['Cost'] = record.get('cost')
                    entity['Discipline'] = record.get('discipline', '')
                    entity['SourceFile'] = record.get('source_file', '')
                    entity['CreatedDate'] = datetime.utcnow()
                    entity['LastModified'] = datetime.utcnow()
                    
                    # Insert into table
                    self.section_table_client.create_entity(entity=entity)
                    successful_inserts += 1
                    logging.debug(f"Successfully inserted section: {record.get('section_name', 'Unknown')}")
                    
                except Exception as e:
                    logging.error(f"Error inserting section record: {str(e)}")
                    logging.error(f"Section record data: {record}")
                    continue
            
            logging.info(f"Successfully inserted {successful_inserts}/{len(section_records)} section records")
            return successful_inserts > 0
            
        except Exception as e:
            logging.error(f"Error in insert_section_records: {str(e)}")
            return False
 
    def insert_boq_records(self, records: List[Dict[str, Any]], project_id: str) -> bool:
        """Insert BOQ records into Azure Table with specific schema"""
        try:
            entities = []
            timestamp = datetime.utcnow().isoformat()
            
            for record in records:
                # Validate required fields
                if not record.get('ItemID') or not record.get('ProjectID'):
                    logging.warning(f"Skipping record with missing ItemID or ProjectID: {record}")
                    continue
                
                # Create entity with the exact BOQ schema
                entity = {
                    # Azure Table required fields
                    "PartitionKey": self._sanitize_key(record['ProjectID']),  # Use ProjectID as partition
                    "RowKey": record['ItemID'],  # Use ItemID as unique row key
                    
                    # BOQ specific fields - maintaining exact schema
                    "ItemID": record['ItemID'],
                    "ProjectID": record['ProjectID'],
                    "SheetName": record.get('SheetName', ''),
                    "SourceRow": record.get('SourceRow', 0),
                    "remarks": record.get('remarks') or '',  # Convert None to empty string
                    "item_number": record.get('item_number') or '',
                    "description": record.get('description') or '',
                    "quantity": record.get('quantity'),  # Keep None as is for numeric fields
                    "unit": record.get('unit') or '',
                    "unit_rate": record.get('unit_rate'),  # Keep None as is for numeric fields  
                    "total_cost": record.get('total_cost'),  # Keep None as is for numeric fields
                    
                    # Metadata fields
                    "ProcessedAt": timestamp
                }
                
                # Handle None values for numeric fields properly
                for field in ['quantity', 'unit_rate', 'total_cost', 'SourceRow']:
                    if entity[field] is None and field != 'SourceRow':
                        entity[field] = None  # Azure Tables can handle None for these
                    elif field == 'SourceRow' and entity[field] is None:
                        entity[field] = 0  # Default SourceRow to 0 if None
                
                # Truncate long string fields if necessary
                string_fields = ['remarks', 'item_number', 'description', 'unit', 'SheetName']
                for field in string_fields:
                    if entity[field] and len(str(entity[field])) > 32000:
                        entity[field] = str(entity[field])[:32000] + "...[truncated]"
                
                entities.append(entity)
            
            # Insert entities one by one with error handling
            successful_inserts = 0
            failed_inserts = 0
            
            for entity in entities:
                try:
                    self.boq_table_client.create_entity(entity=entity)
                    successful_inserts += 1
                    logging.info(f"Inserted BOQ record: ItemID={entity['ItemID']}, SourceRow={entity['SourceRow']}")
                except ResourceExistsError:
                    # Handle duplicate ItemID - try to update instead
                    try:
                        self.boq_table_client.upsert_entity(entity=entity)
                        successful_inserts += 1
                        logging.info(f"Updated existing BOQ record: ItemID={entity['ItemID']}")
                    except AzureError as e:
                        failed_inserts += 1
                        logging.error(f"Error upserting BOQ record {entity['ItemID']}: {str(e)}")
                except AzureError as e:
                    failed_inserts += 1
                    logging.error(f"Error inserting BOQ record {entity['ItemID']}: {str(e)}")
                    logging.error(f"Entity data: {entity}")
            
            logging.info(f"BOQ Records processing complete: {successful_inserts} successful, {failed_inserts} failed")
            return failed_inserts == 0
                
        except Exception as e:
            logging.error(f"Error processing BOQ records: {str(e)}")
            return False
    
    def _sanitize_key(self, key: str) -> str:
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
