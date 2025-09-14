import os
import pymssql
import logging
from typing import List, Dict, Any
from datetime import datetime
import uuid

from utils.boq_utils import is_valid_total_cost

class AzureSQLService:
    def __init__(self):
        # Database connection parameters from environment variables
        self.server = os.environ["AZURE_SQL_SERVER"]  # e.g., "your-server.database.windows.net"
        self.database = os.environ["AZURE_SQL_DATABASE"]  # e.g., "your-database"
        self.username = os.environ["AZURE_SQL_USERNAME"]
        self.password = os.environ["AZURE_SQL_PASSWORD"]
        
        # Initialize database tables
        self._create_tables_if_not_exist()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            return pymssql.connect(
                server=self.server,
                user=self.username,
                password=self.password,
                database=self.database,
                timeout=30,
                login_timeout=30,
                as_dict=False
            )
            # return pyodbc.connect(self.connection_string)
        except Exception as e:
            logging.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _create_tables_if_not_exist(self):
        """Create tables if they don't exist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create Projects table
            projects_table = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='projects' AND xtype='U')
            CREATE TABLE projects (
                ProjectID NVARCHAR(50) PRIMARY KEY,
                ProjectName NVARCHAR(255),
                TotalCost DECIMAL(18,2),
                ProjectDescription NTEXT,
                ProjectDate NVARCHAR(50),
                ProjectCategory NVARCHAR(100),
                SourceFilename NVARCHAR(255),
                CreatedDate DATETIME2,
                LastModified DATETIME2
            )
            """
            
            # Create Sections table
            sections_table = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sections' AND xtype='U')
            CREATE TABLE sections (
                SectionID NVARCHAR(50) PRIMARY KEY,
                ProjectID NVARCHAR(50) NOT NULL,
                SectionName NVARCHAR(255),
                Cost DECIMAL(18,2),
                Discipline NVARCHAR(100),
                SourceFile NVARCHAR(255),
                CreatedDate DATETIME2,
                LastModified DATETIME2,
                FOREIGN KEY (ProjectID) REFERENCES projects(ProjectID)
            )
            """
            
            # Create BOQ Items table
            boq_items_table = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='boqitems' AND xtype='U')
            CREATE TABLE boqitems (
                ItemID NVARCHAR(50) PRIMARY KEY,
                ProjectID NVARCHAR(50) NOT NULL,
                SectionID NVARCHAR(50),
                SheetName NVARCHAR(100),
                SourceRow INT,
                ItemNumber NVARCHAR(50),
                Description NTEXT,
                Quantity DECIMAL(18,4),
                Unit NVARCHAR(50),
                UnitRate DECIMAL(18,2),
                TotalCost DECIMAL(18,2),
                ProcessedAt DATETIME2,
                FOREIGN KEY (ProjectID) REFERENCES projects(ProjectID),
                FOREIGN KEY (SectionID) REFERENCES sections(SectionID)
            )
            """
            
            cursor.execute(projects_table)
            cursor.execute(sections_table)
            cursor.execute(boq_items_table)
            conn.commit()
            
            logging.info("Database tables created/verified successfully")
            
        except Exception as e:
            logging.error(f"Error creating tables: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def insert_project_record(self, project_data: Dict[str, Any], filename: str) -> bool:
        """
        Insert a project record into SQL Database
        
        Args:
            project_data: Dictionary containing project information
            filename: Source filename for tracking
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            
            project_id = project_data.get('project_id')
            # Handle project_date conversion
            project_date = None
            if project_data.get('project_date'):
                if isinstance(project_data['project_date'], datetime):
                    project_date = project_data['project_date'].date()
                else:
                    try:
                        project_date = datetime.strptime(str(project_data['project_date']), '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse project date: {project_data.get('project_date')}")
            
            # Insert query with MERGE to handle duplicates
            insert_query = """
            MERGE projects AS target
            USING (SELECT %s AS ProjectID) AS source
            ON target.ProjectID = source.ProjectID
            WHEN NOT MATCHED THEN
                INSERT (ProjectID, ProjectName, TotalCost, ProjectDescription, 
                       ProjectDate, ProjectCategory, SourceFilename, CreatedDate, LastModified)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            cursor.execute(insert_query, (
                project_id,  # for MERGE condition
                project_id,
                project_data.get('project_name', ''),
                project_data.get('total_cost'),
                project_data.get('project_description', ''),
                project_date,
                project_data.get('project_category', ''),
                filename,
                datetime.utcnow(),
                datetime.utcnow()
            ))
            
            conn.commit()
            logging.info(f"Successfully inserted project record with ID: {project_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error inserting project record: {str(e)}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def insert_section_records(self, section_records: List[Dict[str, Any]]) -> bool:
        """
        Insert section records into SQL Database
        
        Args:
            section_records: List of dictionaries containing section information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not section_records:
                logging.warning("No section records to insert")
                return True
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            successful_inserts = 0
            
            insert_query = """
            MERGE sections AS target
            USING (SELECT %s AS SectionID) AS source
            ON target.SectionID = source.SectionID
            WHEN NOT MATCHED THEN
                INSERT (SectionID, ProjectID, SectionName, Cost, Discipline, 
                       SourceFile, CreatedDate, LastModified)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            for record in section_records:
                try:
                    section_id = str(record.get('section_id', uuid.uuid4()))
                    cost = record.get('cost', 0) if record.get('cost') is not None else 0
                    
                    cursor.execute(insert_query, (
                        section_id,  # for MERGE condition
                        section_id,
                        record.get('project_id', ''),
                        record.get('section_name', ''),
                        cost,
                        record.get('discipline', ''),
                        record.get('source_file', ''),
                        datetime.utcnow(),
                        datetime.utcnow()
                    ))
                    
                    successful_inserts += 1
                    logging.debug(f"Successfully queued section: {record.get('section_name', 'Unknown')}")
                    
                except Exception as e:
                    logging.error(f"Error preparing section record: {str(e)}")
                    logging.error(f"Section record data: {record}")
                    continue
            
            conn.commit()
            logging.info(f"Successfully inserted {successful_inserts}/{len(section_records)} section records")
            return successful_inserts > 0
            
        except Exception as e:
            logging.error(f"Error in insert_section_records: {str(e)}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
                
                
    def insert_boq_records(self, records: List[Dict[str, Any]], project_id: str) -> bool:
            """
            Insert BOQ records into SQL Database
            
            Args:
                records: List of dictionaries containing BOQ information
                project_id: Project ID for the records
            
            Returns:
                bool: True if successful, False otherwise
            """
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                successful_inserts = 0
                skipped_no_total_cost = 0
                skipped_missing_fields = 0
                
                insert_query = """
                MERGE boqitems AS target
                USING (SELECT %s AS ItemID) AS source
                ON target.ItemID = source.ItemID
                WHEN NOT MATCHED THEN
                    INSERT (ItemID, ProjectID, SectionID, SheetName, SourceRow,
                        ItemNumber, Description, Quantity, Unit, UnitRate, 
                        TotalCost, ProcessedAt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                for record in records:
                    # Validate required fields
                    if not record.get('ItemID') or not record.get('ProjectID'):
                        logging.warning(f"Skipping record with missing ItemID or ProjectID: {record}")
                        skipped_missing_fields += 1
                        continue
                    
                    # Validate total_cost
                    total_cost = record.get('TotalCost')
                    if not is_valid_total_cost(total_cost):
                        logging.debug(f"Skipping record with invalid TotalCost: ItemID={record.get('ItemID')}, TotalCost={total_cost}")
                        skipped_no_total_cost += 1
                        continue
                    
                    # Convert total_cost to float if it's a valid string
                    if isinstance(total_cost, str):
                        total_cost = float(total_cost.strip())
                    
                    # Handle NaN values for numeric fields - convert to 0 if total_cost is valid
                    def clean_numeric_field(value):
                        """Convert NaN, None, or invalid numeric values to 0"""
                        if value is None:
                            return 0
                        if isinstance(value, str):
                            if value.strip().lower() in ['nan', '', 'null']:
                                return 0
                            try:
                                return float(value.strip())
                            except (ValueError, AttributeError):
                                return 0
                        if hasattr(value, '__iter__') and str(value).lower() == 'nan':  # Handle pandas NaN
                            return 0
                        try:
                            import math
                            if math.isnan(float(value)):
                                return 0
                        except (TypeError, ValueError):
                            pass
                        return value if value is not None else 0
                    
                    # Clean numeric fields
                    quantity = clean_numeric_field(record.get('Quantity'))
                    unit_rate = clean_numeric_field(record.get('UnitRate'))
                    source_row = record.get('SourceRow', 0)
                    if source_row is None:
                        source_row = 0
                    
                    # Clean unit field (string field)
                    unit = record.get('Unit', '')
                    if unit is None or (isinstance(unit, str) and unit.lower() == 'nan'):
                        unit = ''
                    
                    try:
                        cursor.execute(insert_query, (
                            record['ItemID'],  # for MERGE condition
                            record['ItemID'],
                            record['ProjectID'],
                            record.get('SectionID'),
                            record.get('SheetName', ''),
                            source_row,
                            record.get('ItemNumber', ''),
                            record.get('Description', ''),
                            quantity,  # Cleaned numeric value
                            unit,      # Cleaned string value
                            unit_rate, # Cleaned numeric value
                            total_cost,
                            datetime.utcnow()
                        ))
                        
                        successful_inserts += 1
                        logging.debug(f"Successfully queued BOQ record: ItemID={record['ItemID']}")
                        
                    except Exception as e:
                        logging.error(f"Error preparing BOQ record {record['ItemID']}: {str(e)}")
                        continue
                
                conn.commit()
                logging.info(f"BOQ Records processing complete: {successful_inserts} successful, "
                            f"{skipped_no_total_cost} skipped (invalid TotalCost), "
                            f"{skipped_missing_fields} skipped (missing fields)")
                
                return successful_inserts > 0
                    
            except Exception as e:
                logging.error(f"Error processing BOQ records: {str(e)}")
                return False
            finally:
                if 'conn' in locals():
                    conn.close()
    
                