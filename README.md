# CC Agent Pipeline

This Azure Function is triggered by blob storage events and processes uploaded Excel files. It is designed to work with files organized by **project_oriented_country**, **project_start_year**, **project**, and **filename**.

---

## 🔧 How It Works

When an Excel file is uploaded to a specific path in Azure Blob Storage, this function is triggered automatically.

- The function first checks if the uploaded file is a valid Excel format.
- If valid, it extracts metadata from the blob path (e.g., project_oriented_country, project_start_year, project name, and file name).
- It then passes the blob and metadata to a custom processing function in `filehandler.py`.

---

## 📂 Blob Path Structure
```env
<Blob name>/{project_oriented_country}/{project_start_year}/{project}/{name}
```
### Example:
```env
docs/Qatar/2024/Lusail-water/report_q1.xlsx
```
---

## 🧾 Expected File Format

Only the following file types are processed:

.xlsx
.xls

Files that do not match this format are ignored.

---

## ⚙️ Setup local.settings.json

Update your local development environment with the required settings:
```env
{
  "IsEncrypted": false,
  "Values": {
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AzureWebJobsStorage": "<AzureWebJobsStorage>",
    "AZURE_OPENAI_ENDPOINT": "<AZURE_OPENAI_ENDPOINT>",
    "AZURE_OPENAI_API_KEY": "<AZURE_OPENAI_API_KEY>",
    "AZURE_SQL_SERVER": "<AZURE_SQL_SERVER>",
    "AZURE_SQL_DATABASE": "<AZURE_SQL_DATABASE>",
    "AZURE_SQL_USERNAME": "<AZURE_SQL_USERNAME>",
    "AZURE_SQL_PASSWORD": "<AZURE_SQL_PASSWORD>"
  }
}
```
---

## 📦 Project Structure
```env
project/
│
├── models/                     # Data models and service wrappers
│   ├── open_ai_services.py     # OpenAI API integrations
│   └── processing_models.py    # Domain models (Project, Section, BOQ), schema mapping, processing results
│
├── processors/                 # Excel or data transformation logic
│   └── excel_to_table_processor.py  # Processes Excel files, extracts project info with OpenAI, and stores data in Azure SQL
│
├── services/                   # Service logic (e.g., file operations)
│   └── filehandler.py
│
├── utils/                      # General utilities
│   ├── boq_utils.py            # BOQ processing helpers: description merging, cost validation, key sanitization
│   └── file_utils.py           # File type/format validation
├── env_constant.py             # Environment config constants
├── function_app.py             # Azure Function app entry point
├── host.json                   # Azure Functions host settings
├── local.settings.json         # Local dev environment config
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
---

## 🧠 Function Logic

### Trigger: Blob Trigger

@app.blob_trigger(
    arg_name="_blob", 
    path="docs/{project_oriented_country}/{project_start_year}/{project}/{name}", 
    connection="BLOB_CONNECTION_STRING"
)

### Logic Steps:

1. Trigger fires when a file is uploaded to the specified blob path.
2. Validation: Checks if the file is an Excel file via is_excel_file().
3. Metadata Parsing: Extracts project_oriented_country, project_start_year, project, and filename from blob path.
4. Processing: Calls process_excel_file() from services.filehandler.
5. Logging: Logs success or failure message depending on processing outcome.

---

## 📌 Notes

- Only Excel files are processed — all others are skipped.
- The blob name must follow the correct path format, or metadata parsing will fail.
- The function uses Azure App Settings in production for secrets and connection strings.