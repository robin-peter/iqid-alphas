# System Design Document

This document outlines the configuration, scripts, and considerations for robustness and scalability of the described system.

## 1. `config.json`

The `config.json` file is crucial for managing settings and parameters used by the Python scripts. This approach allows for easy modification of system behavior without altering the source code.

**Purpose:**
*   To store configurable parameters like API keys, file paths, database credentials, and operational flags.
*   To enable different configurations for various environments (development, testing, production).

**Structure and Examples:**

The configuration file will be in JSON format. Below is an example structure:

```json
{
  "global_settings": {
    "log_level": "INFO",
    "log_file": "system_activity.log"
  },
  "data_ingestion": {
    "api_url": "https://api.example.com/data",
    "api_key": "YOUR_API_KEY_HERE",
    "raw_data_path": "/data/raw/",
    "fetch_interval_seconds": 3600,
    "retry_attempts": 3,
    "timeout_seconds": 60
  },
  "data_processing": {
    "input_path": "/data/raw/",
    "output_path": "/data/processed/",
    "processing_rules": [
      {"field": "temperature", "action": "convert_celsius_to_fahrenheit"},
      {"field": "timestamp", "action": "parse_iso_datetime"}
    ],
    "validation_schema_path": "schemas/data_schema.json"
  },
  "reporting": {
    "processed_data_path": "/data/processed/",
    "report_output_path": "/reports/",
    "report_format": "csv", // or "json", "pdf"
    "email_recipients": ["manager@example.com", "team@example.com"],
    "smtp_server": "smtp.example.com",
    "smtp_port": 587
  }
}
```

**Key Considerations:**
*   **Security:** Sensitive information like API keys and database passwords should be handled securely. Consider using environment variables or a secrets management system in production environments, rather than storing them directly in the `config.json` file.
*   **Validation:** The scripts reading this configuration should validate its structure and the types of values to prevent runtime errors.
*   **Modularity:** Group related configurations together to improve readability and maintainability.

## 2. Python Script 1: `data_ingestion.py`

This script is responsible for fetching or receiving data from external sources and storing it in a raw format.

**Purpose:**
*   To connect to data sources (e.g., APIs, databases, message queues).
*   To retrieve data according to predefined schedules or triggers.
*   To store the raw, unaltered data in a designated location.
*   To handle initial data validation (e.g., checking for presence of data).
*   To manage errors during data retrieval, including retries and logging.

**Logical Description (Pseudo-code):**

```python
# data_ingestion.py

import json
import time
import requests # Example for API ingestion
import logging

# Load configuration
config = load_config('config.json')['data_ingestion']
setup_logging(config_global) # Using global config part

def fetch_data_from_api(url, api_key, params=None):
    """Fetches data from a given API endpoint."""
    headers = {'Authorization': f'Bearer {api_key}'}
    for attempt in range(config['retry_attempts']):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=config['timeout_seconds'])
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            logging.info(f"Successfully fetched data from {url}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt) # Exponential backoff
    logging.critical(f"Failed to fetch data from {url} after {config['retry_attempts']} attempts.")
    return None

def save_raw_data(data, filename_prefix="raw_data"):
    """Saves the raw data to a file with a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = f"{config['raw_data_path']}{filename_prefix}_{timestamp}.json"
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Raw data saved to {filepath}")
    except IOError as e:
        logging.error(f"Failed to save raw data to {filepath}: {e}")

def main():
    logging.info("Starting data ingestion process...")
    data = fetch_data_from_api(config['api_url'], config['api_key'])
    if data:
        save_raw_data(data)
    logging.info("Data ingestion process finished.")

if __name__ == "__main__":
    main()
```

**Key Operations:**
1.  Load necessary parameters from `config.json` (API URL, key, paths, retry logic).
2.  Implement a robust data fetching mechanism, possibly with retries and error handling.
3.  Store data in its original format (e.g., JSON, XML, CSV) in the specified raw data directory.
4.  Log all significant events, errors, and outcomes.

## 3. Python Script 2: `data_processing.py`

This script takes the raw data collected by `data_ingestion.py`, cleans it, transforms it, and prepares it for analysis or reporting.

**Purpose:**
*   To read raw data from the storage location.
*   To clean the data (e.g., handle missing values, correct inconsistencies).
*   To transform data (e.g., convert data types, derive new fields, apply business rules).
*   To validate data against a predefined schema or rules.
*   To store the processed, clean data in a structured format in a designated location.

**Logical Description (Pseudo-code):**

```python
# data_processing.py

import json
import os
import logging
# Potentially pandas for more complex transformations

# Load configuration
config = load_config('config.json')['data_processing']
config_ingestion = load_config('config.json')['data_ingestion'] # If needed for input path consistency
setup_logging(config_global)

def load_raw_data(filepath):
    """Loads raw data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Raw data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filepath}")
        return None

def apply_transformations(data, rules):
    """Applies a list of transformation rules to the data."""
    # Example: rule = {"field": "temperature", "action": "convert_celsius_to_fahrenheit"}
    # This would be a simplified placeholder for more complex logic
    transformed_data = data # Start with a copy
    for item in transformed_data: # Assuming data is a list of records
        for rule in rules:
            if rule['field'] in item:
                if rule['action'] == 'convert_celsius_to_fahrenheit' and isinstance(item[rule['field']], (int, float)):
                    item[rule['field']] = (item[rule['field']] * 9/5) + 32
                # Add more rule handlers here
    logging.info(f"Applied {len(rules)} transformation rules.")
    return transformed_data

def validate_data(data, schema_path):
    """Validates data against a JSON schema (conceptual)."""
    # In a real scenario, use a library like jsonschema
    # For now, this is a placeholder
    if not schema_path: # No schema validation if path not provided
        logging.info("No schema validation path provided, skipping validation.")
        return True
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        # Perform validation using the schema (e.g., jsonschema.validate(instance=data, schema=schema))
        logging.info(f"Data validated successfully against {schema_path}.")
        return True # Placeholder for actual validation result
    except Exception as e:
        logging.error(f"Data validation failed or schema error: {e}")
        return False


def save_processed_data(data, filename_prefix="processed_data"):
    """Saves the processed data to a file."""
    # Determine input filename to link processed data to raw, if multiple raw files
    timestamp = time.strftime("%Y%m%d_%H%M%S") # Or derive from input filename
    filepath = f"{config['output_path']}{filename_prefix}_{timestamp}.json"
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Processed data saved to {filepath}")
    except IOError as e:
        logging.error(f"Failed to save processed data to {filepath}: {e}")

def main():
    logging.info("Starting data processing...")
    # This example processes one file. A real system might loop through many.
    # Find the latest raw file, or process all unprocessed files
    raw_files = sorted([os.path.join(config_ingestion['raw_data_path'], f) 
                        for f in os.listdir(config_ingestion['raw_data_path']) if f.startswith('raw_data_')])
    
    if not raw_files:
        logging.info("No raw data files found to process.")
        return

    latest_raw_file = raw_files[-1] # Example: process the latest file
    logging.info(f"Processing raw data file: {latest_raw_file}")
    
    raw_data = load_raw_data(latest_raw_file)
    if raw_data:
        if validate_data(raw_data, config.get('validation_schema_path')): # Use .get for optional keys
            transformed_data = apply_transformations(raw_data, config['processing_rules'])
            save_processed_data(transformed_data)
        else:
            logging.warning(f"Skipping transformation and saving due to validation errors for {latest_raw_file}.")
    logging.info("Data processing finished.")

if __name__ == "__main__":
    main()
```

**Key Operations:**
1.  Load necessary parameters from `config.json` (input/output paths, transformation rules, schema location).
2.  Read raw data files.
3.  Perform data cleaning (handling missing values, type conversions).
4.  Apply transformations as defined in the configuration.
5.  Validate data integrity and structure.
6.  Save processed data to the specified directory, possibly in a more structured format (e.g., cleaned JSON, Parquet).
7.  Log processing steps and any data quality issues encountered.

## 4. Python Script 3: `reporting.py`

This script uses the processed data to generate reports, dashboards, or send notifications.

**Purpose:**
*   To read processed data.
*   To aggregate, summarize, or analyze data to generate insights.
*   To create reports in various formats (e.g., CSV, PDF, JSON).
*   To distribute reports (e.g., save to a file, send via email).
*   (Optional) To update a database or dashboard with the latest information.

**Logical Description (Pseudo-code):**

```python
# reporting.py

import json
import csv
import os
import logging
import smtplib # For email notifications
from email.mime.text import MIMEText

# Load configuration
config = load_config('config.json')['reporting']
config_processing = load_config('config.json')['data_processing'] # For input path consistency
setup_logging(config_global)

def load_processed_data(filepath):
    """Loads processed data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Processed data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from processed file: {filepath}")
        return None

def generate_csv_report(data, output_filepath):
    """Generates a CSV report from the data."""
    if not data or not isinstance(data, list) or not data[0]:
        logging.warning("No data to generate CSV report or data is not in expected list-of-dicts format.")
        return False
    
    try:
        with open(output_filepath, 'w', newline='') as csvfile:
            # Assuming data is a list of dictionaries
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        logging.info(f"CSV report generated at {output_filepath}")
        return True
    except IOError as e:
        logging.error(f"Failed to write CSV report to {output_filepath}: {e}")
        return False
    except Exception as e: # Catch other potential errors like empty data list
        logging.error(f"An unexpected error occurred during CSV generation: {e}")
        return False


def send_email_notification(subject, body, recipients, smtp_config):
    """Sends an email notification."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "system-notifier@example.com" # Should be configurable
    msg['To'] = ", ".join(recipients)

    try:
        with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
            # server.starttls() # If using TLS
            # server.login(smtp_config['username'], smtp_config['password']) # If auth needed
            server.sendmail(msg['From'], recipients, msg.as_string())
        logging.info(f"Email notification sent to {recipients}")
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

def main():
    logging.info("Starting reporting process...")
    # Example: find the latest processed file
    processed_files = sorted([os.path.join(config_processing['output_path'], f) 
                              for f in os.listdir(config_processing['output_path']) if f.startswith('processed_data_')])

    if not processed_files:
        logging.info("No processed data files found for reporting.")
        return

    latest_processed_file = processed_files[-1] # Example: use the latest file
    logging.info(f"Generating report from: {latest_processed_file}")
    
    processed_data = load_processed_data(latest_processed_file)

    if processed_data:
        report_filename = f"report_{time.strftime('%Y%m%d_%H%M%S')}.{config['report_format']}"
        report_filepath = os.path.join(config['report_output_path'], report_filename)

        report_generated = False
        if config['report_format'] == 'csv':
            report_generated = generate_csv_report(processed_data, report_filepath)
        # Add other formats (json, pdf) here
        # elif config['report_format'] == 'json':
        #     save_json_report(processed_data, report_filepath) 
        else:
            logging.warning(f"Unsupported report format: {config['report_format']}")

        if report_generated and config.get('email_recipients'):
            send_email_notification(
                subject="System Report Ready",
                body=f"The report {report_filename} has been generated and is available at {report_filepath}.",
                recipients=config['email_recipients'],
                smtp_config={"server": config['smtp_server'], "port": config['smtp_port']} # Add user/pass if needed
            )
            
    logging.info("Reporting process finished.")

if __name__ == "__main__":
    # Helper functions not defined in pseudo-code but assumed for real script:
    # def load_config(filepath): ...
    # def setup_logging(log_config): ...
    # import time # for timestamps in filenames
    main()
```

**Key Operations:**
1.  Load necessary parameters from `config.json` (input paths, report format, output locations, email settings).
2.  Read processed data.
3.  Perform aggregations or analyses as required for the report.
4.  Generate the report in the specified format(s).
5.  Save the report to the designated directory.
6.  If configured, distribute the report (e.g., via email).
7.  Log all actions and outcomes.

## 5. Robustness and Scalability

Ensuring the system is robust and scalable is critical for long-term viability and performance.

### Robustness (Error Handling, Reliability)

*   **Configuration Management:**
    *   Centralized `config.json` allows for easy changes without code modification.
    *   Include versioning or checksums for `config.json` if complex.
*   **Input Validation:**
    *   Each script should validate its inputs, especially data from external sources or previous steps.
    *   Use schemas (e.g., JSON Schema) for data validation in `data_processing.py`.
*   **Error Handling & Retries:**
    *   Implement comprehensive `try-except` blocks for I/O operations, API calls, and data transformations.
    *   Use specific exception handling rather than generic `except Exception:`.
    *   Implement retry mechanisms with exponential backoff for transient errors (e.g., network issues in `data_ingestion.py`).
    *   Define a "dead-letter queue" or error logging mechanism for data that repeatedly fails processing.
*   **Logging:**
    *   Implement detailed logging across all scripts using the `logging` module.
    *   Log levels (INFO, WARNING, ERROR, CRITICAL) should be configurable.
    *   Include timestamps, module names, and contextual information in log messages.
    *   Centralized logging (e.g., ELK stack, Splunk) for production systems.
*   **Idempotency:**
    *   Design processing steps to be idempotent where possible (i.e., running a step multiple times with the same input produces the same output), to handle retries or reprocessing safely.
    *   For example, check if a raw file has already been processed before starting.
*   **Transactional Operations:**
    *   For critical operations (e.g., updating a database), consider using transactions to ensure atomicity. This is less relevant for file-based processing but important if databases are involved.
*   **Monitoring & Alerting:**
    *   Set up monitoring for system health (CPU, memory, disk space) and application-level metrics (e.g., number of records processed, error rates).
    *   Implement alerts for critical failures or performance degradation.

### Scalability (Handling Increased Load)

*   **Data Volume:**
    *   **Streaming:** For very large or continuous data, consider stream processing frameworks (e.g., Apache Kafka, Spark Streaming, Flink) instead of batch processing.
    *   **Chunking:** Process large files in chunks rather than loading everything into memory at once (e.g., using `pandas` chunking capabilities or line-by-line processing for text files).
    *   **Efficient Data Formats:** Use efficient binary formats like Parquet or Avro for processed data, which offer better compression and query performance than JSON/CSV for large datasets.
*   **Processing Power:**
    *   **Asynchronous Operations:** Use `asyncio` for I/O-bound tasks in `data_ingestion.py` to handle multiple API calls concurrently.
    *   **Parallel Processing:** Utilize the `multiprocessing` module for CPU-bound tasks in `data_processing.py` to leverage multiple cores.
    *   **Distributed Computing:** For very large-scale processing, consider frameworks like Apache Spark or Dask, which can distribute computation across a cluster of machines.
*   **Modular Design:**
    *   The separation into three distinct scripts (`ingestion`, `processing`, `reporting`) allows each component to be scaled independently.
    *   For example, you could run multiple instances of `data_processing.py` if the processing step becomes a bottleneck.
*   **Task Queues:**
    *   Use a task queue (e.g., Celery with RabbitMQ or Redis) to manage the execution of different stages. This decouples the scripts and allows for better load distribution and scalability. For example, `data_ingestion.py` could place a message in a queue when new raw data is available, and worker processes for `data_processing.py` would pick up these messages.
*   **Database Optimization:**
    *   If a database is used (not explicitly in the pseudo-code but common), ensure proper indexing, query optimization, and connection pooling.
*   **Configuration for Scalability:**
    *   Parameters in `config.json` can be adjusted for scaled environments (e.g., number of worker processes, queue names, resource allocations).
*   **Cloud Services:**
    *   Leverage cloud services for scalable storage (e.g., AWS S3, Google Cloud Storage), databases (e.g., RDS, BigQuery), and compute (e.g., EC2, Lambda, Kubernetes). These services often provide auto-scaling capabilities.

By addressing these points, the system can be made more resilient to errors and capable of handling growing amounts of data and processing demands.
```
