# Taxfix Tax Processing System - Setup Instructions

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Weaviate only)
- Google Cloud account with Document AI API enabled
- Min 8GB RAM
- 10MB disk space for document storage and database

## Directory Structure Setup

Create the following directory structure for raw data:

```
taxfix-taxflow/
├── app/                  # Main application code
├── data/                 # Data directory
│   ├── raw/              # Raw document storage
│   │   ├── receipts/     # Raw receipt documents (PDF, JPG, PNG)
│   │   └── income/       # Raw income statements (PDF, JPG, PNG)
│   ├── processed/        # Will store processed documents
│   └── db/               # Will store SQLite databases
├── notebooks/            # Optional analysis notebooks
│   ├── data/             # Notebook data files
│   └── processed_data/   # Processed data for models
└── docker-compose.yml    # Docker compose configuration for Weaviate
```

## Environment Configuration

1. Create a `.env` file in the project root:

```bash
# Google Cloud and Document AI
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
DOCUMENT_AI_LOCATION=eu  # or us, asia

# Document AI Processor IDs
RECEIPT_PROCESSOR_ID=your-receipt-processor-id
INCOME_STATEMENT_PROCESSOR_ID=your-income-statement-processor-id
OCCUPATION_CATEGORY_PROCESSOR_ID=your-occupation-processor-id

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
```

2. Create a `credentials` directory and add your Google Cloud credentials file:

```bash
mkdir -p credentials
# Copy your Google Cloud service account key to credentials/google-credentials.json
```

## Python Environment Setup

Setup your local Python environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


## Raw Data Requirements

Place your tax documents in the appropriate directories:

1. **Receipts**: Place receipt documents (PDF, JPG, PNG) in `data/raw/receipts/`
2. **Income Statements**: Place income statements/payslips in `data/raw/income/`

Document format requirements:
- PDF documents should be searchable (OCR processed)
- Images should be clear and high resolution (min 300 DPI)
- File size should not exceed 10MB per document

## Setup Document AI Processors

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to Document AI > Processors
3. Create the following processors:
   - Receipt parser (specialized)
   - Payslip/Income parser (specialized)
   - General OCR processor for occupation data
4. Copy the processor IDs to your `.env` file

## Running the Application

### Start Weaviate

```bash
docker-compose up -d
```

### Process Documents

```bash
python -m app.main --input data/raw --output data/processed --doc-type all --load-db
```

### Run Vector Database Population

```bash
python -m app.batch.build_vectors --db app/data/db/transactions.db --reset
```

### Run Tax Recommendation Models

```bash
python -m app.batch.process_enriched_data
python -m app.batch.hierarchical_clustering_model
python -m app.batch.causal_uplift_model
```

### Run Example Queries

```bash
python -m app.semantic.examples
```

### Full Pipeline
First create enriched data
```bash
python -m app.batch.process_enriched_data
python -m app.batch.hierarchical_clustering_model
python -m app.batch.causal_uplift_model
```

To run the complete pipeline in one command:

```bash
python -m app.main 
```

## Additional Resources

- Documentation: See `docs/` directory
- Example notebooks: See `notebooks/` directory
- Model explanations: See individual model README files in `app/batch/`

