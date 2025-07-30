# Candidate Search and Evaluation System

A sophisticated candidate matching system that uses vector embeddings and criteria-based filtering to find and rank job candidates. The system evaluates candidates against both hard criteria (strict requirements) and soft criteria (preferred qualifications) across various professional domains.

## Overview

This system processes job queries for different roles (Tax Lawyer, Corporate Lawyer, Radiology, Doctors, Biology Expert, etc.) and matches them against a candidate database using:

- **Vector embeddings** for semantic similarity
- **Hard criteria enforcement** (education, experience, location requirements)
- **Soft criteria scoring** (preferred skills and qualifications)
- **Multi-factor scoring algorithm** with prestige and experience weighting

## Project Structure

```
├── main.py              # Main execution script with job queries and evaluation
├── turbo.py             # Core search engine and database operations
├── .env                 # Environment variables (to be created)
├── logs/               # Generated log files
└── README.md           # This file
```

## Environment Setup

### 1. Prerequisites

- Python 3.8 or higher
- MongoDB access
- Turbopuffer API access
- VoyageAI API access (for embeddings)

### 2. Install Dependencies

Create a virtual environment and install required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install turbopuffer pymongo certifi python-dotenv voyageai numpy requests
```

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# VoyageAI API Key for embeddings
VOYAGE_API_KEY=your_voyage_api_key_here

# MongoDB Connection (already configured in code)
MONGO_URL=mongodb+srv://candidate:aQ7hHSLV9QqvQutP@hardfiltering.awwim.mongodb.net/

# Turbopuffer Configuration (already configured in code)
TURBOPUFFER_API_KEY=tpuf_dQHBpZEvl612XAdP0MvrQY5dbS0omPMy
TURBOPUFFER_REGION=aws-us-west-2

# Your email for API authorization
USER_EMAIL=your_email@example.com
```

### 4. Update Configuration

In `main.py`, update the email variable:

```python
email = "your_email@example.com"  # Replace with your actual email
```

## Running the Setup Script

### Initialize the Database (Optional)

If you need to migrate data from MongoDB to Turbopuffer:

```bash
python turbo.py migrate
```

To clear the Turbopuffer namespace:

```bash
python turbo.py delete
```

## Running the Main Evaluation

Execute the main search and evaluation process:

```bash
python main.py
```

This will:

1. **Process all job queries** - Generate embeddings for job titles, descriptions, and criteria
2. **Search candidates** - Use enhanced vector search with criteria filtering
3. **Score candidates** - Apply multi-factor scoring algorithm
4. **Evaluate results** - Submit to evaluation endpoints and receive scores
5. **Generate logs** - Create detailed logs in the `logs/` directory

## System Features

### Enhanced Search Algorithm

- **Multi-vector combination**: Combines title, description, and criteria vectors
- **Hard criteria enforcement**: Strict filtering on education, experience, and location
- **Semantic similarity**: Uses VoyageAI embeddings for content matching
- **Keyword matching**: Pattern-based matching for specific terms and requirements

### Scoring Components

1. **Hard Criteria (60% weight)**:
   - Years of experience requirements
   - Degree and certification matching
   - Geographic/country requirements
   - Professional keyword matching

2. **Soft Criteria (20% weight)**:
   - Preferred skills and qualifications
   - Industry-specific experience
   - Additional certifications

3. **Semantic Similarity (20% weight)**:
   - Vector-based content similarity
   - Overall profile matching

### Supported Job Categories

- Tax Lawyer
- Junior Corporate Lawyer  
- Radiology Specialist
- Medical Doctors (MD)
- Biology Expert
- Anthropology Researcher
- Mathematics PhD
- Quantitative Finance
- Investment Bankers
- Mechanical Engineers

## Logging and Monitoring

The system generates comprehensive logs including:

- Search process details
- Candidate scoring breakdowns
- API response details
- Error handling and retries
- Performance metrics

Logs are saved to `logs/run_log_YYYY-MM-DD_HH-MM-SS.txt`

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all environment variables are set correctly
2. **Database Connection**: Verify MongoDB connection string and permissions
3. **Embedding Failures**: Check VoyageAI API key and rate limits
4. **Evaluation Endpoint Errors**: Verify email authorization and payload format

### Debug Mode

For detailed debugging, the system includes extensive logging. Check the generated log files for detailed execution traces.

## Performance Optimization

The system includes several optimizations:

- **Concurrent processing** for database operations
- **Batch processing** for large datasets  
- **Vector caching** to reduce API calls
- **Smart candidate filtering** to improve relevance

## API Rate Limits

Be aware of rate limits for:
- VoyageAI embedding API
- Evaluation endpoints
- MongoDB query limits

The system includes retry logic and error handling for these scenarios.
```
