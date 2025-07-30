from turbo import run_turbo, openai_or_rand_vector, search_turbo_enhanced
import requests
import logging
import json
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import sys
# import json
# import requests
# import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = log_dir / f"run_log_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)  # Print to terminal and log
    ]
)
logger = logging.getLogger(__name__)

# Redirect all prints to also log file
class LoggerWriter:
    def __init__(self, stream, logger_func):
        self.stream = stream
        self.logger_func = logger_func
    def write(self, message):
        self.stream.write(message)
        message = message.strip()
        if message:
            self.logger_func(message)
    def flush(self):
        self.stream.flush()

# Redirect print to logger
sys.stdout = LoggerWriter(sys.stdout, logger.info)
sys.stderr = LoggerWriter(sys.stderr, logger.error)
# Updated queries with detailed criteria
Queries = [
    {
        "title": "Tax Lawyer",
        "natural_language_description": "Seasoned attorney with a JD from a top U.S. law school and over three years of legal practice, specializing in corporate tax structuring and compliance. Has represented clients in IRS audits and authored legal opinions on federal tax code matters.",
        "hard_criteria": [
            "JD degree from an accredited U.S. law school",
            "3+ years of experience practicing law"
        ],
        "soft_criteria": [
            "Experience advising clients on tax implications of corporate or financial transactions",
            "Experience handling IRS audits, disputes, or regulatory inquiries",
            "Experience drafting legal opinions or filings related to federal and state tax compliance"
        ]
    },
    {
        "title": "Junior Corporate Lawyer",
        "natural_language_description": "Corporate lawyer with two years of experience at a top-tier international law firm, specializing in M&A support and cross-border contract negotiations. Trained at a leading European law school with additional background in international regulatory compliance.",
        "hard_criteria": [
            "2-4 years of experience as a Corporate Lawyer at a leading law firm in the USA, Europe, or Canada, or in-house at a major global organization",
            "Graduate of a reputed law school in the USA, Europe, or Canada"
        ],
        "soft_criteria": [
            "Experience supporting Corporate M&A transactions, including due diligence and legal documentation",
            "Experience drafting and negotiating legal contracts or commercial agreements",
            "Familiarity with international business law or advising on regulatory requirements across jurisdictions"
        ]
    },
    {
        "title": "Radiology",
        "natural_language_description": "Radiologist with an MD from India and several years of experience reading CT and MRI scans. Well-versed in diagnostic workflows and has worked on projects involving AI-assisted image analysis.",
        "hard_criteria": [
            "MD degree from a medical school in the U.S. or India"
        ],
        "soft_criteria": [
            "Board certification in Radiology (ABR, FRCR, or equivalent) or comparable credential",
            "3+ years of experience interpreting X-ray, CT, MRI, ultrasound, or nuclear medicine studies",
            "Expertise in radiology reporting, diagnostic protocols, differential diagnosis, or AI applications in medical imaging"
        ]
    },
    {
        "title": "Doctors (MD)",
        "natural_language_description": "U.S.-trained physician with over two years of experience as a general practitioner, focused on chronic care management, wellness screenings, and outpatient diagnostics. Skilled in telemedicine and patient education.",
        "hard_criteria": [
            "MD degree from a top U.S. medical school",
            "2+ years of clinical practice experience in the U.S.",
            "Experience working as a General Practitioner (GP)"
        ],
        "soft_criteria": [
            "Familiarity with EHR systems and managing high patient volumes in outpatient or family medicine settings",
            "Comfort with telemedicine consultations, patient triage, and interdisciplinary coordination"
        ]
    },
    {
        "title": "Biology Expert",
        "natural_language_description": "Biologist with a PhD from a top U.S. university, specializing in molecular biology and gene expression",
        "hard_criteria": [
            "Completed undergraduate studies in the U.S., U.K., or Canada",
            "PhD in Biology from a top U.S. university"
        ],
        "soft_criteria": [
            "Research experience in molecular biology, genetics, or cell biology, with publications in peer-reviewed journals",
            "Familiarity with experimental design, data analysis, and lab techniques such as CRISPR, PCR, or sequencing",
            "Experience mentoring students, teaching undergraduate biology courses, or collaborating on interdisciplinary research"
        ]
    },
    {
        "title": "Anthropology",
        "natural_language_description": "PhD student in anthropology at a top U.S. university, focused on labor migration and cultural identity",
        "hard_criteria": [
            "PhD (in progress or completed) from a distinguished program in sociology, anthropology, or economics",
            "PhD program started within the last 3 years"
        ],
        "soft_criteria": [
            "Demonstrated expertise in ethnographic methods, with substantial fieldwork or case study research involving cultural, social, or economic systems",
            "Strong academic output — published papers, working papers, or conference presentations on anthropological or sociological topics",
            "Experience applying anthropological theory to real-world or interdisciplinary contexts (e.g., migration, labor, technology, development), showing both conceptual depth and practical relevance"
        ]
    },
    {
        "title": "Mathematics PhD",
        "natural_language_description": "Mathematician with a PhD from a leading U.S, specializing in statistical inference and stochastic processes. Published and experienced in both theoretical and applied research.",
        "hard_criteria": [
            "Completed undergraduate studies in the U.S., U.K., or Canada",
            "PhD in Mathematics or Statistics from a top U.S. university"
        ],
        "soft_criteria": [
            "Research expertise in pure or applied mathematics, statistics, or probability, with peer-reviewed publications or preprints",
            "Proficiency in mathematical modeling, proof-based reasoning, or algorithmic problem-solving"
        ]
    },
    {
        "title": "Quantitative Finance",
        "natural_language_description": "MBA graduate from a top U.S. program with 3+ years of experience in quantitative finance, including roles in risk modeling and algorithmic trading at a global investment firm. Skilled in Python and financial modeling, with expertise in portfolio optimization and derivatives pricing.",
        "hard_criteria": [
            "MBA from a Prestigious U.S. university (M7 MBA)",
            "3+ years of experience in quantitative finance, including roles such as risk modeling, algorithmic trading, or financial engineering"
        ],
        "soft_criteria": [
            "Experience applying financial modeling techniques to real-world problems like portfolio optimization or derivatives pricing",
            "Proficiency with Python for quantitative analysis and exposure to financial libraries (e.g., QuantLib or equivalent)",
            "Demonstrated ability to work in high-stakes environments such as global investment firms, showing applied knowledge of quantitative methods in production settings"
        ]
    },
    {
        "title": "Bankers",
        "natural_language_description": "Healthcare investment banker with over two years at a leading advisory firm, focused on M&A for multi-site provider groups and digital health companies. Currently working in a healthcare-focused growth equity fund, driving diligence and investment strategy.",
        "hard_criteria": [
            "MBA from a U.S. university",
            "2+ years of prior work experience in investment banking, corporate finance, or M&A advisory"
        ],
        "soft_criteria": [
            "Specialized experience in healthcare-focused investment banking or private equity, including exposure to sub-verticals like biotech, pharma services, or provider networks",
            "Led or contributed to transactions involving healthcare M&A, recapitalizations, or growth equity investments",
            "Familiarity with healthcare-specific metrics, regulatory frameworks, and value creation strategies (e.g., payer-provider integration, RCM optimization)"
        ]
    },
    {
        "title": "Mechanical Engineers",
        "natural_language_description": "Mechanical engineer with over three years of experience in product development and structural design, using tools like SolidWorks and ANSYS. Led thermal system simulations and supported prototyping for electromechanical components in an industrial R&D setting.",
        "hard_criteria": [
            "Higher degree in Mechanical Engineering from an accredited university",
            "3+ years of professional experience in mechanical design, product development, or systems engineering"
        ],
        "soft_criteria": [
            "Experience with CAD tools (e.g., SolidWorks, AutoCAD) and mechanical simulation tools (e.g., ANSYS, COMSOL)",
            "Demonstrated involvement in end-to-end product lifecycle — from concept through prototyping to manufacturing or testing",
            "Domain specialization in areas like thermal systems, fluid dynamics, structural analysis, or mechatronics"
        ]
    }
]

def get_config_filename(query_title):
    """Map query titles to config filenames."""
    mapping = {
        "Tax Lawyer": "tax_lawyer.yml",
        "Junior Corporate Lawyer": "junior_corporate_lawyer.yml",
        "Radiology": "radiology.yml",
        "Doctors (MD)": "doctors_md.yml",
        "Biology Expert": "biology_expert.yml",
        "Anthropology": "anthropology.yml",
        "Mathematics PhD": "mathematics_phd.yml",
        "Quantitative Finance": "quantitative_finance.yml",
        "Bankers": "bankers.yml",
        "Mechanical Engineers": "mechanical_engineers.yml",
    }
    return mapping.get(query_title, query_title.lower().replace(" ", "_") + ".yml")

def evaluate_single_query(config_filename, candidate_ids, email):
    print(candidate_ids)
    """Evaluate a single query using the evaluation endpoint."""
    url = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "rupaknalla1034@gmail.com"
    }
    payload = {
        "config_path": config_filename,
        "object_ids": candidate_ids[:10]  # Ensure max 10 candidates
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            import json
            logger.info("Evaluation result:\n%s", json.dumps(result, indent=2))
            return result.get('average_final_score', 0.0)
        else:
            logger.error(f"Evaluation failed for {config_filename}: {response.status_code} - {response.text}")
            return 0.0
    except Exception as e:
        logger.error(f"Error evaluating {config_filename}: {e}")
        return 0.0

def main():
    """Main function with improved search and evaluation."""
    config_candidates = {}
    evaluation_scores = {}
    email = "rupaknalla1034@gmail.com"  # Replace with your email
    
    logger.info("Starting enhanced search process...")
    
    for query in Queries:
        # logger.info(f"\nProcessing query: {query['title']}")
        
        # Generate embeddings
        title_vec = openai_or_rand_vector(query["title"])
        description_vec = openai_or_rand_vector(query["natural_language_description"])
        
        # Generate embeddings for criteria
        hard_criteria_vectors = [openai_or_rand_vector(hard) for hard in query["hard_criteria"]]
        soft_criteria_vectors = [openai_or_rand_vector(soft) for soft in query["soft_criteria"]]
        
        # Use enhanced search function
        candidate_ids = search_turbo_enhanced(
            title_vec, 
            description_vec, 
            hard_criteria_vectors, 
            soft_criteria_vectors,
            query["hard_criteria"],  # Pass text for enhanced matching
            query["soft_criteria"]   # Pass text for enhanced matching
        )
        # print(candidate_ids)
        config_filename = get_config_filename(query["title"])
        config_candidates[config_filename] = candidate_ids
        
        # Evaluate this query
        # logger.info(f"Evaluating {config_filename}...")
        score = evaluate_single_query(config_filename, candidate_ids, email)
        evaluation_scores[config_filename] = score
        logger.info(f"Score for {config_filename}: {score:.3f}")
    
    # Print evaluation results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    total_score = 0.0
    for config_file, score in evaluation_scores.items():
        print(f"{config_file:<30} | Score: {score:.3f}")
        total_score += score
    
    avg_score = total_score / len(evaluation_scores) if evaluation_scores else 0.0
    print(f"\nAverage Score: {avg_score:.3f}")
    print("="*60)
    
    # Submit final results
    print("\n--- Final Submission Payload ---")
    final_payload = {"config_candidates": config_candidates}
    print(final_payload)
    
    # Submit to evaluation endpoint
    url = "https://mercor-dev--search-eng-interview.modal.run/grade"
    headers = {
        "Content-Type": "application/json",
        "Authorization": email
    }
    
    try:
        response = requests.post(url, headers=headers, json=final_payload, timeout=60)
        print(f"\n--- Final Submission Response ---")
        print(f"Status Code: {response.status_code}")
        data = json.loads(response.text)
        print("Response:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error submitting final results: {e}")

if __name__ == "__main__":
    main()