import turbopuffer
import time
import concurrent.futures
from threading import Lock
import argparse
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
import math
import re
from collections import defaultdict
import json
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pymongo import MongoClient
import certifi

# Configuration
BATCH_SIZE = 10_000
TURBOPUFFER_REGION = "aws-us-west-2"
COLLECTION_NAME = "linkedin_data_subset"
DB_NAME = "interview_data"
TPUF_NAMESPACE_NAME = "Rupak_Nalla_tpuf_key"
MAX_RETRIES = 10
NUM_THREADS = 10

TURBOPUFFER_API_KEY = "tpuf_dQHBpZEvl612XAdP0MvrQY5dbS0omPMy"
MONGO_URL = "mongodb+srv://candidate:aQ7hHSLV9QqvQutP@hardfiltering.awwim.mongodb.net/"

# Initialize Turbopuffer client
tpuf = turbopuffer.Turbopuffer(
    api_key=TURBOPUFFER_API_KEY,
    region="aws-us-west-2",
)
ns = tpuf.namespace(TPUF_NAMESPACE_NAME)

@dataclass
class CandidateData:
    """Enhanced candidate data structure for better processing."""
    id: str
    name: str
    vector: List[float]
    rerank_summary: str
    experience: List[Dict]
    education: Dict
    years_of_experience: int
    country: str
    skills: List[str]
    awards_certifications: List[Dict]
    prestige_score: float
    
    @classmethod
    def from_mongo_doc(cls, doc: Dict) -> 'CandidateData':
        """Create CandidateData from MongoDB document."""
        return cls(
            id=str(doc.get("_id", "")),
            name=doc.get("name", ""),
            vector=doc.get("embedding", []),
            rerank_summary=doc.get("rerankSummary", ""),
            experience=doc.get("experience", []),
            education=doc.get("education", {}),
            years_of_experience=doc.get("yearsOfWorkExperience", 0),
            country=doc.get("country", ""),
            skills=doc.get("skills", []),
            awards_certifications=doc.get("awardsAndCertifications", []),
            prestige_score=doc.get("prestigeScore", 0.0)
        )

def get_mongo_collection(collection_name: str, db_name: str):
    """Get MongoDB collection with enhanced projection."""
    url = MONGO_URL
    client: MongoClient = MongoClient(url, tlsCAFile=certifi.where())
    return client[db_name][collection_name]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    try:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    except Exception:
        return 0.0

def weighted_vector_combination(vectors: List[List[float]], weights: List[float] = None) -> List[float]:
    """Combine multiple vectors with weights."""
    if not vectors or not all(vectors):
        return []
    
    if weights is None:
        weights = [1.0] * len(vectors)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return []
    weights = [w / total_weight for w in weights]
    
    # Ensure all vectors have same dimension
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        return []
    
    # Combine vectors
    combined = [0.0] * dim
    for vec, weight in zip(vectors, weights):
        for i, val in enumerate(vec):
            combined[i] += val * weight
    
    return combined

class CriteriaAnalyzer:
    """Enhanced criteria analysis for better matching."""
    
    def __init__(self):
        self.degree_patterns = {
            'jd': r'\b(jd|j\.d\.?|juris\s+doctor|law\s+degree)\b',
            'md': r'\b(md|m\.d\.?|medical\s+degree|doctor\s+of\s+medicine)\b',
            'phd': r'\b(phd|ph\.d\.?|doctorate|doctoral)\b',
            'mba': r'\b(mba|m\.b\.a\.?|master\s+of\s+business)\b',
            'bachelor': r'\b(bachelor|b\.a\.?|b\.s\.?|bs|ba)\b',
            'master': r'\b(master|m\.a\.?|m\.s\.?|ms|ma)\b'
        }
        
        self.country_patterns = {
            'us': r'\b(u\.s\.?a?|united\s+states?|american?)\b',
            'canada': r'\b(canada|canadian)\b',
            'uk': r'\b(u\.k\.?|united\s+kingdom|britain|british)\b',
            'europe': r'\b(europe|european)\b',
            'india': r'\b(india|indian)\b'
        }
        
        self.experience_patterns = {
            'years': r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|practice|work)',
            'range': r'(\d+)-(\d+)\s*years?',
            'minimum': r'(?:minimum|min|at\s+least)\s+(\d+)\+?\s*years?'
        }
        
    def extract_years_requirement(self, text: str) -> Tuple[int, int]:
        """Extract minimum and maximum years of experience."""
        text_lower = text.lower()
        min_years, max_years = 0, float('inf')
        
        # Check for minimum patterns
        min_matches = re.findall(self.experience_patterns['minimum'], text_lower)
        if min_matches:
            min_years = max(min_years, int(min_matches[0]))
        
        # Check for range patterns
        range_matches = re.findall(self.experience_patterns['range'], text_lower)
        if range_matches:
            range_min, range_max = int(range_matches[0][0]), int(range_matches[0][1])
            min_years = max(min_years, range_min)
            max_years = min(max_years, range_max)
        
        # Check for simple years patterns
        year_matches = re.findall(self.experience_patterns['years'], text_lower)
        if year_matches:
            min_years = max(min_years, int(year_matches[0]))
        
        return min_years, max_years if max_years != float('inf') else None
    
    def extract_required_degrees(self, text: str) -> List[str]:
        """Extract required degree types."""
        text_lower = text.lower()
        degrees = []
        
        for degree_type, pattern in self.degree_patterns.items():
            if re.search(pattern, text_lower):
                degrees.append(degree_type)
        
        return degrees
    
    def extract_required_countries(self, text: str) -> List[str]:
        """Extract required countries/regions."""
        text_lower = text.lower()
        countries = []
        
        for country, pattern in self.country_patterns.items():
            if re.search(pattern, text_lower):
                countries.append(country)
        
        return countries
    
    def extract_required_skills(self, text: str) -> List[str]:
        """Extract specific skills or expertise areas."""
        text_lower = text.lower()
        skills = []
        
        # Common skill patterns
        skill_patterns = [
            r'(python|java|c\+\+|javascript|sql)',
            r'(machine\s+learning|deep\s+learning|ai|artificial\s+intelligence)',
            r'(financial\s+modeling|risk\s+management|portfolio\s+optimization)',
            r'(m&a|mergers?\s+and\s+acquisitions?)',
            r'(corporate\s+law|tax\s+law|intellectual\s+property)',
            r'(radiology|oncology|cardiology|pathology)',
            r'(cad|solidworks|ansys|autocad)',
            r'(quantitative\s+finance|algorithmic\s+trading)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        return skills

class EnhancedCandidateScorer:
    """Enhanced scoring system with strict hard criteria enforcement."""
    
    def __init__(self):
        self.analyzer = CriteriaAnalyzer()
        
    def score_years_experience(self, candidate: CandidateData, required_min: int, required_max: Optional[int] = None) -> float:
        """Score based on years of experience requirement."""
        candidate_years = candidate.years_of_experience
        
        if candidate_years >= required_min:
            if required_max and candidate_years > required_max:
                # Slightly penalize over-qualification
                return 0.9
            return 1.0
        else:
            # Graduated penalty for under-qualification
            ratio = candidate_years / required_min if required_min > 0 else 0
            return max(0.0, ratio * 0.5)  # Max 50% score if under-qualified
    
    def score_degree_requirement(self, candidate: CandidateData, required_degrees: List[str]) -> float:
        """Score based on degree requirements."""
        if not required_degrees:
            return 1.0
        
        candidate_degrees = []
        education = candidate.education
        
        # Extract degrees from candidate education
        if isinstance(education, dict) and 'degrees' in education:
            for degree_info in education['degrees']:
                degree_text = str(degree_info.get('degree', '')).lower()
                field_text = str(degree_info.get('fieldOfStudy', '')).lower()
                combined_text = f"{degree_text} {field_text}"
                
                for degree_type, pattern in self.analyzer.degree_patterns.items():
                    if re.search(pattern, combined_text):
                        candidate_degrees.append(degree_type)
        
        # Also check rerank summary for degree mentions
        summary_text = candidate.rerank_summary.lower()
        for degree_type, pattern in self.analyzer.degree_patterns.items():
            if re.search(pattern, summary_text):
                candidate_degrees.append(degree_type)
        
        # Check certifications
        for cert in candidate.awards_certifications:
            cert_name = str(cert.get('name', '')).lower()
            for degree_type, pattern in self.analyzer.degree_patterns.items():
                if re.search(pattern, cert_name):
                    candidate_degrees.append(degree_type)
        
        candidate_degrees = list(set(candidate_degrees))
        
        # Score based on matches
        matches = sum(1 for req_degree in required_degrees if req_degree in candidate_degrees)
        return matches / len(required_degrees) if required_degrees else 1.0
    
    def score_country_requirement(self, candidate: CandidateData, required_countries: List[str]) -> float:
        """Score based on country/region requirements."""
        if not required_countries:
            return 1.0
        
        candidate_country = candidate.country.lower()
        candidate_summary = candidate.rerank_summary.lower()
        candidate_text = f"{candidate_country} {candidate_summary}"
        
        # Check for country matches
        for req_country in required_countries:
            pattern = self.analyzer.country_patterns.get(req_country)
            if pattern and re.search(pattern, candidate_text):
                return 1.0
        
        # Check education locations
        education = candidate.education
        if isinstance(education, dict) and 'degrees' in education:
            for degree_info in education['degrees']:
                school_text = str(degree_info.get('school', '')).lower()
                if any(re.search(self.analyzer.country_patterns.get(country, ''), school_text) 
                       for country in required_countries):
                    return 1.0
        
        return 0.0
    
    def score_semantic_similarity(self, candidate: CandidateData, query_vector: List[float]) -> float:
        """Score based on semantic similarity."""
        return cosine_similarity(candidate.vector, query_vector)
    
    def score_hard_criteria(self, candidate: CandidateData, criteria_text: str, criteria_vector: List[float]) -> Dict[str, float]:
        """Comprehensive hard criteria scoring."""
        scores = {}
        
        # Extract requirements
        min_years, max_years = self.analyzer.extract_years_requirement(criteria_text)
        required_degrees = self.analyzer.extract_required_degrees(criteria_text)
        required_countries = self.analyzer.extract_required_countries(criteria_text)
        
        # Score each component
        scores['years'] = self.score_years_experience(candidate, min_years, max_years) if min_years > 0 else 1.0
        scores['degree'] = self.score_degree_requirement(candidate, required_degrees)
        scores['country'] = self.score_country_requirement(candidate, required_countries)
        scores['semantic'] = self.score_semantic_similarity(candidate, criteria_vector)
        
        # Text-based keyword matching
        scores['keywords'] = self.score_keyword_matching(candidate, criteria_text)
        
        return scores
    
    def score_keyword_matching(self, candidate: CandidateData, criteria_text: str) -> float:
        """Enhanced keyword matching."""
        criteria_lower = criteria_text.lower()
        candidate_text = f"{candidate.rerank_summary} {candidate.name}".lower()
        
        # Extract important keywords
        keywords = []
        
        # Professional terms
        prof_patterns = [
            r'\b(attorney|lawyer|counsel|advocate)\b',
            r'\b(doctor|physician|md|radiologist)\b',
            r'\b(engineer|engineering|technical)\b',
            r'\b(banker|banking|finance|financial)\b',
            r'\b(professor|researcher|scientist)\b'
        ]
        
        for pattern in prof_patterns:
            if re.search(pattern, criteria_lower):
                keywords.extend(re.findall(pattern, criteria_lower))
        
        # Score based on keyword presence
        if not keywords:
            return 0.5  # Neutral score if no specific keywords
        
        matches = sum(1 for keyword in keywords if keyword in candidate_text)
        return matches / len(keywords)

def search_turbo_enhanced(title_vec: List[float], description_vec: List[float], 
                         hard_criteria_vectors: List[List[float]], soft_criteria_vectors: List[List[float]],
                         hard_criteria_texts: List[str] = None, soft_criteria_texts: List[str] = None) -> List[str]:
    """Enhanced search with comprehensive candidate evaluation."""
    
    # logger.info("🧠 Generating embeddings...")
    # logger.info(f"Generated {len(hard_criteria_vectors)} hard criteria and {len(soft_criteria_vectors)} soft criteria vectors")
    
    # Create combined query vector
    all_vectors = [v for v in [title_vec, description_vec] + hard_criteria_vectors if v]
    weights = [0.3, 0.4] + [0.3/len(hard_criteria_vectors)] * len(hard_criteria_vectors)
    combined_query_vector = weighted_vector_combination(all_vectors, weights[:len(all_vectors)])
    
    if not combined_query_vector:
        logger.error("Failed to create combined query vector")
        return []
    
    # Retrieve candidates from vector database
    try:
        result = ns.query(
            rank_by=("vector", "ANN", combined_query_vector),
            top_k=500,  # Retrieve more for better filtering
            include_attributes=True
        )
        raw_candidates = result.rows
        # logger.info(f"Retrieved {len(raw_candidates)} candidates from ANN search.")
    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        return []
    
    # Fetch full candidate data from MongoDB for detailed scoring
    candidate_ids = [safe_get_attribute(c, 'id') for c in raw_candidates if safe_get_attribute(c, 'id')]
    mongo_candidates = fetch_candidates_from_mongo(candidate_ids[:100])  # Limit for performance
    
    # Score candidates with enhanced criteria
    scorer = EnhancedCandidateScorer()
    scored_candidates = []
    
    for candidate in mongo_candidates:
        try:
            score_details = calculate_comprehensive_score(
                candidate, combined_query_vector, hard_criteria_vectors, 
                soft_criteria_vectors, hard_criteria_texts or [], soft_criteria_texts or [], scorer
            )
            
            scored_candidates.append((score_details['final_score'], candidate, score_details))
        except Exception as e:
            logger.warning(f"Error scoring candidate {candidate.id}: {e}")
            continue
    
    # Sort by final score
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Enhanced logging
    logger.info("Top 10 candidates with detailed scores:")
    for i, (final_score, candidate, details) in enumerate(scored_candidates[:10]):
        logger.info(f"Rank {i+1}: Score {final_score:.4f} | ID: {candidate.id} | Name: {candidate.name}")
        logger.info(f"  Summary: {candidate.rerank_summary[:100]}...")
        logger.info(f"  Hard criteria: {details.get('hard_score', 0):.3f}, Soft criteria: {details.get('soft_score', 0):.3f}")
    
    # Return top candidate IDs
    result_ids = [candidate.id for _, candidate, _ in scored_candidates[:10]]
    logger.info(f"Returning {len(result_ids)} candidate IDs")
    
    return result_ids

def fetch_candidates_from_mongo(candidate_ids: List[str]) -> List[CandidateData]:
    """Fetch full candidate data from MongoDB."""
    collection = get_mongo_collection(COLLECTION_NAME, DB_NAME)
    
    # Convert string IDs to ObjectIds
    from bson import ObjectId
    object_ids = []
    for cid in candidate_ids:
        try:
            object_ids.append(ObjectId(cid))
        except Exception:
            continue
    
    # Fetch documents
    cursor = collection.find({"_id": {"$in": object_ids}})
    candidates = []
    
    for doc in cursor:
        try:
            candidates.append(CandidateData.from_mongo_doc(doc))
        except Exception as e:
            logger.warning(f"Error processing candidate {doc.get('_id')}: {e}")
            continue
    
    return candidates

def calculate_comprehensive_score(candidate: CandidateData, query_vector: List[float],
                                hard_criteria_vectors: List[List[float]], soft_criteria_vectors: List[List[float]],
                                hard_criteria_texts: List[str], soft_criteria_texts: List[str],
                                scorer: EnhancedCandidateScorer) -> Dict[str, float]:
    """Calculate comprehensive score with hard criteria enforcement."""
    
    # Hard criteria evaluation (must pass ALL)
    hard_scores = []
    hard_components = []
    
    for i, (hard_vec, hard_text) in enumerate(zip(hard_criteria_vectors, hard_criteria_texts)):
        hard_detail = scorer.score_hard_criteria(candidate, hard_text, hard_vec)
        
        # Weighted combination of hard criteria components
        component_score = (
            0.3 * hard_detail['semantic'] +
            0.25 * hard_detail['years'] +
            0.25 * hard_detail['degree'] +
            0.1 * hard_detail['country'] +
            0.1 * hard_detail['keywords']
        )
        
        hard_scores.append(component_score)
        hard_components.append(hard_detail)
    
    # Hard criteria enforcement - must meet minimum threshold for ALL
    hard_threshold = 0.6
    hard_min_score = min(hard_scores) if hard_scores else 0.0
    hard_avg_score = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
    
    # Soft criteria evaluation (bonus)
    soft_scores = []
    for soft_vec in soft_criteria_vectors:
        soft_score = cosine_similarity(candidate.vector, soft_vec)
        soft_scores.append(soft_score)
    
    soft_avg_score = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
    
    # Overall semantic similarity
    semantic_score = cosine_similarity(candidate.vector, query_vector)
    
    # Final score calculation with hard criteria gate
    if hard_min_score < hard_threshold:
        # Severe penalty for not meeting hard criteria
        final_score = hard_min_score * 0.4  # Maximum 40% of minimum hard score
    else:
        # Standard scoring for candidates meeting hard criteria
        final_score = (
            0.2 * semantic_score +
            0.6 * hard_avg_score +
            0.2 * soft_avg_score
        )
        
        # Bonus for exceptional hard criteria performance
        if hard_min_score > 0.8:
            final_score *= 1.1
    
    return {
        'final_score': min(final_score, 1.0),
        'hard_score': hard_avg_score,
        'hard_min': hard_min_score,
        'soft_score': soft_avg_score,
        'semantic_score': semantic_score,
        'hard_components': hard_components
    }

def safe_get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get attribute from various object types."""
    try:
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name, default)
        
        if hasattr(obj, '__getitem__'):
            try:
                return obj[attr_name]
            except (KeyError, TypeError):
                pass
        
        if hasattr(obj, 'model_extra') and obj.model_extra:
            return obj.model_extra.get(attr_name, default)
            
        if hasattr(obj, '__dict__'):
            return obj.__dict__.get(attr_name, default)
            
        return default
    except Exception:
        return default

# Keep existing utility functions
load_dotenv()

def openai_or_rand_vector(text: str) -> List[float]:
    """Generate embeddings using VoyageAI or fallback to random vectors."""
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    if not VOYAGE_API_KEY:
        print("VOYAGE_API_KEY not set, using random 1024-dim vectors")
        import random
        return [random.random() for _ in range(1024)]

    try:
        from voyageai import Client
        client = Client(api_key=VOYAGE_API_KEY)
        response = client.embed(texts=[text], model="voyage-3")
        return response.embeddings[0]
    except Exception as e:
        print(f"Embedding error: {e}, using random vector.")
        import random
        return [random.random() for _ in range(1024)]

# Keep existing migration functions unchanged
def run_turbo():
    """Migration function."""
    parser = argparse.ArgumentParser(description="MongoDB to Turbopuffer migration tool")
    parser.add_argument("action", choices=["delete", "migrate"], help="Action to perform", default="migrate", nargs="?")
    args = parser.parse_args()

    if args.action == "delete":
        logger.info("Clearing Turbopuffer namespace...")
        delete_namespace()
        return

    logger.info("Starting migration from MongoDB to Turbopuffer")
    
    total_docs = get_total_document_count()
    logger.info(f"Total documents to process: {total_docs}")

    batch_ranges = [(skip, BATCH_SIZE) for skip in range(0, total_docs, BATCH_SIZE)]

    total_processed = 0
    lock = Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(fetch_and_upsert_batch, skip, limit) for skip, limit in batch_ranges]
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_count = future.result()
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                batch_count = 0

            with lock:
                total_processed += batch_count
            logger.info(f"Total processed so far: {total_processed}")

    logger.info(f"Migration completed! Total documents processed: {total_processed}")

def delete_namespace():
    """Delete namespace function."""
    try:
        ns.delete_all()
        logger.info("Namespace cleared successfully")
    except Exception as e:
        logger.error(f"Namespace already cleared")

def fetch_and_upsert_batch(skip: int, limit: int):
    """Fetch and upsert batch function."""
    collection = get_mongo_collection(collection_name=COLLECTION_NAME, db_name=DB_NAME)

    logger.info(f"Fetching batch: skip={skip}, limit={limit}")

    cursor = (
        collection.find({}, {"embedding": 1, "email": 1, "rerankSummary": 1, "country": 1, "name": 1, "linkedinId": 1})
        .sort("_id", 1)
        .skip(skip)
        .limit(limit)
    )

    batch = []
    for doc in cursor:
        profile = {
            "id": str(doc.get("_id")),
            "vector": doc.get("embedding", []),
            "email": doc.get("email", ""),
            "rerank_summary": doc.get("rerankSummary", ""),
            "country": doc.get("country", ""),
            "name": doc.get("name", ""),
            "linkedin_id": doc.get("linkedinId", ""),
        }
        batch.append(profile)

    if not batch:
        logger.info("No more documents found")
        return 0

    if upsert_batch_to_turbopuffer(batch):
        return len(batch)
    else:
        return 0

def upsert_batch_to_turbopuffer(batch):
    """Upsert batch to turbopuffer function."""
    for i in range(MAX_RETRIES):
        try:
            ns.write(
                upsert_rows=batch,
                distance_metric="cosine_distance",
                schema={
                    "id": "string",
                    "rerank_summary": {"type": "string", "full_text_search": True},
                },
            )

            logger.info(f"Successfully upserted batch to Turbopuffer")
            return True
        except Exception as e:
            logger.error(f"Error upserting batch to Turbopuffer: {e}")
            if i < MAX_RETRIES - 1:
                logger.info(f"Retrying in {i + 1} seconds...")
                time.sleep(i + 1)
            else:
                logger.error(f"Turbopuffer upsert failed after {MAX_RETRIES} attempts")
                return False

def get_total_document_count():
    """Get total document count function."""
    collection = get_mongo_collection(collection_name=COLLECTION_NAME, db_name=DB_NAME)
    return collection.count_documents({})