import pdfplumber
from sentence_transformers import SentenceTransformer, util
import re
import logging
from typing import Dict, Tuple, List
import numpy as np
import json
from dataclasses import dataclass
import uuid
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Question:
    id: str
    text: str
    student_answer: str
    sample_answer: str
    parameters: str
    keywords: List[str]
    points: float

class OptimizedAssignmentGrader:
    def __init__(self, pdf_path: str, parameters: Dict[str, str], sample_answers: Dict[str, str], keywords: Dict[str, List[str]], total_points: float):
        start_time = time.time()
        self.pdf_path = pdf_path
        self.parameters = parameters
        self.sample_answers = sample_answers
        self.keywords = keywords
        self.total_points = total_points
        logger.info(f"Initializing model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        self.questions: List[Question] = []
        self.keyword_penalty = 0.1  # Default, overridden by leniency

    def extract_pdf_content(self) -> None:
        try:
            start_time = time.time()
            with pdfplumber.open(self.pdf_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            pattern = r"Q(\d+):\s*(.*?)\s*Ans:\s*(.*?)(?=Q\d+:|$)"
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for q_num, question, answer in matches:
                q_id = f"Q{q_num}"
                points = self.total_points / len(matches)
                self.questions.append(Question(
                    id=q_id,
                    text=question.strip(),
                    student_answer=answer.strip(),
                    sample_answer=self.sample_answers.get(q_id, ""),
                    parameters=self.parameters.get(q_id, ""),
                    keywords=self.keywords.get(q_id, []),
                    points=points
                ))
            logger.info(f"Extracted {len(self.questions)} questions from PDF in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def parse_parameters(self, params: str) -> Tuple[Dict[str, float], str]:
        if not params:
            return {"similarity": 1.0}, "not lenient"
        try:
            param_dict = {}
            leniency = "not lenient"  # Default
            for param in params.split(","):
                name, value = param.split(":") if ":" in param else (param, "1.0")
                name = name.strip().lower()
                if name == "leniency":
                    leniency = value.strip().lower()
                else:
                    param_dict[name] = float(value)
            total_weight = sum(param_dict.values())
            if total_weight > 0:
                for key in param_dict:
                    param_dict[key] /= total_weight
            return param_dict, leniency
        except Exception as e:
            logger.warning(f"Parameter parsing failed: {e}. Using default similarity.")
            return {"similarity": 1.0}, "not lenient"

    def check_keywords(self, answer: str, keywords: List[str], leniency: str) -> Tuple[float, List[str]]:
        if not keywords:
            return 1.0, []
        answer_lower = answer.lower()
        missing_keywords = [kw for kw in keywords if kw.lower() not in answer_lower]
        penalty = {"very lenient": 0.05, "lenient": 0.1, "not lenient": 0.2}.get(leniency, 0.1)
        keyword_score = 1.0 - (len(missing_keywords) * penalty)
        return max(0.0, keyword_score), missing_keywords

    def evaluate_answer(self, question: Question) -> Tuple[float, List[str]]:
        if not question.student_answer or not question.sample_answer:
            logger.debug(f"No answer or sample for {question.id}")
            return 0.0, question.keywords
        start_time = time.time()
        embeddings = self.model.encode([question.student_answer, question.sample_answer], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        params, leniency = self.parse_parameters(question.parameters)
        
        # Apply leniency to similarity
        similarity_boost = {"very lenient": 1.2, "lenient": 1.1, "not lenient": 1.0}.get(leniency, 1.0)
        adjusted_similarity = min(similarity * similarity_boost, 1.0)
        
        score = adjusted_similarity * params.get("similarity", 1.0)
        
        # Relax clarity and completeness in lenient modes
        clarity_relax = {"very lenient": 0.8, "lenient": 0.9, "not lenient": 1.0}.get(leniency, 1.0)
        completeness_relax = {"very lenient": 0.8, "lenient": 0.9, "not lenient": 1.0}.get(leniency, 1.0)
        
        if "clarity" in params:
            word_count = len(question.student_answer.split())
            clarity_score = min(1.0, word_count / 15)
            score *= (clarity_score * clarity_relax * params.get("clarity", 0.0) + 
                     (1 - params.get("clarity", 0.0)))
        if "accuracy" in params and adjusted_similarity < 0.5:
            score *= 0.9 * params.get("accuracy", 0.0) + (1 - params.get("accuracy", 0.0))
        if "completeness" in params:
            length_ratio = len(question.student_answer) / len(question.sample_answer)
            completeness_score = min(1.0, length_ratio)
            score *= (completeness_score * completeness_relax * params.get("completeness", 0.0) + 
                     (1 - params.get("completeness", 0.0)))
        
        keyword_score, missing_keywords = self.check_keywords(question.student_answer, question.keywords, leniency)
        score *= keyword_score
        logger.debug(f"Evaluated {question.id} in {time.time() - start_time:.2f} seconds")
        return max(0.0, min(score * question.points, question.points)), missing_keywords

    def grade_assignment(self) -> Tuple[float, Dict[str, Tuple[float, List[str]]]]:
        start_time = time.time()
        self.extract_pdf_content()
        if not self.questions:
            logger.error("No questions extracted")
            return 0.0, {}
        grading_start = time.time()
        results = [self.evaluate_answer(q) for q in self.questions]  # Sequential processing
        logger.info(f"Grading completed in {time.time() - grading_start:.2f} seconds")
        score_dict = {q.id: (score, missing_keywords) for q, (score, missing_keywords) in zip(self.questions, results)}
        total_score = sum(score for score, _ in results)
        logger.info(f"Total score: {total_score:.2f}/{self.total_points} in {time.time() - start_time:.2f} seconds")
        return round(total_score, 2), score_dict

    def generate_report(self) -> str:
        total_score, scores = self.grade_assignment()
        report = {
            "assignment_id": str(uuid.uuid4()),
            "total_points": self.total_points,
            "earned_points": total_score,
            "questions": [
                {
                    "id": q.id,
                    "question": q.text,
                    "student_answer": q.student_answer,
                    "sample_answer": q.sample_answer,
                    "parameters": q.parameters,
                    "keywords": q.keywords,
                    "missing_keywords": scores.get(q.id, (0.0, []))[1],
                    "points_earned": scores.get(q.id, (0.0, []))[0],
                    "points_possible": q.points
                } for q in self.questions
            ]
        }
        return json.dumps(report, indent=2)

# Run the grader for hello.pdf with long answers
if __name__ == "__main__":
    start_time = time.time()
    pdf_path = "hello.pdf"
    parameters = {
        "Q1": "accuracy:0.7,clarity:0.3,leniency:very lenient",
        "Q2": "completeness:0.5,similarity:0.5,leniency:lenient"
    }
    sample_answers = {
        "Q1": "Python is a high-level, versatile programming language used for web development, data science, and automation, known for its readable syntax and extensive libraries.",
        "Q2": "A loop is a programming construct that repeats a block of code until a condition is met, used for tasks like iterating over data or automating processes."
    }
    keywords = {
        "Q1": ["Python", "programming", "syntax"],
        "Q2": ["loop", "repeat", "condition"]
    }
    total_points = 20.0
    grader = OptimizedAssignmentGrader(pdf_path, parameters, sample_answers, keywords, total_points)
    report = grader.generate_report()
    with open("grading_report.json", "w") as f:
        f.write(report)
    print(json.dumps(json.loads(report), indent=2))
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")