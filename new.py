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
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, ne_chunk
import os
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded with retries."""
    nltk.data.path = ['C:\\Users\\saad4\\AppData\\Roaming\\nltk_data']
    resource_paths = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab',
        'words': 'corpora/words'
    }
    max_retries = 3
    for resource, path in resource_paths.items():
        for attempt in range(max_retries):
            try:
                nltk.data.find(path)
                logger.debug(f"NLTK resource {resource} found.")
                break
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource} (Attempt {attempt + 1}/{max_retries})")
                try:
                    nltk.download(resource, quiet=True, download_dir='C:\\Users\\saad4\\AppData\\Roaming\\nltk_data')
                    logger.info(f"Successfully downloaded {resource}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to download {resource}: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Could not download {resource} after {max_retries} attempts.")

# Run NLTK resource check
ensure_nltk_resources()

@dataclass
class Question:
    id: str
    text: str
    student_answer: str
    sample_answers: List[str]
    parameters: str
    keywords: Dict[str, float]
    points: float

class OptimizedAssignmentGrader:
    def __init__(self, pdf_path: str, parameters: Dict[str, str], sample_answers: Dict[str, List[str]], keywords: Dict[str, Dict[str, float]], total_points: float):
        """Initialize grader with PDF path, parameters, sample answers, keywords, and total points."""
        start_time = time.time()
        self.pdf_path = pdf_path
        self.parameters = parameters
        self.sample_answers = sample_answers
        self.keywords = keywords
        self.total_points = total_points
        logger.info("Initializing SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        self.questions: List[Question] = []
        self.keyword_penalty = 0.1

    def extract_pdf_content(self) -> None:
        """Extract questions and answers from PDF using regex."""
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
                    sample_answers=self.sample_answers.get(q_id, []),
                    parameters=self.parameters.get(q_id, ""),
                    keywords=self.keywords.get(q_id, {}),
                    points=points
                ))
            logger.info(f"Extracted {len(self.questions)} questions in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def parse_parameters(self, params: str) -> Tuple[Dict[str, float], str]:
        """Parse grading parameters (e.g., 'completeness:0.5,similarity:0.5,leniency:lenient')."""
        if not params:
            return {"similarity": 1.0}, "not lenient"
        try:
            param_dict = {}
            leniency = "not lenient"
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
            logger.warning(f"Parameter parsing failed: {e}")
            return {"similarity": 1.0}, "not lenient"

    def is_synonym_present(self, word: str, answer_tokens: List[str]) -> bool:
        """Check if synonyms of a word are present in the answer."""
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower().replace('_', ' '))
            return any(token in synonyms for token in answer_tokens)
        except Exception as e:
            logger.warning(f"Synonym check failed for {word}: {e}")
            return False

    def check_keywords(self, answer: str, keywords: Dict[str, float], leniency: str) -> Tuple[float, List[str]]:
        """Check for keywords or their synonyms, applying weighted penalties."""
        if not keywords:
            return 1.0, []
        answer_lower = answer.lower()
        try:
            answer_tokens = word_tokenize(answer_lower)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}. Falling back to simple check.")
            answer_tokens = answer_lower.split()
        missing_keywords = []
        total_weight = sum(keywords.values())
        penalty_weight = 0.0
        for kw, weight in keywords.items():
            kw_lower = kw.lower()
            found = (kw_lower in answer_lower or
                     (len(kw_lower) >= 4 and any(kw_lower in word for word in answer_lower.split())) or
                     self.is_synonym_present(kw, answer_tokens))
            if not found:
                missing_keywords.append(kw)
                penalty_weight += weight
        # Stricter penalties: 0.5 for not lenient, 0.3 for lenient, 0.1 for very lenient
        penalty = {"very lenient": 0.1, "lenient": 0.3, "not lenient": 0.5}.get(leniency, 0.5)
        keyword_score = 1.0 - (penalty_weight / total_weight * penalty) if total_weight > 0 else 1.0
        # If all keywords are missing, score is near-zero
        if len(missing_keywords) == len(keywords):
            keyword_score = 0.1
        return max(0.0, keyword_score), missing_keywords

    def grammar_score(self, answer: str) -> float:
        """Calculate grammar score based on presence of verbs and nouns."""
        try:
            tokens = word_tokenize(answer)
            tags = pos_tag(tokens)
            has_verb = any(tag.startswith('VB') for _, tag in tags)
            has_noun = any(tag.startswith('NN') for _, tag in tags)
            return 1.0 if has_verb and has_noun else 0.6
        except Exception as e:
            logger.warning(f"Grammar scoring failed: {e}. Returning default score.")
            return 0.6

    def lexical_diversity(self, answer: str) -> float:
        """Measure vocabulary variety (unique words / total words)."""
        try:
            tokens = word_tokenize(answer.lower())
            if not tokens:
                return 0.0
            unique = set(tokens)
            return len(unique) / len(tokens)
        except Exception as e:
            logger.warning(f"Lexical diversity failed: {e}. Returning default score.")
            return 0.0

    def extract_named_entities(self, answer: str) -> List[str]:
        """Extract named entities (e.g., 'Python') from the answer."""
        try:
            chunks = ne_chunk(pos_tag(word_tokenize(answer)))
            return [" ".join(c[0] for c in tree) for tree in chunks if hasattr(tree, 'label')]
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {e}")
            return []

    def evaluate_answer(self, question: Question, custom_weights: Dict[str, float] = None) -> Tuple[float, List[str], str]:
        """Grade a single answer based on similarity, completeness, clarity, and keywords."""
        if not question.student_answer.strip() or not question.sample_answers:
            logger.debug(f"No valid answer or sample answers for {question.id}")
            return 0.0, list(question.keywords.keys()), "No valid answer or sample answers provided."
        start_time = time.time()
        similarity = 0.0
        try:
            student_embedding = self.model.encode([question.student_answer], convert_to_tensor=True)
            sample_embeddings = self.model.encode([sa for sa in question.sample_answers if sa.strip()], convert_to_tensor=True)
            similarities = [util.cos_sim(student_embedding[0], sample_embedding).item() for sample_embedding in sample_embeddings]
            similarity = max(similarities) if similarities else 0.0
            logger.debug(f"Similarity for {question.id}: {similarity:.2f} (max of {len(similarities)} sample answers)")
        except Exception as e:
            logger.warning(f"Similarity calculation failed for {question.id}: {e}. Defaulting to 0.0.")
            similarity = 0.0

        # Check language (expect English)
        language_penalty = 1.0
        try:
            lang = detect(question.student_answer)
            if lang != 'en':
                language_penalty = 0.1
                logger.debug(f"Non-English answer detected for {question.id}: language={lang}")
        except Exception as e:
            logger.warning(f"Language detection failed for {question.id}: {e}")

        params, leniency = self.parse_parameters(question.parameters)
        if custom_weights:
            for k in custom_weights:
                params[k] = custom_weights[k]
        similarity_boost = {"very lenient": 1.2, "lenient": 1.1, "not lenient": 1.0}.get(leniency, 1.0)
        # Cap low similarity to avoid rewarding vague overlaps
        if similarity < 0.3:
            similarity = 0.0
        adjusted_similarity = min(similarity * similarity_boost, 1.0)
        feedback = []

        score = adjusted_similarity * params.get("similarity", 1.0)

        completeness_relax = {"very lenient": 0.8, "lenient": 0.9, "not lenient": 1.0}.get(leniency, 1.0)
        if "completeness" in params:
            avg_sample_length = np.mean([len(sa) for sa in question.sample_answers if sa.strip()]) if question.sample_answers else 1
            length_ratio = len(question.student_answer) / avg_sample_length if avg_sample_length > 0 else 0.0
            completeness_score = min(1.0, length_ratio)
            # Reduce completeness for low-similarity answers
            if adjusted_similarity < 0.3:
                completeness_score = min(completeness_score, 0.2)
            adjusted_completeness = completeness_score if completeness_score >= 1.0 else completeness_score * completeness_relax
            score += adjusted_completeness * params.get("completeness", 0.0)
            if completeness_score < 0.8:
                feedback.append("Answer is less detailed than expected.")

        clarity_relax = {"very lenient": 0.8, "lenient": 0.9, "not lenient": 1.0}.get(leniency, 1.0)
        if "clarity" in params:
            grammar = self.grammar_score(question.student_answer)
            diversity = self.lexical_diversity(question.student_answer)
            clarity_score = (grammar + diversity) / 2
            adjusted_clarity = clarity_score if clarity_score >= 1.0 else clarity_score * clarity_relax
            score += adjusted_clarity * params.get("clarity", 0.0)
            if clarity_score < 0.8:
                feedback.append("Answer could improve in grammar or vocabulary variety.")

        if "accuracy" in params and adjusted_similarity < 0.5:
            score *= 0.9 * params.get("accuracy", 0.0) + (1 - params.get("accuracy", 0.0))
            feedback.append("Answer lacks accuracy compared to sample answers.")

        keyword_score, missing_keywords = self.check_keywords(question.student_answer, question.keywords, leniency)
        score *= keyword_score
        if missing_keywords:
            feedback.append(f"Missing key concepts: {', '.join(missing_keywords)}.")

        named_entities = self.extract_named_entities(question.student_answer)
        if named_entities:
            entity_boost = 0.05 * sum(1 for kw in question.keywords if kw.lower() in [ne.lower() for ne in named_entities])
            score = min(score + entity_boost, 1.0)
            if entity_boost > 0:
                feedback.append("Good use of relevant named entities.")

        # Apply language penalty
        score *= language_penalty
        if language_penalty < 1.0:
            feedback.append("Answer is not in English, which is required.")

        if not feedback:
            feedback.append("Well done, answer aligns well with sample answers.")

        logger.debug(f"Evaluated {question.id} in {time.time() - start_time:.2f} seconds")
        return max(0.0, min(score * question.points, question.points)), missing_keywords, ". ".join(feedback)

    def grade_assignment(self, custom_weights: Dict[str, float] = None) -> Tuple[float, Dict[str, Tuple[float, List[str], str]]]:
        """Grade all questions and compile results."""
        start_time = time.time()
        self.extract_pdf_content()
        if not self.questions:
            logger.error("No questions extracted")
            return 0.0, {}
        grading_start = time.time()
        results = [self.evaluate_answer(q, custom_weights) for q in self.questions]
        logger.info(f"Grading completed in {time.time() - grading_start:.2f} seconds")
        score_dict = {q.id: (score, missing_keywords, feedback) for q, (score, missing_keywords, feedback) in zip(self.questions, results)}
        total_score = sum(score for score, _, _ in results)
        logger.info(f"Total score: {total_score:.2f}/{self.total_points} in {time.time() - start_time:.2f} seconds")
        return round(total_score, 2), score_dict

    def generate_report(self, custom_weights: Dict[str, float] = None) -> str:
        """Generate a JSON report with grading details."""
        total_score, scores = self.grade_assignment(custom_weights)
        report = {
            "assignment_id": str(uuid.uuid4()),
            "total_points": self.total_points,
            "earned_points": total_score,
            "questions": [
                {
                    "id": q.id,
                    "question": q.text,
                    "student_answer": q.student_answer,
                    "sample_answers": q.sample_answers,
                    "parameters": q.parameters,
                    "keywords": list(q.keywords.keys()),
                    "missing_keywords": scores.get(q.id, (0.0, [], ""))[1],
                    "points_earned": scores.get(q.id, (0.0, [], ""))[0],
                    "points_possible": q.points,
                    "feedback": scores.get(q.id, (0.0, [], ""))[2],
                    "named_entities": self.extract_named_entities(q.student_answer)
                } for q in self.questions
            ]
        }
        return json.dumps(report, indent=2)

class ReinforcementLearner:
    def __init__(self, grader: OptimizedAssignmentGrader, learning_rate: float = 0.01):
        """Initialize RL to optimize grading weights."""
        self.grader = grader
        self.learning_rate = learning_rate
        self.weights = {
            "similarity": 0.6,
            "clarity": 0.2,
            "completeness": 0.2
        }

    def reward(self, predicted: float, actual: float) -> float:
        """Calculate reward based on prediction error."""
        return -abs(predicted - actual)

    def train(self, training_data: List[Tuple[str, str, List[str], float]], epochs: int = 10) -> Dict[str, float]:
        """Train weights using training data."""
        for epoch in range(epochs):
            total_reward = 0
            for question_text, student_answer, sample_answers, actual_score in training_data:
                q = Question(
                    id=str(uuid.uuid4()),
                    text=question_text,
                    student_answer=student_answer,
                    sample_answers=sample_answers,
                    parameters="",
                    keywords={},
                    points=1.0
                )
                predicted_score, _, _ = self.grader.evaluate_answer(q, self.weights)
                r = self.reward(predicted_score, actual_score)

                grad = predicted_score - actual_score
                for key in self.weights:
                    self.weights[key] -= self.learning_rate * grad
                    self.weights[key] = max(0.1, min(self.weights[key], 1.0))

                total_reward += r

            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for key in self.weights:
                    self.weights[key] /= total_weight

            logger.info(f"Epoch {epoch + 1}, Total Reward: {total_reward:.4f}, Weights: {self.weights}")

        return self.weights

def load_config(config_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, float]], float]:
    """Load grading configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        sample_answers = config.get("sample_answers", {})
        # Convert single strings to lists for backward compatibility
        for q_id in sample_answers:
            if isinstance(sample_answers[q_id], str):
                sample_answers[q_id] = [sample_answers[q_id]]
        return (
            config.get("parameters", {}),
            sample_answers,
            config.get("keywords", {}),
            config.get("total_points", 20.0)
        )
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        raise

def load_trainer_data(trainer_path: str) -> List[Tuple[str, str, List[str], float]]:
    """Load training data from trainer.json."""
    try:
        with open(trainer_path, 'r') as f:
            data = json.load(f)
        training_data = []
        for item in data:
            sample_answers = item.get("sample_answers", [item["sample_answer"]] if "sample_answer" in item else [])
            if not sample_answers or not item.get("question_text") or not item.get("student_answer") or "expected_score" not in item:
                logger.warning(f"Skipping invalid training item: {item}")
                continue
            training_data.append((
                item["question_text"],
                item["student_answer"],
                sample_answers,
                float(item["expected_score"])
            ))
        if not training_data:
            raise ValueError("No valid training data loaded from trainer.json")
        logger.info(f"Loaded {len(training_data)} training examples from {trainer_path}")
        return training_data
    except Exception as e:
        logger.warning(f"Failed to load trainer.json: {e}. Using minimal default training data.")
        return [
            (
                "What is Python?",
                "Python is a popular programming language used in AI and web development.",
                [
                    "Python is a high-level, versatile programming language used for web development, data science, and automation.",
                    "Python is a widely-used programming language known for its simplicity and applications in AI."
                ],
                0.95
            ),
            (
                "Explain what a loop is in programming.",
                "A loop repeats code while a condition is true.",
                [
                    "A loop is a programming construct that repeats a block of code until a condition is met.",
                    "A loop allows code to be executed repeatedly based on a condition."
                ],
                0.90
            )
        ]

if __name__ == "__main__":
    start_time = time.time()
    pdf_path = "test2.pdf"
    config_path = "grader_config.json"
    trainer_path = "trainer.json"
    parameters, sample_answers, keywords, total_points = load_config(config_path)
    grader = OptimizedAssignmentGrader(pdf_path, parameters, sample_answers, keywords, total_points)

    # Load and train RL
    training_data = load_trainer_data(trainer_path)
    rl = ReinforcementLearner(grader, learning_rate=0.01)
    learned_weights = rl.train(training_data, epochs=10)
    logger.info(f"Applying learned weights: {learned_weights}")

    # Grade with learned weights
    report = grader.generate_report(learned_weights)
    with open("grading_report.json", "w") as f:
        f.write(report)
    print(json.dumps(json.loads(report), indent=2))
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")