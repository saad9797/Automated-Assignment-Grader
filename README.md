**📚 AI Assignment Grader**

Welcome to the AI Assignment Grader! This Python tool automatically grades student assignments from PDF files, comparing answers to ideal responses using AI. It checks for key concepts, language (English only), and clarity, saving teachers time while providing fair, detailed feedback. Perfect for programming quizzes!

✨ What It Does 

Reads questions and answers from a PDF (e.g., demo3.pdf).
Scores answers out of 10 based on:
Similarity to sample answers (using AI).
Presence of keywords (e.g., "class", "loop").
English language (non-English answers score near 0).
Grammar and detail.


Generates a grading_report.json with scores and feedback (e.g., "Missing key concepts: method").
Learns to improve scoring using a training file (trainer3.json).


🛠️ Requirements 

Operating System: Windows
Python: Version 3.12
Libraries:
pdfplumber
sentence-transformers
numpy
nltk
langdetect


**Files:**

demo3.pdf: Assignment PDF with format Q1: Question? Ans: Answer...
demo3.json: Grading rules (sample answers, keywords).
trainer3.json: Training data for scoring.




**🚀 Setup**

Install Python 3.12:

Download from python.org.
Ensure pip is installed.


Install Libraries:

Open Command Prompt and run:pip install pdfplumber sentence-transformers numpy nltk langdetect




Download NLTK Resources:

Run this Python code to download required data:import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words'], download_dir='C:\\Users\\saad4\\AppData\\Roaming\\nltk_data')




Prepare Files:

Place the following in C:\Users\saad4\Desktop\AI Assignment:
assignment_grader.py (the main script).
demo3.pdf (assignment PDF).
demo3.json (grading configuration).
trainer3.json (training data).






**▶️ How to Run**

Open Command Prompt:

Press Win + R, type cmd, and press Enter.


Navigate to Project Folder:
cd C:\Users\saad4\Desktop\AI Assignment


Run the Grader:
python assignment_grader.py


This reads demo3.pdf, grades answers using demo3.json and trainer3.json, and creates grading_report.json.


Check Output:

Open grading_report.json in a text editor (e.g., Notepad).
Look for:
earned_points: Total score (e.g., 15.5/50.0).
Per question:
points_earned: Score out of 10 (e.g., 9.5 for correct, 0.1 for non-English).
feedback: Comments like "Well done" or "Answer is not in English".
missing_keywords: Keywords not found in the answer.








**📝 Example Files**

demo3.pdf:
Q1: What is polymorphism in programming?
Ans: Polymorphism is like a chameleon changing colors...
Q2: Explain exception handling.
Ans: Un manejo de excepciones es...


Correct answers score ~9–10/10; irrelevant or non-English answers score ~0–1/10.


demo3.json:
{
    "parameters": {"Q1": "similarity:0.7,completeness:0.3,leniency:not lenient"},
    "sample_answers": {"Q1": ["Polymorphism allows method overriding..."]},
    "keywords": {"Q1": {"polymorphism": 2.0, "method": 1.0}},
    "total_points": 50.0
}


trainer3.json:
[
    {
        "question_text": "What is polymorphism?",
        "student_answer": "Polymorphism lets classes share methods...",
        "sample_answers": ["Polymorphism allows method overriding..."],
        "expected_score": 0.9
    }
]




**🛡️ Troubleshooting**

Error: "PDF extraction failed":

Ensure demo3.pdf exists and follows the format Q1: ... Ans: ....
Test with a simple PDF:Q1: Test?
Ans: Test answer.




Error: "No questions extracted":

Check demo3.pdf for correct formatting.
Ensure questions start with Q1:, Q2:, etc.


Non-English Answers Not Penalized:

Verify langdetect:from langdetect import detect
print(detect("Un manejo de excepciones..."))  # Should print 'es'


Reinstall: pip install langdetect.


NLTK Errors:

Redownload resources:import nltk
nltk.download('maxent_ne_chunker_tab', download_dir='C:\\Users\\saad4\\AppData\\Roaming\\nltk_data')


Clear corrupted data:rmdir /s C:\Users\saad4\AppData\Roaming\nltk_data




Scores Too High for Irrelevant Answers:

Edit assignment_grader.py, in evaluate_answer:if similarity < 0.2:  # Change from 0.3 to 0.2
    similarity = 0.0


Rerun the grader.




📌 **Notes**

Output: grading_report.json shows scores and feedback for each question.
Customization: Edit demo3.json to add questions, keywords, or change scoring rules.
Help: If stuck, check logs in Command Prompt or share grading_report.json.

Happy grading! This tool makes marking assignments fast and fair. Run it and check the results in grading_report.json! 🎉
