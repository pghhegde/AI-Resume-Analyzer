# backend/app.py
import os
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_skills_from_text,
    compute_similarity_and_matches
)

app = Flask(__name__)
CORS(app)


REMOTIVE_API = "https://remotive.com/api/remote-jobs"


@app.route("/jobs", methods=["GET"])
def get_jobs():
    """
    Returns a list of job postings from Remotive (free public jobs API).
    """
    try:
        resp = requests.get(REMOTIVE_API, timeout=10)
        resp.raise_for_status()  # Check for HTTP errors
        data = resp.json()
        jobs = []
        for job in data.get("jobs", [])[:60]:  # limit to first 60 results
            jobs.append({
                "id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company_name"),
                "location": job.get("candidate_required_location"),
                "description": job.get("description", "")
            })
        return jsonify(jobs)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch jobs: {e}")
        return jsonify({"error": "Failed to fetch jobs from the API", "details": str(e)}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    """
    Analyzes a resume against a given job description.
    Accepts a file upload for the resume and job_description via form data.
    """
    file = request.files.get("resume")
    jd_text = request.form.get("job_description", "")

    if not file:
        return jsonify({"error": "No resume uploaded"}), 400
    
    if not jd_text:
        return jsonify({"error": "No job description provided"}), 400

    # Save uploaded file temporarily
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    file.save(tmp_path)

    try:
        resume_text = ""
        if file.filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(tmp_path)
        elif file.filename.lower().endswith(".docx"):
            resume_text = extract_text_from_docx(tmp_path)
        else:
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()

        score, matched_keywords = compute_similarity_and_matches(resume_text, jd_text)
        skills = extract_skills_from_text(resume_text)

        result = {
            "skills": skills,
            "matched_keywords": matched_keywords,
            "score": score,  # Use raw score for cleaner frontend logic
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Failed to analyze resume", "details": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)