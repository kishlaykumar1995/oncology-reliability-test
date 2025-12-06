import os
import json
import google.generativeai as genai
from PyPDF2 import PdfReader
from models import OncologyStudy

# Configure API Key (Best practice: set GOOGLE_API_KEY in your environment variables)
# If not set, you can hardcode it here for testing: genai.configure(api_key="YOUR_KEY")
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class OncologyExtractor:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the extractor with a specific Gemini model.
        gemini-2.0-flash is recommended for speed and cost-efficiency with long context.
        """
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 10) -> str:
        """
        Extracts text from a PDF file.
        Limits to max_pages to avoid token limits on very large appendices.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            # Extract text from the first N pages (usually contains Abstract, Methods, Results)
            count = min(len(reader.pages), max_pages)
            for i in range(count):
                page = reader.pages[i]
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def analyze_paper(self, pdf_path: str) -> OncologyStudy:
        """
        Main method to process a PDF and return a structured OncologyStudy object.
        """
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}...")

        # 1. Get Raw Text
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            raise ValueError(f"No text extracted from {filename}")

        # 2. Construct Prompt
        # We inject the JSON schema implicitly by asking for the specific structure
        # that matches our Pydantic model.
        prompt = f"""
        You are an expert Oncology Research Assistant. Your task is to extract structured clinical data from the following research paper text.

        Return the result primarily as a JSON object that strictly follows this structure:

        {{
            "filename": "{filename}",
            "title": "Exact title of the paper",
            "publication_year": 2024,
            "population": {{
                "cancer_type": "...",
                "disease_stage": "...",
                "patient_age_range": "...",
                "sex_distribution": "...",
                "comorbidities_summary": "...",
                "prior_treatments_summary": "..."
            }},
            "design": {{
                "study_type": "One of: RCT, Prospective Cohort, Retrospective, Case Series, Meta-Analysis, Other",
                "sample_size": 100,
                "intervention": "...",
                "comparator": "...",
                "primary_endpoint": "...",
                "secondary_endpoints": ["..."],
                "followup_duration": "..."
            }},
            "results": {{
                "primary_result_text": "...",
                "primary_result_numeric": "...",
                "safety_profile_summary": "...",
                "limitations_summary": "..."
            }}
        }}

        Ensure 'sample_size' is an integer. If a value is not found, use null.

        --- BEGIN PAPER TEXT ---
        {full_text[:30000]}
        --- END PAPER TEXT ---
        """
        # Note: We truncate text to ~30k chars (~7-8k tokens) to be safe,
        # though Gemini 1.5/2.0 can handle much more.

        # 3. Call LLM
        try:
            response = self.model.generate_content(prompt)

            # 4. Parse & Validate JSON
            # The response.text should be a JSON string thanks to response_mime_type config
            data = json.loads(response.text)

            # 5. Pydantic Validation
            study_data = OncologyStudy(**data)

            # 6. Apply Reliability Heuristic (Rule-based post-processing)
            study_data.predicted_reliability = self._calculate_reliability(study_data)

            return study_data

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            return None

    def _calculate_reliability(self, study: OncologyStudy) -> str:
        """
        Simple rule-based heuristic to assign Low/Medium/High reliability.
        """
        score = 0

        # 1. Design Weight
        if study.design.study_type == 'RCT':
            score += 3
        elif study.design.study_type == 'Prospective Cohort':
            score += 2
        elif study.design.study_type == 'Retrospective':
            score += 1

        # 2. Sample Size Weight
        if study.design.sample_size > 500:
            score += 2
        elif study.design.sample_size > 100:
            score += 1

        # 3. Comparator Weight
        if study.design.comparator and study.design.comparator.lower() != 'none':
            score += 1

        # Categorize
        if score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"