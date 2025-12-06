import os
import json
from extractor import OncologyExtractor

# Configuration
PDF_FOLDER = "papers"  # Create this folder and put your PDFs inside
OUTPUT_FILE = "extracted_oncology_data.json"

def main():
    # 1. Setup
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created folder '{PDF_FOLDER}'. Please add your PDF files there and run again.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{PDF_FOLDER}'.")
        return

    print(f"Found {len(pdf_files)} papers to process.")

    # 2. Initialize Extractor
    # Make sure you have set GOOGLE_API_KEY environment variable
    try:
        extractor = OncologyExtractor()
    except Exception as e:
        print("Error initializing extractor. Did you set GOOGLE_API_KEY?")
        return

    # 3. Process Batch
    results = []

    for filename in pdf_files:
        path = os.path.join(PDF_FOLDER, filename)
        try:
            study_data = extractor.analyze_paper(path)
            if study_data:
                results.append(study_data.model_dump())
                print(f"✅ Success: {filename} -> Reliability: {study_data.predicted_reliability}")
        except Exception as e:
            print(f"❌ Error processing {filename}")

    # 4. Save Results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nExtraction complete. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()