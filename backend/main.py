from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from agents import jd_analyzer_agent, resume_screener_agent, ranking_agent, email_agent
import PyPDF2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/analyze")
async def analyze(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...)
):
    try:
        # Agent 1 — JD Analyzer
        jd_analysis = jd_analyzer_agent(job_description)

        # Agent 2 — Resume Screener (runs for each resume)
        screenings = []
        candidate_names = []
        for resume in resumes:
            file_bytes = await resume.read()
            if resume.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_bytes)
            else:
                resume_text = file_bytes.decode("utf-8")
            
            screening = resume_screener_agent(resume_text, jd_analysis)
            screenings.append(screening)
            candidate_names.append(resume.filename.replace(".pdf","").replace(".txt",""))

        # Agent 3 — Ranking Agent
        rankings = ranking_agent(screenings)

        # Agent 4 — Email Agent (for top candidate)
        top_candidate = candidate_names[0] if candidate_names else "Candidate"
        selected_email = email_agent(top_candidate, "selected", "the applied position")
        rejected_email = email_agent(top_candidate, "rejected", "the applied position")

        return {
            "jd_analysis": jd_analysis,
            "screenings": screenings,
            "rankings": rankings,
            "selected_email": selected_email,
            "rejected_email": rejected_email,
            "candidate_names": candidate_names
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "RecruitIQ API is running!"}
