import os
from openai import OpenAI
from dotenv import load_dotenv
import pathlib

dotenv_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

token = os.environ.get("GITHUB_TOKEN", "")
model = "gpt-4o"

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

def jd_analyzer_agent(job_description: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are the JD Analyzer Agent. 
                Analyze the job description and extract:
                1. Required skills (list)
                2. Experience needed
                3. Key responsibilities
                4. Must-have qualifications
                Return as a clean structured text."""
            },
            {
                "role": "user",
                "content": f"Analyze this job description:\n{job_description}"
            }
        ]
    )
    return response.choices[0].message.content

def resume_screener_agent(resume_text: str, jd_analysis: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are the Resume Screener Agent.
                Compare the resume against the job requirements and:
                1. List matching skills
                2. List missing skills
                3. Give a match score out of 100
                4. Write 2 line candidate summary
                Return as clean structured text."""
            },
            {
                "role": "user",
                "content": f"Job Requirements:\n{jd_analysis}\n\nResume:\n{resume_text}"
            }
        ]
    )
    return response.choices[0].message.content

def ranking_agent(all_screenings: list) -> str:
    screenings_text = "\n\n---\n\n".join(
        [f"Candidate {i+1}:\n{s}" for i, s in enumerate(all_screenings)]
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are the Ranking Agent.
                Rank all candidates from best to worst fit.
                For each candidate give:
                1. Rank number
                2. Name if available
                3. Score
                4. One line reason
                Return as a clean numbered list."""
            },
            {
                "role": "user",
                "content": f"Rank these candidates:\n{screenings_text}"
            }
        ]
    )
    return response.choices[0].message.content

def email_agent(candidate_name: str, status: str, job_title: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are the Email Agent.
                Write a professional HR email.
                If selected: warm interview invitation email.
                If rejected: polite rejection email.
                Keep it short, professional and friendly."""
            },
            {
                "role": "user",
                "content": f"Write a {status} email for {candidate_name} for the {job_title} position."
            }
        ]
    )
    return response.choices[0].message.content
