"""Hand-written test documents. They are synthetic and contain no real
person's data. They exercise documented scoring rules; they do NOT
establish that the rubric agrees with human recruiters."""

from datetime import date

from src.data.resume_parser import parse_resume
from src.scoring.requirements import extract_job_requirements

REFERENCE_DATE = date(2026, 1, 1)

JOB_DATA_ANALYST = """Data Analyst

About the role:
You will analyse sales data and build dashboards for the commercial team.

Requirements:
- SQL
- Python
- Power BI or Tableau
- Statistics
- 2+ years of experience in an analytics role
- Bachelor's degree in Statistics, Computer Science, or a related field

Preferred:
- AWS
- Docker
"""

RESUME_STRONG = """SUMMARY
Data analyst with a focus on retail sales analytics.

SKILLS
SQL, Python, Tableau, Statistics, Excel

EXPERIENCE
Data Analyst, Contoso Retail (Jan 2022 - Dec 2025)
- Built Tableau dashboards used by 40 regional managers.
- Wrote SQL and Python pipelines to clean daily sales data.
- Applied statistics to evaluate promotion uplift.

INTERNSHIPS
Analytics Intern, Fabrikam (Jun 2021 - Dec 2021)
- Automated weekly Excel reports with Python.

PROJECTS
- Deployed a demand-forecasting prototype on AWS using Docker.

EDUCATION
B.S. in Statistics, Riverside University, 2021
"""

RESUME_PARTIAL = """SUMMARY
Junior analyst moving into data work.

SKILLS
Excel, SQL, Power BI

EXPERIENCE
Operations Assistant, Northwind (Mar 2024 - Dec 2025)
- Prepared SQL queries for monthly inventory reports.
- Maintained the stock spreadsheet and reconciled supplier invoices each week.

EDUCATION
B.A. in Economics, Lakeside College, 2023
"""

RESUME_WEAK = """SUMMARY
Chef with a passion for seasonal menus.

SKILLS
Menu planning, food safety, kitchen management

EXPERIENCE
Sous Chef, The Olive Tree (2018 - 2025)
- Managed a team of 6 cooks and planned weekly menus.
- Ran daily stock checks, supplier orders and hygiene inspections.
- Trained new kitchen staff on knife skills and plating standards.

EDUCATION
Diploma in Culinary Arts, City Catering School, 2017
"""


def job(text: str = JOB_DATA_ANALYST) -> dict:
    extracted = extract_job_requirements(text)
    return {"title": extracted["title"], "text": text, "requirements": extracted["requirements"]}


def parsed(text: str) -> dict:
    return parse_resume(text)


def fake_semantic(resume_text: str, job_text: str) -> float:
    """Deterministic stand-in for the sentence-transformer: word-set overlap."""
    a, b = set(resume_text.lower().split()), set(job_text.lower().split())
    return len(a & b) / max(1, len(a | b))
