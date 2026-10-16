# AI Assistance Disclosure

**Course:** SWE 3513 – Artificial Intelligence  
**Assessment:** Individual Assignment 2 – Musanze Harvest Intelligence Service  
**Student:** NZAYISENGA Erneste  
**Registration number:** 25/27341  
**Institutional email:** ug2527341@ines.ac.rw  
**Verification code:** AI-A2-7341

## Purpose of AI assistance

I used AI-assisted tools as support while developing and documenting this individual assignment. The assistance covered understanding the assignment instructions and terminology, planning the project structure, drafting and refining Python code, identifying implementation issues, suggesting testing commands, and preparing initial documentation.

## Tools and activities

- **ChatGPT:** Helped me understand the assignment requirements; explained machine learning terminology; suggested an implementation plan, terminal commands, testing procedures, and report structure; and assisted with drafting documentation.
- **AI coding assistant in Visual Studio Code:** Assisted with writing and refactoring Python source code, configuring the reproducible machine-learning pipeline, CLI inference and FastAPI service, and creating automated tests. It also executed commands and reported results during development.

I provided project-specific details and requirements, including my registration number and institutional username. The project uses the required personal random seed (7341) and verification code (AI-A2-7341).

## Testing and verification

I personally ran commands in the VS Code PowerShell terminal and observed the following:

- `python -m pytest -q`: six tests passed during an earlier run.
- The dataset pipeline ran with `--student-id ug2527341@ines.ac.rw --seed 7341` and printed the expected dataset fingerprint.
- The prediction CLI returned a valid result and rejected a sample with missing fields with exit code 1.
- Manual PowerShell requests to `/health`, `/model-info` and `/predict` returned valid responses; an out-of-range rainfall input was rejected with HTTP 422.

Following code changes, the AI coding assistant reported that eight automated tests passed and the model artifacts were regenerated. These later results should be checked against the final local project before submission.

## Responsibility and limitations

I am responsible for reviewing the submitted code and documents, verifying results, explaining the model choices, and completing the required individual demonstration. AI-generated suggestions and code may contain mistakes, so they do not replace testing or personal understanding.

The project dataset is synthetic. Its predictions are for classroom decision-support demonstration only and should not be treated as validated guidance for real farms or used to make decisions about loans, insurance or public services without expert review.

## Declaration

This statement describes the AI assistance used in preparing the project. I will ensure that the submitted materials and demonstration accurately represent the work completed and the results verified.
