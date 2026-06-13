from crewai import Task
from core.engine import cooldown_callback
from core.agents import (
    screener_agent, evaluator_agent, decision_agent,
    interview_agent, scheduler_agent, comms_agent,
)


def build_pipeline(
    resume_text: str,
    job_desc: str,
    company: str,
    department: str,
    scheduling_url: str,
) -> list:
    """
    Builds a 6-task sequential recruitment pipeline.
    Each task declares context= explicitly so CrewAI injects prior outputs
    regardless of version (avoids the implicit context bug in crewai <0.80).
    """

    # ── Task 1: Resume Screening ──────────────────────────────────────────────
    t1 = Task(
        description=(
            f"Parse the resume below. Extract every field precisely — do NOT guess or invent.\n\n"
            f"Extract:\n"
            f"1. Full candidate name\n"
            f"2. Email and phone (write 'Not provided' if absent)\n"
            f"3. Total years of professional experience (calculate from dates)\n"
            f"4. All technical skills, frameworks, languages, and tools\n"
            f"5. Education: degree, institution, graduation year\n"
            f"6. Last 3 roles: job title | company | duration | 1-sentence summary of responsibilities\n\n"
            f"RESUME:\n"
            f"{'='*60}\n"
            f"{resume_text}\n"
            f"{'='*60}"
        ),
        expected_output=(
            "### Resume Screening Summary\n"
            "**Candidate Name:** [Name]\n"
            "**Contact:** [Email] | [Phone or 'Not provided']\n"
            "**Total Experience:** [X years]\n"
            "**Technical Skills:** [comma-separated list]\n"
            "**Education:** [Degree, Institution, Year]\n\n"
            "**Work History:**\n"
            "| Title | Company | Duration | Key Responsibility |\n"
            "|---|---|---|---|\n"
            "| ... | ... | ... | ... |\n"
        ),
        agent=screener_agent,
        callback=cooldown_callback,
    )

    # ── Task 2: Technical Evaluation ─────────────────────────────────────────
# ── Task 2: Technical Evaluation (Updated for Custom Shortform) ─────────
    t2 = Task(
        description=(
            f"You have the candidate's resume summary from Task 1.\n"
            f"Now evaluate them against this job description:\n\n"
            f"JOB DESCRIPTION:\n"
            f"{'='*60}\n"
            f"{job_desc}\n"
            f"{'='*60}\n\n"
            f"NOTE ON JD FORMAT:\n"
            f"- The user may list required skills using custom prefixes like 'skills-', 'skills:', 'skilli are', or 'requirements:'.\n"
            f"- Parse whatever follows these custom markers as the core required skills for evaluation.\n\n"
            f"Produce ALL of the following:\n"
            f"1. Skill match table: for every required skill in the JD, show if the candidate has it\n"
            f"2. Overall Match Score 0-100. Format exactly as: Match Score: [number]\n"
            f"3. Top 3 strengths (with evidence from the resume)\n"
            f"4. Top 3 skill gaps (specific missing skills, not vague statements)\n"
        ),
        expected_output=(
            "### Technical Evaluation\n\n"
            "Match Score: [0-100]\n\n"
            "| Required Skill | Found in Resume | Evidence / Signal |\n"
            "|---|---|---|\n"
            "| [skill] | Yes / No / Partial | [1-line note] |\n\n"
            "**Top 3 Strengths:**\n"
            "1. ...\n2. ...\n3. ...\n\n"
            "**Top 3 Skill Gaps:**\n"
            "1. ...\n2. ...\n3. ...\n"
        ),
        agent=evaluator_agent,
        context=[t1],
        callback=cooldown_callback,
    )

    # ── Task 3: Hiring Decision ───────────────────────────────────────────────
    t3 = Task(
        description=(
            f"You have the resume summary (Task 1) and the technical evaluation (Task 2).\n"
            f"Produce a final hiring recommendation for the {department} role at {company}.\n\n"
            f"RULES:\n"
            f"- Your verdict label must appear EXACTLY as one of these three strings:\n"
            f"  [STRONG HIRE]  |  [CONDITIONAL REVIEW]  |  [REJECT]\n"
            f"- Use [STRONG HIRE] if Match Score >= 80 and no critical gaps\n"
            f"- Use [CONDITIONAL REVIEW] if Match Score 55-79 or minor gaps exist\n"
            f"- Use [REJECT] if Match Score < 55 or critical required skills are completely missing\n"
            f"- Write a 2-3 sentence justification referencing specific evidence. Be direct."
        ),
        expected_output=(
            "### Hiring Recommendation\n"
            "**Verdict:** [STRONG HIRE | CONDITIONAL REVIEW | REJECT]\n\n"
            "**Justification:** [2-3 sentences citing specific evidence from Tasks 1 and 2]\n"
        ),
        agent=decision_agent,
        context=[t1, t2],
        callback=cooldown_callback,
    )

    # ── Task 4: Interview Question Set ────────────────────────────────────────
    t4 = Task(
        description=(
            f"You have the resume summary (Task 1), skill evaluation (Task 2), "
            f"and hiring verdict (Task 3).\n\n"
            f"Check the verdict from Task 3:\n"
            f"- If the verdict contains [REJECT], write only this line and nothing else:\n"
            f"  'Interview questions not generated — candidate did not progress.'\n\n"
            f"- Otherwise, generate a structured question set for the {department} role at {company}:\n"
            f"  * 3 Foundational questions — test skills the candidate claimed to have\n"
            f"  * 3 Applied scenario questions — target their specific skill gaps from Task 2\n"
            f"  * 2 System design questions — appropriate for their seniority level\n"
            f"  * 1 Behavioural question — relevant to their work history\n\n"
            f"  After each question add: *What a strong answer looks like: [1 sentence]*"
        ),
        expected_output=(
            "### Interview Question Set\n\n"
            "**Foundational (testing claimed skills)**\n"
            "1. [Question]\n   *Strong answer: ...*\n"
            "2. [Question]\n   *Strong answer: ...*\n"
            "3. [Question]\n   *Strong answer: ...*\n\n"
            "**Applied Scenarios (targeting skill gaps)**\n"
            "4. [Question]\n   *Strong answer: ...*\n"
            "5. [Question]\n   *Strong answer: ...*\n"
            "6. [Question]\n   *Strong answer: ...*\n\n"
            "**System Design**\n"
            "7. [Question]\n   *Strong answer: ...*\n"
            "8. [Question]\n   *Strong answer: ...*\n\n"
            "**Behavioural**\n"
            "9. [Question]\n   *Strong answer: ...*\n"
        ),
        agent=interview_agent,
        context=[t1, t2, t3],
        callback=cooldown_callback,
    )

    # ── Task 5: Scheduling Plan ───────────────────────────────────────────────
    t5 = Task(
        description=(
            f"You have the hiring verdict from Task 3 and the candidate name from Task 1.\n\n"
            f"If the verdict from Task 3 contains [REJECT]:\n"
            f"  Set Status to DEACTIVATED. Do NOT include the booking URL. "
            f"Write a one-line closure note only.\n\n"
            f"If the verdict is [STRONG HIRE] or [CONDITIONAL REVIEW]:\n"
            f"  Set Status to ACTIVE.\n"
            f"  Use the candidate's actual name (do not write 'Candidate').\n"
            f"  Panel: {department} team at {company}\n"
            f"  Format: 45-minute live technical interview\n"
            f"  Booking link (copy verbatim): {scheduling_url}\n"
            f"  Agenda: Introduction (5 min) → Technical Questions (30 min) → Q&A (10 min)\n"
        ),
        expected_output=(
            "### Interview Scheduling Plan\n"
            "**Status:** [ACTIVE | DEACTIVATED]\n"
            "**Candidate:** [Name]\n"
            "**Panel:** [Department] team at [Company]\n"
            "**Format:** 45-minute live technical interview\n"
            "**Booking Link:** [URL | N/A]\n"
            "**Agenda:** Introduction (5 min) → Technical Questions (30 min) → Q&A (10 min)\n"
        ),
        agent=scheduler_agent,
        context=[t1, t3],
        callback=cooldown_callback,
    )

    # ── Task 6: Outreach Email ────────────────────────────────────────────────
    t6 = Task(
        description=(
            f"Draft the outreach email to the candidate on behalf of {company}'s {department} team.\n\n"
            f"Use the candidate's real name from Task 1. NEVER write 'Dear Candidate'.\n\n"
            f"Tone rules based on Task 3 verdict:\n"
            f"  [STRONG HIRE]       → Warm, enthusiastic, clear next steps with booking link\n"
            f"  [CONDITIONAL REVIEW] → Positive but measured, mention next steps with booking link\n"
            f"  [REJECT]            → Respectful, brief, no booking link, encourage future applications\n\n"
            f"Booking link (only for non-rejected): {scheduling_url}\n\n"
            f"Style rules:\n"
            f"  - No labels like 'Role:' or 'Status:' in the email body\n"
            f"  - Flow naturally: greeting → why you're writing → what happens next → sign-off\n"
            f"  - 2-3 paragraphs max\n"
            f"  - Sign off as: The {department} Talent Team, {company}\n"
        ),
        expected_output=(
            "### Candidate Outreach Email\n\n"
            "**Subject:** [Specific, relevant subject line — not generic]\n\n"
            "Dear [Candidate Name],\n\n"
            "[Paragraph 1: context / reason for writing]\n\n"
            "[Paragraph 2: specifics — next steps or closure]\n\n"
            "[Paragraph 3 (optional): warm closing]\n\n"
            "Best regards,\n"
            f"The {department} Talent Team\n"
            f"{company}\n"
        ),
        agent=comms_agent,
        context=[t1, t3, t5],
        callback=cooldown_callback,
    )

    return [t1, t2, t3, t4, t5, t6]