import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pypdf import PdfReader


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """
    Extracts plain text from an uploaded PDF file object.
    Raises ValueError if the PDF contains no extractable text
    (e.g. scanned image PDFs need OCR — not supported here).
    """
    text = ""
    try:
        reader = PdfReader(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    if not text.strip():
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned image — please upload a text-based (digital) PDF."
        )
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_score(task_raw: str) -> int:
    """
    Extracts the numeric match score from Task 2 output.
    Looks for 'Match Score: 75' or 'Match Score:75' anywhere in the text.
    Returns 70 as a safe default if not found.
    """
    if not task_raw:
        return 70
    match = re.search(r"Match\s+Score\s*:\s*(\d{1,3})", task_raw, re.IGNORECASE)
    return min(int(match.group(1)), 100) if match else 70


def parse_verdict(task_raw: str) -> str:
    """
    Extracts the verdict from Task 3 output.
    Checks for the three possible labels in priority order.
    Returns 'CONDITIONAL REVIEW' if none found to avoid 'UNKNOWN'.
    """
    if not task_raw:
        return "CONDITIONAL REVIEW"
    upper = task_raw.upper()
    if "STRONG HIRE" in upper:
        return "STRONG HIRE"
    if "WEAK" in upper or "REJECT" in upper:
        return "WEAK"
    if "CONDITIONAL REVIEW" in upper:
        return "CONDITIONAL REVIEW"
    return "CONDITIONAL REVIEW"


def split_task_outputs(raw_report: str) -> list[str]:
    """
    Attempts to split the full crew raw output into per-task chunks.
    Upgraded to use resilient fuzzy matching regex boundaries so it 
    never crashes if the LLM modifies the header punctuation strings.
    """
    if not raw_report:
        return [""] * 6

    # Fuzzy regex tags matching your agent layout definitions
    patterns = [
        r"(?:###?\s*)?(?:👤\s*)?Resume\s+Screening\s+Summary",
        r"(?:###?\s*)?(?:📊\s*)?Technical\s+Evaluation",
        r"(?:###?\s*)?(?:⚖️\s*)?Hiring\s+Recommendation",
        r"(?:###?\s*)?(?:❓\s*)?Interview\s+Question\s+Set",
        r"(?:###?\s*)?(?:📅\s*)?Interview\s+Scheduling\s+Plan",
        r"(?:###?\s*)?(?:✉️\s*)?Candidate\s+Outreach\s+Email"
    ]
    
    sections = []
    positions = []
    
    # Track starting index locations for each agent report section
    for p in patterns:
        match = re.search(p, raw_report, re.IGNORECASE)
        positions.append(match.start() if match else -1)
        
    for i in range(len(positions)):
        start = positions[i]
        if start == -1:
            sections.append("")
            continue
            
        # Isolate text up until the next section boundary header begins
        end = len(raw_report)
        for next_start in positions[i + 1:]:
            if next_start != -1:
                end = next_start
                break
                
        sections.append(raw_report[start:end].strip())
        
    # Safe structural block line-split fallback if matching strings are completely missing
    if not any(sections):
        chunks = raw_report.split("\n\n")
        while len(chunks) < 6:
            chunks.append("")
        return chunks[:6]
        
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL SENDING
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """
    Converts basic markdown to plain text suitable for an email body.
    Removes **bold**, *italic*, ### headers, and | table rows.
    """
    # Remove markdown headers
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    # Remove table rows (lines starting with |)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    # Remove table dividers (|---|---|)
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def send_email(
    sender: str,
    app_password: str,
    recipient: str,
    subject: str,
    email_task_output: str,
) -> None:
    """
    Sends the Task 6 email draft to the candidate via Gmail SMTP.
    Strips markdown and internal pipeline headers before sending.
    Connection is always closed via context manager, even on error.
    """
    msg = MIMEMultipart("alternative")
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = subject

    # 🟢 FIXED: Case-insensitive regex split that handles markdown hashes, 
    # emoji variations, and trailing text safely so it never returns an empty slice.
    email_match = re.split(r"(?:###?\s*)?(?:✉️\s*)?Candidate\s+Outreach\s+Email.*", email_task_output, flags=re.IGNORECASE)
    if len(email_match) > 1:
        body_raw = email_match[-1]
    else:
        body_raw = email_task_output

    # Remove the Subject line from body (it's already in the email header)
    body_raw = re.sub(r"^\*\*Subject:\*\*.*$", "", body_raw, flags=re.MULTILINE)

    # Convert markdown to plain text
    body_plain = _strip_markdown(body_raw)

    msg.attach(MIMEText(body_plain, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())