from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_pdf(student, model_name, prediction, probability, readiness, readiness_text,
              careers, gaps, actions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=16)
    h = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=7)
    normal = styles["BodyText"]
    story = []

    story.append(Paragraph("CareerCompass AI", title))
    story.append(Paragraph("Student Success & Placement Intelligence Report", h))
    story.append(Paragraph(
        "This report is a decision-support output from the CareerCompass AI prototype. "
        "If the application is using the supplied demo dataset, the ML result is illustrative "
        "and should not be treated as a validated institutional placement probability.", normal
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Placement Assessment", h))
    assessment = [
        ["Best Model", model_name],
        ["Placement Prediction", "Likely to be placed" if prediction == 1 else "Needs improvement"],
        ["Model Probability", f"{probability:.2f}%"],
        ["Readiness Score", f"{readiness:.2f}%"],
        ["Readiness Level", readiness_text],
    ]
    t = Table(assessment, colWidths=[150, 330])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eaf2ff")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("Student Profile", h))
    profile_rows = [[k.replace("_", " "), str(student[k])] for k in student]
    t = Table(profile_rows, colWidths=[220, 260])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("Top Career Recommendations", h))
    career_rows = [["Career", "Match Score"]] + [[r["Career"], f'{r["Match Score"]:.2f}%'] for _, r in careers.head(3).iterrows()]
    t = Table(career_rows, colWidths=[320, 160])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf2ff")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("Skill Gaps", h))
    if len(gaps):
        gap_rows = [["Skill", "Current", "Required", "Gap"]] + [
            [r["Skill"].replace("_", " "), str(r["Current"]), str(r["Required"]), str(r["Gap"])]
            for _, r in gaps.head(8).iterrows()
        ]
    else:
        gap_rows = [["Result", "No major gaps detected for the top career."]]
    t = Table(gap_rows, colWidths=[180, 100, 100, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf2ff")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("Suggested Improvement Roadmap", h))
    for action in actions:
        story.append(Paragraph("• " + action, normal))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 12))
    story.append(Paragraph("CareerCompass AI prototype – generated report", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
