from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.units import inch


def generate_report(filename, data_dict):
    doc = SimpleDocTemplate(filename)
    elements = []

    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    elements.append(Paragraph("<b>Resume Evaluation Report</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.4 * inch))

    for key, value in data_dict.items():
        elements.append(Paragraph(f"<b>{key}:</b> {value}", normal))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
