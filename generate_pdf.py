"""generate_pdf.py
Create a professional, well-formatted PDF report from training artifacts.
This script reads `model_metadata.json` and images saved under `artifacts/` and produces `report.pdf`.
Enhanced with beautiful styling, proper alignment, and comprehensive sections.
"""
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                PageBreak, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from pathlib import Path
import json
import textwrap
from datetime import datetime

ROOT = Path(__file__).resolve().parent
MD = ROOT / 'DOCUMENTATION.md'
META = ROOT / 'model_metadata.json'
ART = ROOT / 'artifacts'
OUT = ROOT / 'report.pdf'

def create_custom_styles():
    """Create custom paragraph styles for better formatting"""
    styles = getSampleStyleSheet()
    
    # Custom title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Custom heading style
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3b82f6'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#e0e7ff'),
        borderPadding=5,
        backColor=colors.HexColor('#eff6ff')
    ))
    
    # Custom subheading style
    styles.add(ParagraphStyle(
        name='CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    ))
    
    # Custom body text style
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    ))
    
    # Custom bullet style
    styles.add(ParagraphStyle(
        name='CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=6
    ))
    
    return styles

def add_header(story, styles):
    """Add professional header section"""
    # Title
    story.append(Paragraph('🏥 Breast Cancer Prediction System', styles['CustomTitle']))
    story.append(Spacer(1, 6))
    
    # Subtitle
    subtitle = f'<font size=12 color="#64748b"><i>ML-Powered Diagnostic Support System</i></font>'
    story.append(Paragraph(subtitle, styles['Normal']))
    story.append(Spacer(1, 6))
    
    # Date
    date_str = datetime.now().strftime('%B %d, %Y')
    story.append(Paragraph(f'<font size=10 color="#94a3b8">Generated: {date_str}</font>', 
                          ParagraphStyle('center', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(Spacer(1, 30))

def add_executive_summary(story, styles, meta):
    """Add executive summary section"""
    story.append(Paragraph('📊 Executive Summary', styles['CustomHeading']))
    story.append(Spacer(1, 12))
    
    summary_text = """
    This report presents the results of a machine learning system developed for breast cancer 
    diagnosis prediction. The system utilizes multiple model architectures including traditional 
    machine learning pipelines, ensemble stacking methods, and optional deep learning approaches 
    to provide accurate and interpretable predictions.
    """
    story.append(Paragraph(summary_text.strip(), styles['CustomBody']))
    story.append(Spacer(1, 20))

def add_model_metadata(story, styles, meta):
    """Add model metadata in a professional table format"""
    story.append(Paragraph('🤖 Model Configuration', styles['CustomHeading']))
    story.append(Spacer(1, 12))
    
    if not meta:
        story.append(Paragraph('No metadata available.', styles['CustomBody']))
        return
    
    # Create data for table
    data = [['Parameter', 'Value']]
    
    # Format metadata nicely
    key_mapping = {
        'model': 'Model Type',
        'test_size': 'Test Set Size',
        'random_state': 'Random State',
        'cv_folds': 'Cross-Validation Folds',
        'best_params': 'Best Parameters',
        'train_accuracy': 'Training Accuracy',
        'test_accuracy': 'Test Accuracy',
        'cv_mean_score': 'CV Mean Score',
        'cv_std_score': 'CV Std Score',
        'feature_count': 'Feature Count',
        'selected_features': 'Selected Features'
    }
    
    for key, value in meta.items():
        display_key = key_mapping.get(key, key.replace('_', ' ').title())
        
        # Format value
        if isinstance(value, float):
            display_value = f'{value:.4f}'
        elif isinstance(value, dict):
            display_value = '<br/>'.join([f'{k}: {v}' for k, v in value.items()])
        else:
            display_value = str(value)
        
        data.append([display_key, display_value])
    
    # Create table with styling
    table = Table(data, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f5f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))

def add_performance_metrics(story, styles):
    """Add performance metrics section"""
    story.append(Paragraph('📈 Model Performance', styles['CustomHeading']))
    story.append(Spacer(1, 12))
    
    performance_text = """
    The model's performance is evaluated using multiple metrics to ensure comprehensive 
    assessment of diagnostic accuracy. Key metrics include accuracy, precision, recall, 
    F1-score, and AUC-ROC. The confusion matrix and ROC curve visualizations below provide 
    detailed insight into the model's classification performance across different thresholds.
    """
    story.append(Paragraph(performance_text.strip(), styles['CustomBody']))
    story.append(Spacer(1, 15))

def add_visualizations(story, styles):
    """Add visualization sections with descriptions"""
    viz_info = [
        {
            'file': 'confusion_matrix.png',
            'title': '🎯 Confusion Matrix',
            'description': 'The confusion matrix shows the distribution of true positives, true negatives, '
                          'false positives, and false negatives. This visualization helps identify if the model '
                          'has any systematic biases in its predictions.'
        },
        {
            'file': 'roc_curve.png',
            'title': '📊 ROC Curve',
            'description': 'The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability '
                          'of the model at various threshold settings. The Area Under the Curve (AUC) provides '
                          'a single metric for model performance, where 1.0 represents perfect classification.'
        },
        {
            'file': 'feature_importances.png',
            'title': '🔍 Feature Importance',
            'description': 'Feature importance analysis identifies which clinical measurements contribute most '
                          'to the model\'s predictions. This helps medical professionals understand the key '
                          'factors driving diagnostic decisions.'
        }
    ]
    
    for viz in viz_info:
        img_path = ART / viz['file']
        if img_path.exists():
            story.append(PageBreak())
            story.append(Paragraph(viz['title'], styles['CustomHeading']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(viz['description'], styles['CustomBody']))
            story.append(Spacer(1, 15))
            
            # Center-aligned image
            img = Image(str(img_path), width=5.5*inch, height=4*inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 15))

def add_technical_documentation(story, styles):
    """Add technical documentation section"""
    if not MD.exists():
        return
    
    story.append(PageBreak())
    story.append(Paragraph('📚 Technical Documentation', styles['CustomHeading']))
    story.append(Spacer(1, 12))
    
    try:
        doc_text = MD.read_text(encoding='utf-8')
        
        # Parse markdown-like sections
        lines = doc_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            # Handle headings
            if line.startswith('###'):
                text = line.replace('###', '').strip()
                story.append(Paragraph(text, styles['CustomSubHeading']))
            elif line.startswith('##'):
                text = line.replace('##', '').strip()
                story.append(Paragraph(text, styles['CustomHeading']))
            elif line.startswith('#'):
                text = line.replace('#', '').strip()
                story.append(Paragraph(text, styles['CustomTitle']))
            elif line.startswith('- ') or line.startswith('* '):
                text = '• ' + line[2:]
                story.append(Paragraph(text, styles['CustomBullet']))
            else:
                # Regular paragraph
                story.append(Paragraph(line, styles['CustomBody']))
        
    except Exception as e:
        story.append(Paragraph(f'Failed to include documentation: {str(e)}', styles['CustomBody']))

def add_footer_section(story, styles):
    """Add conclusion and footer"""
    story.append(PageBreak())
    story.append(Paragraph('✅ Conclusion', styles['CustomHeading']))
    story.append(Spacer(1, 12))
    
    conclusion = """
    This breast cancer prediction system demonstrates the successful application of machine learning 
    to medical diagnostics. The model achieves high accuracy while maintaining interpretability through 
    feature importance analysis and SHAP values. The system is designed to assist medical professionals 
    in making informed diagnostic decisions, complementing traditional diagnostic methods.
    """
    story.append(Paragraph(conclusion.strip(), styles['CustomBody']))
    story.append(Spacer(1, 20))
    
    # Technology stack
    story.append(Paragraph('🔧 Technology Stack', styles['CustomSubHeading']))
    story.append(Spacer(1, 8))
    
    tech_items = [
        '• Backend: FastAPI, Python 3.9+',
        '• ML Framework: Scikit-learn, TensorFlow (optional)',
        '• Frontend: Next.js, React, Tailwind CSS',
        '• Visualization: Matplotlib, Seaborn, Plotly',
        '• Explainability: SHAP values'
    ]
    
    for item in tech_items:
        story.append(Paragraph(item, styles['CustomBullet']))
    
    story.append(Spacer(1, 30))
    
    # Footer note
    footer_text = '<font size=9 color="#94a3b8"><i>This report was automatically generated. ' \
                 'For more information, please refer to the project documentation.</i></font>'
    story.append(Paragraph(footer_text, ParagraphStyle('center', parent=styles['Normal'], alignment=TA_CENTER)))

def build_pdf():
    """Build the complete PDF report with enhanced formatting"""
    doc = SimpleDocTemplate(
        str(OUT), 
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = create_custom_styles()
    story = []
    
    # Load metadata
    meta = {}
    if META.exists():
        try:
            meta = json.loads(META.read_text())
        except Exception as e:
            print(f'Warning: Could not load metadata: {e}')
    
    # Build report sections
    add_header(story, styles)
    add_executive_summary(story, styles, meta)
    add_model_metadata(story, styles, meta)
    add_performance_metrics(story, styles)
    add_visualizations(story, styles)
    add_technical_documentation(story, styles)
    add_footer_section(story, styles)
    
    # Generate PDF
    doc.build(story)
    print(f'✅ Generated enhanced PDF report: {OUT}')
    print(f'📄 File size: {OUT.stat().st_size / 1024:.2f} KB')

if __name__ == '__main__':
    build_pdf()
