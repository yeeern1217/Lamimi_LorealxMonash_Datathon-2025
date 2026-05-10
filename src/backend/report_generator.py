import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import os
from groq import Groq
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')

load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def generate_video_summary(title, description):
    """
    Generates a brief, engaging summary for a YouTube video using Groq API.

    Args:
        title (str): The title of the YouTube video.
        description (str): The description of the YouTube video.

    Returns:
        str: A concise summary of the video.
    """
        
    system_prompt = (
        "You are an AI assistant specialized in creating concise and engaging summaries for YouTube videos. "
        "Your tone should be friendly, informative, and appealing to viewers. "
        "Focus on capturing the main idea and key points in 1-2 sentences, return original text with not format."
    )
    
    user_prompt = f"Please provide a brief summary for this video:\n\nTitle: {title}\nDescription: {description}"
    
    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return chat_completion.choices[0].message.content

def calculate_comprehensive_kpis(df: pd.DataFrame) -> dict:
    """Calculate comprehensive KPIs with additional metrics."""
    total_comments = len(df)
    spam_count = df['is_spam'].sum() if 'is_spam' in df.columns else 0
    valid_comments = total_comments - spam_count
    
    # Sentiment analysis
    sentiment_dist = df['sentiment_label'].value_counts(normalize=True).to_dict() if 'sentiment_label' in df.columns else {}
    
    # Engagement metrics
    engagement_metrics = {}
    if 'weighted_relevance' in df.columns:
        engagement_metrics = {
            "High Relevance (>0.7)": len(df[df['weighted_relevance'] > 0.7]),
            "Medium Relevance (0.3-0.7)": len(df[(df['weighted_relevance'] >= 0.3) & (df['weighted_relevance'] <= 0.7)]),
            "Low Relevance (<0.3)": len(df[df['weighted_relevance'] < 0.3])
        }
    
    kpis = {
        "total_comments": total_comments,
        "spam_count": spam_count,
        "valid_comments": valid_comments,
        "spam_ratio": (spam_count / total_comments) * 100 if total_comments > 0 else 0,
        "avg_relevance": df['weighted_relevance'].mean() if 'weighted_relevance' in df.columns else 0,
        "avg_product_resonance": df['product_resonance_score'].mean() if 'product_resonance_score' in df.columns else 0,
        "sentiment_distribution": sentiment_dist,
        "category_distribution": df['cluster_label'].value_counts(normalize=True).to_dict() if 'cluster_label' in df.columns else {},
        "engagement_metrics": engagement_metrics,
        "top_categories": df['cluster_label'].value_counts().head(3).to_dict() if 'cluster_label' in df.columns else {}
    }
    return kpis

def generate_insightful_analysis(kpis: dict, video_title: str, df: pd.DataFrame) -> dict:
    """Generate comprehensive analysis using Groq AI."""
    
    # Prepare data for AI analysis
    sentiment_summary = ", ".join([f"{sentiment}: {percent:.1%}" 
                                 for sentiment, percent in kpis['sentiment_distribution'].items()])
    
    top_categories = ", ".join([f"{category} ({count} comments)" 
                              for category, count in kpis.get('top_categories', {}).items()])
    
    system_prompt = f"""
    You are a senior marketing insights analyst at a major beauty brand. Your task is to provide a comprehensive analysis of YouTube comments for a video.

    VIDEO: "{video_title}"

    KEY METRICS:
    - Total Comments: {kpis['total_comments']}
    - Valid Comments (non-spam): {kpis['valid_comments']}
    - Spam Ratio: {kpis['spam_ratio']:.1f}%
    - Average Relevance Score: {kpis['avg_relevance']:.2f}
    - Average Product Resonance: {kpis['avg_product_resonance']:.2f}
    - Sentiment Distribution: {sentiment_summary}
    - Top Categories: {top_categories}

    ANALYSIS FRAMEWORK:
    1. EXECUTIVE SUMMARY: Brief overview of comment engagement and quality
    2. AUDIENCE INSIGHTS: Deep dive into sentiment and audience reactions
    3. PRODUCT ANALYSIS: What products/categories are resonating
    4. CONTENT PERFORMANCE: How well the video content engaged viewers
    5. ACTIONABLE RECOMMENDATIONS: Specific, data-driven marketing recommendations

    REQUIREMENTS:
    - Be concise but insightful (200-300 words total)
    - Focus on business implications
    - Highlight both strengths and opportunities
    - Provide specific, actionable recommendations
    - Use professional marketing terminology
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate a comprehensive analysis following the framework above."}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        analysis_text = response.choices[0].message.content
        
        # Parse the analysis into sections
        sections = {
            "executive_summary": "",
            "audience_insights": "", 
            "product_analysis": "",
            "content_performance": "",
            "recommendations": ""
        }
        
        current_section = "executive_summary"
        for line in analysis_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if 'EXECUTIVE SUMMARY' in line.upper():
                current_section = "executive_summary"
            elif 'AUDIENCE INSIGHTS' in line.upper() or 'AUDIENCE' in line.upper():
                current_section = "audience_insights"
            elif 'PRODUCT ANALYSIS' in line.upper() or 'PRODUCT' in line.upper():
                current_section = "product_analysis"
            elif 'CONTENT PERFORMANCE' in line.upper() or 'CONTENT' in line.upper():
                current_section = "content_performance"
            elif 'RECOMMENDATIONS' in line.upper() or 'ACTIONABLE' in line.upper():
                current_section = "recommendations"
            else:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        return sections
        
    except Exception as e:
        return {"error": f"Analysis generation failed: {str(e)}"}

def create_visualizations(kpis: dict, df: pd.DataFrame) -> dict:
    """Create matplotlib visualizations for the report."""
    viz_files = {}
    
    try:
        # Sentiment distribution pie chart
        if kpis['sentiment_distribution']:
            plt.figure(figsize=(6, 4))
            sentiments = list(kpis['sentiment_distribution'].keys())
            sizes = [kpis['sentiment_distribution'][s] for s in sentiments]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9c80e']
            
            plt.pie(sizes, labels=sentiments, autopct='%1.1f%%', colors=colors[:len(sentiments)])
            plt.title('Comment Sentiment Distribution')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            viz_files['sentiment_chart'] = img_buffer
            plt.close()

        # Top categories bar chart
        if kpis.get('top_categories'):
            plt.figure(figsize=(8, 4))
            categories = list(kpis['top_categories'].keys())
            counts = list(kpis['top_categories'].values())
            
            bars = plt.bar(categories, counts, color=['#6a0dad', '#9b59b6', '#8e44ad'])
            plt.title('Top Product Categories Discussed')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Number of Comments')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            viz_files['categories_chart'] = img_buffer
            plt.close()

        # Relevance distribution
        if 'weighted_relevance' in df.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(df['weighted_relevance'].dropna(), bins=10, alpha=0.7, color='#3498db', edgecolor='black')
            plt.title('Comment Relevance Distribution')
            plt.xlabel('Relevance Score')
            plt.ylabel('Number of Comments')
            plt.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            viz_files['relevance_chart'] = img_buffer
            plt.close()
            
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return viz_files

def generate_pdf_report(df: pd.DataFrame) -> bytes:
    """Generate a professional PDF report with enhanced formatting and visualizations."""
    if df.empty:
        return b""

    try:
        # Create buffer for PDF
        pdf_buffer = io.BytesIO()
        
        # Initialize document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50'),
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=20
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50')
        )
        
        # Story elements
        story = []
        
        # Get data
        video_title = df['title'].iloc[0] if 'title' in df.columns else "Unknown Video"
        kpis = calculate_comprehensive_kpis(df)
        analysis = generate_insightful_analysis(kpis, video_title, df)
        visualizations = create_visualizations(kpis, df)
        
        # Title
        story.append(Paragraph("YouTube Comment Analysis Report", title_style))
        story.append(Paragraph(f"Video: {video_title}", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Key Metrics Table
        story.append(Paragraph("Key Metrics at a Glance", heading_style))
        
        metrics_data = [
            ['Metric', 'Value', 'Insight'],
            ['Total Comments', str(kpis['total_comments']), 'Overall engagement level'],
            ['Valid Comments', str(kpis['valid_comments']), 'Quality conversation volume'],
            ['Spam Ratio', f"{kpis['spam_ratio']:.1f}%", 'Comment section health'],
            ['Avg Relevance', f"{kpis['avg_relevance']:.2f}", 'Content alignment quality'],
            ['Avg Product Resonance', f"{kpis['avg_product_resonance']:.2f}", 'Product discussion intensity']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        if 'executive_summary' in analysis:
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(analysis['executive_summary'], body_style))
        
        # Visualizations
        if visualizations:
            story.append(Paragraph("Data Visualizations", heading_style))
            
            # Create table for charts
            viz_data = []
            current_row = []
            
            for i, (viz_name, viz_buffer) in enumerate(visualizations.items()):
                img = Image(viz_buffer, width=3*inch, height=2.5*inch)
                current_row.append(img)
                
                if len(current_row) == 2 or i == len(visualizations) - 1:
                    viz_data.append(current_row)
                    current_row = []
            
            if viz_data:
                viz_table = Table(viz_data)
                viz_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(viz_table)
                story.append(Spacer(1, 20))
        
        # Detailed Analysis Sections
        sections_order = [
            ('audience_insights', 'Audience Insights'),
            ('product_analysis', 'Product Analysis'),
            ('content_performance', 'Content Performance'),
        ]
        
        for section_key, section_title in sections_order:
            if section_key in analysis and analysis[section_key]:
                story.append(Paragraph(section_title, heading_style))
                story.append(Paragraph(analysis[section_key], body_style))
        
        # Recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            story.append(Paragraph("Actionable Recommendations", heading_style))
            story.append(Paragraph(analysis['recommendations'], body_style))
        
        # Detailed KPIs
        story.append(Paragraph("Detailed Metrics", heading_style))
        
        # Sentiment Distribution
        if kpis['sentiment_distribution']:
            sentiment_data = [['Sentiment', 'Percentage']]
            for sentiment, percentage in kpis['sentiment_distribution'].items():
                sentiment_data.append([sentiment.capitalize(), f"{percentage:.1%}"])
            
            sentiment_table = Table(sentiment_data, colWidths=[3*inch, 1.5*inch])
            sentiment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
            ]))
            story.append(Paragraph("Sentiment Distribution", styles['Heading3']))
            story.append(sentiment_table)
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        # Clean up visualization buffers
        for viz_buffer in visualizations.values():
            viz_buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return b""