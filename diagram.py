# generate_workflow_diagram.py
"""
Generate a professional workflow diagram for Emotion Recognition Project
Install: pip install matplotlib matplotlib-patches
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.lines as mlines

def create_workflow_diagram():
    """Create a professional vertical workflow diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title box
    title_box = FancyBboxPatch((1, 12.5), 8, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#2C3E50',
                               edgecolor='black',
                               linewidth=2)
    ax.add_patch(title_box)
    ax.text(5, 13.1, 'EMOTION RECOGNITION PROJECT', 
            fontsize=18, weight='bold', ha='center', color='white')
    ax.text(5, 12.7, 'WORKFLOW', 
            fontsize=16, weight='bold', ha='center', color='white')
    
    # Define workflow stages
    stages = [
        {
            'title': '1. PROBLEM\nDEFINITION',
            'items': ['• Identify need', '• Research gap', '• Set objectives'],
            'color': '#3498DB',
            'y': 11
        },
        {
            'title': '2. DATA\nACQUISITION',
            'items': ['• Download MELD', '• Explore data', '• Understand format'],
            'color': '#E74C3C',
            'y': 9.5
        },
        {
            'title': '3. DATA\nPREPROCESSING',
            'items': ['• Clean text', '• Handle missing', '• Split dataset'],
            'color': '#F39C12',
            'y': 8
        },
        {
            'title': '4. MODEL\nSELECTION',
            'items': ['• BERT baseline', '• DialogueRNN', '• Architecture design'],
            'color': '#27AE60',
            'y': 6.5
        },
        {
            'title': '5. MODEL\nTRAINING',
            'items': ['• Train models', '• Validate', '• Tune params'],
            'color': '#8E44AD',
            'y': 5
        },
        {
            'title': '6. EVALUATION',
            'items': ['• Test accuracy', '• Confusion matrix', '• F1 scores'],
            'color': '#16A085',
            'y': 3.5
        },
        {
            'title': '7. STREAMLIT\nAPPLICATION',
            'items': ['• Web interface', '• User testing', '• Go live!'],
            'color': '#C0392B',
            'y': 2
        }
    ]
    
    # Draw workflow boxes
    box_width = 3.5
    box_height = 1.2
    x_center = 5
    
    for i, stage in enumerate(stages):
        # Main box
        box = FancyBboxPatch((x_center - box_width/2, stage['y'] - box_height/2), 
                            box_width, box_height,
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black',
                            linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)
        
        # Title text
        ax.text(x_center, stage['y'] + 0.2, stage['title'], 
                fontsize=12, weight='bold', ha='center', va='center', color='white')
        
        # Detail box
        detail_width = 4
        detail_height = 0.8
        detail_y = stage['y'] - 0.9
        
        detail_box = Rectangle((x_center - detail_width/2, detail_y - detail_height/2),
                              detail_width, detail_height,
                              facecolor='white',
                              edgecolor=stage['color'],
                              linewidth=1.5)
        ax.add_patch(detail_box)
        
        # Detail text
        detail_text = '\n'.join(stage['items'])
        ax.text(x_center, detail_y, detail_text, 
                fontsize=9, ha='center', va='center')
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x_center, stage['y'] - box_height/2 - 0.4),
                                  (x_center, stages[i+1]['y'] + box_height/2 + 0.4),
                                  arrowstyle='-|>',
                                  shrinkA=5, shrinkB=5,
                                  mutation_scale=20,
                                  facecolor='black',
                                  linewidth=2)
            ax.add_patch(arrow)
    
    # Add side annotations
    # Timeline
    timeline_x = 8.5
    ax.text(timeline_x, 11.5, 'Week 1', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 9, 'Week 1', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 7, 'Week 1-2', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 5.5, 'Week 2', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 4, 'Week 2-3', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 2.5, 'Week 3', fontsize=9, style='italic', color='gray')
    ax.text(timeline_x, 1, 'Week 3-4', fontsize=9, style='italic', color='gray')
    
    # Add key metrics
    metrics_box = FancyBboxPatch((0.5, 0.2), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#ECF0F1',
                                edgecolor='black',
                                linewidth=1)
    ax.add_patch(metrics_box)
    ax.text(2.5, 0.6, 'Final Results:', fontsize=10, weight='bold', ha='center')
    ax.text(2.5, 0.3, '65-69% Accuracy | 64-68% F1 Score', 
            fontsize=9, ha='center', style='italic')
    
    # Add technology stack
    tech_box = FancyBboxPatch((5.5, 0.2), 4, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#ECF0F1',
                             edgecolor='black',
                             linewidth=1)
    ax.add_patch(tech_box)
    ax.text(7.5, 0.6, 'Tech Stack:', fontsize=10, weight='bold', ha='center')
    ax.text(7.5, 0.3, 'Python | PyTorch | BERT | Streamlit', 
            fontsize=9, ha='center', style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('emotion_recognition_workflow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('emotion_recognition_workflow.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ Workflow diagram saved as:")
    print("   - emotion_recognition_workflow.png")
    print("   - emotion_recognition_workflow.pdf")
    
    plt.show()

# ...existing code...

def create_simplified_workflow():
    """Create a simplified horizontal workflow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7, 'Emotion Recognition Project Workflow', 
            fontsize=20, weight='bold', ha='center')
    
    # Simplified stages for horizontal layout (removed Deployment step)
    stages = [
        {'name': 'Data\nCollection', 'color': '#3498DB', 'x': 2},
        {'name': 'Data\nPrep', 'color': '#E74C3C', 'x': 4},
        {'name': 'Model\nTraining', 'color': '#27AE60', 'x': 6},
        {'name': 'Evaluation', 'color': '#F39C12', 'x': 8},
        {'name': 'Streamlit\nApp', 'color': '#C0392B', 'x': 10}
    ]
    
    y_center = 4
    box_size = 1.5
    
    for i, stage in enumerate(stages):
        # Draw box
        box = FancyBboxPatch((stage['x'] - box_size/2, y_center - box_size/2), 
                            box_size, box_size,
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black',
                            linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)
        
        # Add text
        ax.text(stage['x'], y_center, stage['name'], 
                fontsize=11, weight='bold', ha='center', va='center', 
                color='white')
        
        # Add arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((stage['x'] + box_size/2 + 0.1, y_center),
                                  (stages[i+1]['x'] - box_size/2 - 0.1, y_center),
                                  arrowstyle='-|>',
                                  mutation_scale=20,
                                  linewidth=2)
            ax.add_patch(arrow)
    
    # Add key details below (removed Deployment detail)
    details = [
        'MELD Dataset\n13,708 samples',
        'Clean & \nTokenize',
        'BERT &\nDialogueRNN',
        '65-69%\nAccuracy',
        'Interactive\nWeb Demo'
    ]
    
    for i, (stage, detail) in enumerate(zip(stages, details)):
        ax.text(stage['x'], 2, detail, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='white', 
                         edgecolor=stage['color']))
    
    # Add timeline
    ax.text(7, 1, 'Timeline: 4 Weeks', 
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor='#ECF0F1'))
    
    plt.tight_layout()
    plt.savefig('emotion_recognition_workflow_horizontal.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()

# ...existing

def create_detailed_technical_workflow():
    """Create a detailed technical workflow with code snippets"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    title_text = ax.text(6, 15, 'Technical Implementation Workflow', 
                        fontsize=22, weight='bold', ha='center')
    
    # Define technical stages with code snippets
    stages = [
        {
            'title': '1. Data Loading',
            'code': 'df = pd.read_csv("train_sent_emo.csv")\ntokenizer = AutoTokenizer.from_pretrained("bert-base")',
            'y': 13.5
        },
        {
            'title': '2. Data Preprocessing',
            'code': 'def clean_text(text):\n    return text.lower().strip()\n\ndf["clean_text"] = df["text"].apply(clean_text)',
            'y': 11.5
        },
        {
            'title': '3. Model Definition',
            'code': 'class BERTEmotionClassifier(nn.Module):\n    def __init__(self):\n        self.bert = AutoModel.from_pretrained("bert-base")\n        self.classifier = nn.Linear(768, 7)',
            'y': 9.5
        },
        {
            'title': '4. Training Loop',
            'code': 'for epoch in range(5):\n    model.train()\n    for batch in train_loader:\n        loss = criterion(output, labels)\n        loss.backward()',
            'y': 7.5
        },
        {
            'title': '5. Evaluation',
            'code': 'model.eval()\nwith torch.no_grad():\n    predictions = model(test_data)\n    accuracy = calculate_accuracy(predictions, labels)',
            'y': 5.5
        },
        {
            'title': '6. Streamlit Deployment',
            'code': 'st.title("Emotion Recognition")\ntext = st.text_area("Enter text:")\nif st.button("Analyze"):\n    emotion = predict(text)\n    st.write(f"Emotion: {emotion}")',
            'y': 3.5
        }
    ]
    
    for i, stage in enumerate(stages):
        # Title box
        title_box = FancyBboxPatch((1, stage['y'] + 0.3), 10, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#34495E',
                                  edgecolor='black',
                                  linewidth=1.5)
        ax.add_patch(title_box)
        ax.text(6, stage['y'] + 0.7, stage['title'], 
                fontsize=14, weight='bold', ha='center', color='white')
        
        # Code box
        code_box = FancyBboxPatch((1, stage['y'] - 1.2), 10, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#F8F8F8',
                                 edgecolor='#34495E',
                                 linewidth=1)
        ax.add_patch(code_box)
        ax.text(6, stage['y'] - 0.5, stage['code'], 
                fontsize=9, ha='center', va='center', 
                family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='#E8E8E8'))
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((6, stage['y'] - 1.3),
                                  (6, stages[i+1]['y'] + 1.2),
                                  arrowstyle='-|>',
                                  mutation_scale=20,
                                  linewidth=2,
                                  color='#E74C3C')
            ax.add_patch(arrow)
    
    # Add results box
    results_box = FancyBboxPatch((1, 0.5), 10, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#27AE60',
                                edgecolor='black',
                                linewidth=2,
                                alpha=0.9)
    ax.add_patch(results_box)
    ax.text(6, 1.25, 'FINAL OUTPUT', 
            fontsize=14, weight='bold', ha='center', color='white')
    ax.text(6, 0.8, '✓ 65-69% Accuracy  ✓ Web Interface  ✓ API Ready', 
            fontsize=11, ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('emotion_recognition_technical_workflow.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating workflow diagrams...")
    
    # Generate all three versions
    print("\n1. Creating vertical workflow diagram...")
    create_workflow_diagram()
    
    print("\n2. Creating horizontal workflow diagram...")
    create_simplified_workflow()
    
    print("\n3. Creating technical workflow diagram...")
    create_detailed_technical_workflow()
    
    print("\n✅ All workflow diagrams generated successfully!")