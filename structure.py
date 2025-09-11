# clean_project_structure_diagram.py
"""
Create a clean, non-overlapping project structure diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as patches

def create_clean_project_structure():
    """Create a clean, well-spaced project structure diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    title_box = FancyBboxPatch((3, 18.5), 10, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='#2C3E50',
                               edgecolor='none')
    ax.add_patch(title_box)
    ax.text(8, 19, 'EMOTION RECOGNITION PROJECT STRUCTURE', 
            fontsize=20, weight='bold', ha='center', color='white')
    
    # Root folder - centered at top
    root_box = FancyBboxPatch((6.5, 17), 3, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor='#3498DB',
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(root_box)
    ax.text(8, 17.4, '📁 emotion_recognition/', 
            fontsize=13, weight='bold', ha='center', color='white')
    
    # Define structure with better spacing
    structure = {
        'root_files': {
            'title': '📁 Root Files',
            'y_start': 15,
            'x': 1,
            'width': 4,
            'color': '#E74C3C',
            'files': [
                ('⚙️ config.py', 'Configuration'),
                ('🏃 train.py', 'Training script'),
                ('🤖 train_dialogue_rnn.py', 'DialogueRNN train'),
                ('📊 evaluate.py', 'Evaluation'),
                ('🌐 streamlit_app.py', 'Web interface')
            ]
        },
        'models': {
            'title': '📁 models/',
            'y_start': 15,
            'x': 6,
            'width': 4,
            'color': '#27AE60',
            'files': [
                ('🧠 bert_baseline.py', 'BERT model'),
                ('🔄 dialogue_rnn.py', 'Context model'),
                ('📦 __init__.py', 'Package init')
            ]
        },
        'utils': {
            'title': '📁 utils/',
            'y_start': 15,
            'x': 11,
            'width': 4,
            'color': '#F39C12',
            'files': [
                ('📥 data_loader.py', 'Data loading'),
                ('🔄 data_loader_dialoguernn.py', 'DRN loader'),
                ('📈 metrics.py', 'Metrics'),
                ('🎨 visualization.py', 'Plotting')
            ]
        },
        'data': {
            'title': '📁 data/',
            'y_start': 8,
            'x': 3,
            'width': 4.5,
            'color': '#8E44AD',
            'files': [
                ('📊 train_sent_emo.csv', '9,989 samples'),
                ('📊 dev_sent_emo.csv', '1,109 samples'),
                ('📊 test_sent_emo.csv', '2,610 samples')
            ]
        },
        'outputs': {
            'title': '📁 outputs/',
            'y_start': 8,
            'x': 8.5,
            'width': 4.5,
            'color': '#16A085',
            'files': [
                ('💾 best_model.pth', 'Saved model'),
                ('📁 results/', 'Result files'),
                ('📊 confusion_matrix.png', 'Confusion matrix'),
                ('📈 training_curves.png', 'Training plots')
            ]
        }
    }
    
    # Draw folders and files with proper spacing
    for folder_key, folder_info in structure.items():
        # Calculate folder box position
        folder_y = folder_info['y_start']
        
        # Draw folder header
        folder_box = FancyBboxPatch(
            (folder_info['x'], folder_y), 
            folder_info['width'], 0.8,
            boxstyle="round,pad=0.05",
            facecolor=folder_info['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(folder_box)
        
        ax.text(
            folder_info['x'] + folder_info['width']/2, 
            folder_y + 0.4,
            folder_info['title'], 
            fontsize=12, weight='bold', 
            ha='center', color='white'
        )
        
        # Draw connection from root to folder
        root_x = 8
        root_y = 17
        folder_center_x = folder_info['x'] + folder_info['width']/2
        
        # Simple straight line connection
        ax.plot([root_x, folder_center_x], [root_y, folder_y + 0.8],
                'k-', linewidth=1, alpha=0.3)
        
        # Draw files with proper spacing
        file_y_start = folder_y - 0.5
        file_spacing = 0.6  # Increased spacing between files
        
        for i, (file_name, description) in enumerate(folder_info['files']):
            file_y = file_y_start - (i * file_spacing)
            
            # File box
            file_box = Rectangle(
                (folder_info['x'] + 0.1, file_y - 0.35), 
                folder_info['width'] - 0.2, 0.5,
                facecolor='white',
                edgecolor=folder_info['color'],
                linewidth=1
            )
            ax.add_patch(file_box)
            
            # File name (left aligned)
            ax.text(
                folder_info['x'] + 0.2, file_y,
                file_name,
                fontsize=9, va='center', ha='left'
            )
            
            # Description (right aligned)
            ax.text(
                folder_info['x'] + folder_info['width'] - 0.2, file_y,
                description,
                fontsize=7, va='center', ha='right',
                style='italic', color='gray'
            )
    
    # Add workflow connections with labels (simplified)
    # Define connections with better positioning
    connections = [
        {
            'from': (3, 13.5),  # train.py position
            'to': (8, 14.5),    # models folder
            'label': 'uses models',
            'color': '#E74C3C'
        },
        {
            'from': (8, 13),    # models
            'to': (5.5, 8.8),   # data folder
            'label': 'loads data',
            'color': '#27AE60'
        },
        {
            'from': (3, 12.5),  # train.py
            'to': (10.5, 8.8),  # outputs folder
            'label': 'saves results',
            'color': '#3498DB'
        },
        {
            'from': (3, 11),    # streamlit
            'to': (10.5, 7.5), # saved model
            'label': 'deploys model',
            'color': '#9B59B6'
        }
    ]
    
    # Draw connections
    for conn in connections:
        arrow = patches.FancyArrowPatch(
            conn['from'], conn['to'],
            connectionstyle="arc3,rad=.2",
            arrowstyle='->',
            mutation_scale=20,
            linewidth=2,
            color=conn['color'],
            alpha=0.6
        )
        ax.add_patch(arrow)
        
        # Add label at midpoint
        mid_x = (conn['from'][0] + conn['to'][0]) / 2
        mid_y = (conn['from'][1] + conn['to'][1]) / 2
        
        ax.text(
            mid_x, mid_y, conn['label'],
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3",
                     facecolor='white',
                     edgecolor=conn['color'],
                     alpha=0.9)
        )
    
    # Add legend at bottom
    legend_y = 3
    legend_box = FancyBboxPatch(
        (2, legend_y - 0.5), 12, 2,
        boxstyle="round,pad=0.1",
        facecolor='#ECF0F1',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(legend_box)
    
    ax.text(8, legend_y + 1.2, 'Component Types', 
            fontsize=13, weight='bold', ha='center')
    
    # Legend items in a grid
    legend_items = [
        ('⚙️ Configuration', '#E74C3C'),
        ('🧠 Models', '#27AE60'),
        ('📈 Utilities', '#F39C12'),
        ('📊 Data', '#8E44AD'),
        ('💾 Outputs', '#16A085'),
        ('🌐 Web App', '#3498DB')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x = 3 + (i % 3) * 4
        y = legend_y + 0.5 - (i // 3) * 0.5
        
        # Color indicator
        indicator = Rectangle((x - 0.3, y - 0.15), 0.3, 0.3,
                            facecolor=color, edgecolor='black')
        ax.add_patch(indicator)
        
        # Label
        ax.text(x + 0.1, y, label, fontsize=10, va='center')
    
    # Add project statistics at bottom
    stats_y = 0.5
    stats_box = FancyBboxPatch(
        (2, stats_y - 0.3), 12, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#34495E',
        edgecolor='none'
    )
    ax.add_patch(stats_box)
    
    stats = "📊 15 Python Files | 3 Models | 13,708 Samples | 65-69% Accuracy | Streamlit Deployed"
    ax.text(8, stats_y + 0.1, stats, 
            fontsize=11, ha='center', color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig('clean_project_structure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def create_simplified_structure():
    """Create an even simpler, hierarchical structure"""
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(6, 13, 'Project Structure Overview', 
            fontsize=18, weight='bold', ha='center')
    
    # Main components in a hierarchical layout
    components = [
        # (name, level, x_offset, color, icon)
        ('emotion_recognition/', 0, 0, '#2C3E50', '📁'),
        
        # Level 1 - Main folders
        ('Root Scripts', 1, -3, '#E74C3C', '⚙️'),
        ('models/', 1, -1, '#27AE60', '🧠'),
        ('utils/', 1, 1, '#F39C12', '📈'),
        ('data/', 1, -2, '#8E44AD', '📊'),
        ('outputs/', 1, 2, '#16A085', '💾'),
        
        # Level 2 - Key files
        ('config.py', 2, -3.5, '#E74C3C', ''),
        ('train.py', 2, -2.5, '#E74C3C', ''),
        ('streamlit_app.py', 2, -3, '#E74C3C', ''),
        
        ('bert_baseline.py', 2, -1, '#27AE60', ''),
        ('dialogue_rnn.py', 2, -0.5, '#27AE60', ''),
        
        ('data_loader.py', 2, 1, '#F39C12', ''),
        ('metrics.py', 2, 1.5, '#F39C12', ''),
        
        ('CSV files (13,708)', 2, -2, '#8E44AD', ''),
        
        ('best_model.pth', 2, 2, '#16A085', ''),
        ('results/', 2, 2.5, '#16A085', '')
    ]
    
    # Base positions
    base_y = 11
    level_spacing = 2.5
    
    # Draw components
    for name, level, x_offset, color, icon in components:
        y = base_y - (level * level_spacing)
        x = 6 + x_offset
        
        # Box size based on level
        box_width = 3 - (level * 0.5)
        box_height = 0.6 - (level * 0.1)
        
        # Draw box
        if level < 2:  # Only boxes for folders
            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=2 - level * 0.5,
                alpha=0.9 - level * 0.1
            )
            ax.add_patch(box)
            
            text_color = 'white' if level == 0 else 'white'
        else:  # Just text for files
            text_color = color
        
        # Draw text
        display_name = f"{icon} {name}" if icon else name
        fontsize = 12 - level * 2
        weight = 'bold' if level < 2 else 'normal'
        
        ax.text(x, y, display_name, 
                fontsize=fontsize, weight=weight,
                ha='center', va='center', color=text_color)
        
        # Draw connections
        if level > 0:
            # Find parent
            if level == 1:
                parent_y = base_y
                parent_x = 6
            else:
                # Connect to appropriate folder
                if 'config' in name or 'train' in name or 'streamlit' in name:
                    parent_y = base_y - level_spacing
                    parent_x = 6 - 3
                elif 'bert' in name or 'dialogue' in name:
                    parent_y = base_y - level_spacing
                    parent_x = 6 - 1
                elif 'data_loader' in name or 'metrics' in name:
                    parent_y = base_y - level_spacing
                    parent_x = 6 + 1
                elif 'CSV' in name:
                    parent_y = base_y - level_spacing
                    parent_x = 6 - 2
                else:  # outputs
                    parent_y = base_y - level_spacing
                    parent_x = 6 + 2
            
            # Draw line
            ax.plot([parent_x, x], [parent_y - 0.3, y + 0.3],
                   'k-', linewidth=1, alpha=0.3)
    
    # Add key workflow arrows at bottom
    workflow_y = 2
    ax.text(6, workflow_y + 1, 'Key Workflow', 
            fontsize=14, weight='bold', ha='center')
    
    # Workflow steps
    steps = ['Load Data', 'Train Model', 'Evaluate', 'Deploy']
    colors = ['#8E44AD', '#27AE60', '#F39C12', '#3498DB']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = 2 + i * 2.5
        
        # Step box
        step_box = FancyBboxPatch(
            (x - 0.8, workflow_y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(step_box)
        
        ax.text(x, workflow_y, step, 
                fontsize=10, ha='center', va='center',
                color='white', weight='bold')
        
        # Arrow to next step
        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch(
                (x + 0.8, workflow_y),
                (x + 1.7, workflow_y),
                arrowstyle='->',
                mutation_scale=15,
                linewidth=2,
                color='gray'
            )
            ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('simplified_project_structure.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Creating clean project structure diagrams...")
    
    print("\n1. Generating detailed clean structure...")
    create_clean_project_structure()
    
    print("\n2. Generating simplified hierarchical structure...")
    create_simplified_structure()
    
    print("\n✅ Diagrams created successfully!")
    print("Files generated:")
    print("  - clean_project_structure.png (detailed, no overlaps)")
    print("  - simplified_project_structure.png (hierarchical view)")