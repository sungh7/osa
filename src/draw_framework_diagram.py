import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Paths
RESULTS_PATH = Path('/data/knhanes/paper_revision/results')
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

def draw_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Function to draw box
    def draw_box(x, y, width, height, text, color='lightblue', edgecolor='black'):
        rect = patches.FancyBboxPatch((x, y), width, height, 
                                      boxstyle="round,pad=0.1", 
                                      linewidth=2, 
                                      edgecolor=edgecolor, 
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=11, fontweight='bold', multialignment='center')
        return x + width/2, y, y + height # Return connection points (center_x, bottom_y, top_y)

    # 1. Population Input
    c1, b1, t1 = draw_box(3.5, 8.5, 3, 1, "Target Population\n(Adults Age 40+)", color='#e6f3ff')
    
    # Arrow to Stage 1
    ax.annotate("", xy=(5, 7.6), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    # 2. Stage 1 Model
    c2, b2, t2 = draw_box(3.5, 6.5, 3, 1, "Stage 1: STOP-Bang\n(Screening)", color='#ffebcc')

    # Arrow to Screen Positive (The issue area)
    # We draw an arrow down. The decision "Score >= 3" is the label.
    ax.annotate("", xy=(5, 5.0), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    # Label for the arrow (The "Screen Positive" part)
    # Positioning it to the side to avoid arrow overlap
    ax.text(5.1, 5.8, "Screen Positive\n(Score ≥ 3)", 
            ha='left', va='center', fontsize=10, color='darkred', weight='bold')
    
    # Arrow for Screen Negative
    ax.annotate("", xy=(7.5, 7.0), xytext=(6.5, 7.0),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(7.6, 7.0, "Low Risk (<3)\n→ Observation", ha='left', va='center', fontsize=10)


    # 3. Decision Node / Stage 2 Input
    # Instead of just "Screen Positive", this flows into Stage 2
    c3, b3, t3 = draw_box(3.5, 4.0, 3, 1, "Stage 2: Triage Model\n(HSAT Features: ODI, SpO2)", color='#d9ead3')

    # Arrow to Outcome
    ax.annotate("", xy=(5, 2.5), xytext=(5, 4.0),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    # 4. Outcomes
    # Left: High Risk
    ax.annotate("", xy=(3.0, 2.0), xytext=(5, 2.8),
                arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    c4a, b4a, t4a = draw_box(1.0, 1.0, 3, 1, "Priority Referral\n(Predicted Severe OSA)", color='#f4cccc', edgecolor='red')

    # Right: Low Risk (Triage Negative)
    ax.annotate("", xy=(7.0, 2.0), xytext=(5, 2.8),
                arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    c4b, b4b, t4b = draw_box(6.0, 1.0, 3, 1, "Routine Referral / Monitoring\n(Predicted Non-Severe)", color='#fce5cd')


    # Title
    ax.text(5, 9.8, "Two-Stage OSA Screening Framework", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = RESULTS_PATH / '16_two_stage_framework_diagram.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {save_path}")

if __name__ == "__main__":
    draw_diagram()
