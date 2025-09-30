import plotly.graph_objects as go
import plotly.express as px

# Since Mermaid service is unavailable, create a workflow diagram using Plotly
# Define the workflow steps
steps = [
    "Create Knowledge Base",
    "Configure Data Source", 
    "Run Ingestion",
    "Create Agent",
    "Configure Settings",
    "Create Endpoint",
    "Test Chat Interface"
]

# Create positions for the workflow steps
x_positions = list(range(len(steps)))
y_positions = [0] * len(steps)

# Create the figure
fig = go.Figure()

# Add workflow steps as scatter points
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325']

for i, (step, color) in enumerate(zip(steps, colors)):
    fig.add_trace(go.Scatter(
        x=[x_positions[i]], 
        y=[y_positions[i]],
        mode='markers+text',
        marker=dict(size=80, color=color, line=dict(width=2, color='white')),
        text=step,
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        showlegend=False,
        hovertemplate=f'<b>{step}</b><extra></extra>'
    ))

# Add arrows between steps
for i in range(len(steps)-1):
    fig.add_annotation(
        x=x_positions[i+1]-0.4,
        y=y_positions[i+1],
        ax=x_positions[i]+0.4,
        ay=y_positions[i],
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='#333333'
    )

# Update layout
fig.update_layout(
    title="Agent Creation Workflow",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, len(steps)-0.5]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-1, 1]
    ),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Save as PNG and SVG
fig.write_image("agent_workflow.png")
fig.write_image("agent_workflow.svg", format="svg")

print("Workflow diagram saved as PNG and SVG")