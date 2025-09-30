import plotly.graph_objects as go
import plotly.express as px

# Create hierarchical flowchart using Plotly since Mermaid service is unavailable
# Define the hierarchy data
hierarchy_data = [
    {"level": "Data Store", "description": "Object Storage/Database", "y": 6},
    {"level": "Data Source", "description": "Connection Details", "y": 5},
    {"level": "Knowledge Base", "description": "Vector Storage", "y": 4},
    {"level": "Agent", "description": "Autonomous LLM System", "y": 3},
    {"level": "Endpoint", "description": "Access Point", "y": 2},
    {"level": "User Interaction", "description": "Interface Layer", "y": 1}
]

# Create the figure
fig = go.Figure()

# Define colors from the brand palette
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C']

# Add nodes (boxes)
for i, item in enumerate(hierarchy_data):
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[item['y']],
        mode='markers+text',
        marker=dict(
            size=120,
            color=colors[i % len(colors)],
            symbol='square'
        ),
        text=f"{item['level']}<br>{item['description']}",
        textposition='middle center',
        textfont=dict(color='white', size=12),
        showlegend=False,
        hovertemplate=f"<b>{item['level']}</b><br>{item['description']}<extra></extra>"
    ))

# Add connecting arrows
for i in range(len(hierarchy_data) - 1):
    fig.add_trace(go.Scatter(
        x=[0.5, 0.5],
        y=[hierarchy_data[i]['y'] - 0.3, hierarchy_data[i+1]['y'] + 0.3],
        mode='lines',
        line=dict(color='#13343B', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add arrowhead
    fig.add_annotation(
        x=0.5,
        y=hierarchy_data[i+1]['y'] + 0.3,
        ax=0.5,
        ay=hierarchy_data[i+1]['y'] + 0.5,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='#13343B',
        showarrow=True
    )

# Update layout
fig.update_layout(
    title="AI Agent Data Structure Hierarchy",
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0.5, 6.5]),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Save as both PNG and SVG
fig.write_image("ai_agent_hierarchy.png")
fig.write_image("ai_agent_hierarchy.svg", format="svg")

print("Hierarchical flowchart created using Plotly and saved as both PNG and SVG files")