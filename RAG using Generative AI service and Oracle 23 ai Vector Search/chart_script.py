import plotly.graph_objects as go

# Create a comprehensive RAG pipeline flowchart using plotly
fig = go.Figure()

# Define exact process labels from the data
ingestion_processes = ["Load Documents", "Split into Chunks", "Generate Embeddings", "Store in Vector DB"]
retrieval_processes = ["User Query", "Query Embedding", "Vector Search", "Top-K Results"]
generation_processes = ["Context + Query", "LLM Processing", "Final Response"]

# Define positions for the three phases (properly aligned)
# Ingestion phase (left column)
ingestion_x = [1] * 4
ingestion_y = [4, 3, 2, 1]

# Retrieval phase (middle column) 
retrieval_x = [3] * 4
retrieval_y = [4, 3, 2, 1]

# Generation phase (right column)
generation_x = [5] * 3
generation_y = [4, 3, 2]

# Color scheme using brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57']
box_color = '#13343B'

# Add boxes for Ingestion phase
for i, (x, y, label) in enumerate(zip(ingestion_x, ingestion_y, ingestion_processes)):
    fig.add_shape(
        type="rect",
        x0=x-0.45, y0=y-0.18, x1=x+0.45, y1=y+0.18,
        fillcolor=colors[0], opacity=0.9,
        line=dict(width=2, color=box_color)
    )
    fig.add_annotation(
        x=x, y=y,
        text=label,
        showarrow=False,
        font=dict(color='white', size=9, family="Arial"),
        align='center'
    )

# Add boxes for Retrieval phase
for i, (x, y, label) in enumerate(zip(retrieval_x, retrieval_y, retrieval_processes)):
    fig.add_shape(
        type="rect",
        x0=x-0.45, y0=y-0.18, x1=x+0.45, y1=y+0.18,
        fillcolor=colors[1], opacity=0.9,
        line=dict(width=2, color=box_color)
    )
    fig.add_annotation(
        x=x, y=y,
        text=label,
        showarrow=False,
        font=dict(color='white', size=9, family="Arial"),
        align='center'
    )

# Add boxes for Generation phase
for i, (x, y, label) in enumerate(zip(generation_x, generation_y, generation_processes)):
    fig.add_shape(
        type="rect",
        x0=x-0.45, y0=y-0.18, x1=x+0.45, y1=y+0.18,
        fillcolor=colors[2], opacity=0.9,
        line=dict(width=2, color=box_color)
    )
    fig.add_annotation(
        x=x, y=y,
        text=label,
        showarrow=False,
        font=dict(color='white', size=9, family="Arial"),
        align='center'
    )

# Add sequential arrows within each phase
# Ingestion phase arrows (between all steps)
for i in range(len(ingestion_y) - 1):
    fig.add_annotation(
        x=ingestion_x[i], y=ingestion_y[i+1] + 0.25,
        ax=ingestion_x[i], ay=ingestion_y[i] - 0.25,
        axref='x', ayref='y',
        xref='x', yref='y',
        arrowhead=2, arrowsize=1.2, arrowwidth=2,
        arrowcolor=box_color,
        showarrow=True
    )

# Retrieval phase arrows (between all steps)
for i in range(len(retrieval_y) - 1):
    fig.add_annotation(
        x=retrieval_x[i], y=retrieval_y[i+1] + 0.25,
        ax=retrieval_x[i], ay=retrieval_y[i] - 0.25,
        axref='x', ayref='y',
        xref='x', yref='y',
        arrowhead=2, arrowsize=1.2, arrowwidth=2,
        arrowcolor=box_color,
        showarrow=True
    )

# Generation phase arrows (between all steps)
for i in range(len(generation_y) - 1):
    fig.add_annotation(
        x=generation_x[i], y=generation_y[i+1] + 0.25,
        ax=generation_x[i], ay=generation_y[i] - 0.25,
        axref='x', ayref='y',
        xref='x', yref='y',
        arrowhead=2, arrowsize=1.2, arrowwidth=2,
        arrowcolor=box_color,
        showarrow=True
    )

# Add arrows between phases (connecting last step of one to first step of next)
# Ingestion to Retrieval (Store in Vector DB to User Query)
fig.add_annotation(
    x=retrieval_x[0] - 0.45, y=retrieval_y[0],
    ax=ingestion_x[-1] + 0.45, ay=ingestion_y[-1],
    axref='x', ayref='y',
    xref='x', yref='y',
    arrowhead=2, arrowsize=1.5, arrowwidth=3,
    arrowcolor=box_color,
    showarrow=True
)

# Retrieval to Generation (Top-K Results to Context + Query)
fig.add_annotation(
    x=generation_x[0] - 0.45, y=generation_y[0],
    ax=retrieval_x[-1] + 0.45, ay=retrieval_y[-1],
    axref='x', ayref='y',
    xref='x', yref='y',
    arrowhead=2, arrowsize=1.5, arrowwidth=3,
    arrowcolor=box_color,
    showarrow=True
)

# Add phase titles
fig.add_annotation(x=1, y=4.8, text="Ingestion", showarrow=False, 
                  font=dict(size=14, color=colors[0], family="Arial Black"))
fig.add_annotation(x=3, y=4.8, text="Retrieval", showarrow=False, 
                  font=dict(size=14, color=colors[1], family="Arial Black"))
fig.add_annotation(x=5, y=4.8, text="Generation", showarrow=False, 
                  font=dict(size=14, color=colors[2], family="Arial Black"))

# Update layout
fig.update_layout(
    title="RAG Pipeline Process Flow",
    showlegend=False,
    xaxis=dict(range=[0, 6], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0.5, 5.3], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save as both PNG and SVG
fig.write_image('rag_pipeline.png')
fig.write_image('rag_pipeline.svg', format='svg')

print("Complete RAG pipeline flowchart with all processes saved successfully!")