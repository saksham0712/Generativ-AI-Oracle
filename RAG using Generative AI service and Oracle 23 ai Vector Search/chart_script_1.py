import plotly.graph_objects as go
import plotly.express as px

# Create a visual representation of document chunking process using Plotly
fig = go.Figure()

# Add rectangles to represent the original document and chunks
# Original document
fig.add_shape(type="rect", x0=0, y0=8, x1=10, y1=9, 
              fillcolor="#B3E5EC", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=5, y=8.5, text="Original Document", showarrow=False, font=dict(size=14))

# Chunk size options
fig.add_shape(type="rect", x0=0, y0=6, x1=4, y1=7, 
              fillcolor="#FFCDD2", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=2, y=6.5, text="Small Chunks<br>Specific Content", showarrow=False, font=dict(size=10))

fig.add_shape(type="rect", x0=6, y0=6, x1=10, y1=7, 
              fillcolor="#A5D6A7", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=8, y=6.5, text="Large Chunks<br>Rich Context", showarrow=False, font=dict(size=10))

# Splitting strategies
fig.add_shape(type="rect", x0=0, y0=4, x1=3, y1=5, 
              fillcolor="#9FA8B0", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=1.5, y=4.5, text="Paragraph<br>Split", showarrow=False, font=dict(size=10))

fig.add_shape(type="rect", x0=3.5, y0=4, x1=6.5, y1=5, 
              fillcolor="#9FA8B0", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=5, y=4.5, text="Sentence<br>Split", showarrow=False, font=dict(size=10))

fig.add_shape(type="rect", x0=7, y0=4, x1=10, y1=5, 
              fillcolor="#9FA8B0", opacity=0.7, line=dict(color="black"))
fig.add_annotation(x=8.5, y=4.5, text="Word<br>Split", showarrow=False, font=dict(size=10))

# Overlapping chunks visualization
chunk_colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C"]

# Show overlapping chunks with different colors
for i in range(5):
    x_start = i * 1.5
    x_end = x_start + 2.5
    fig.add_shape(type="rect", x0=x_start, y0=2, x1=x_end, y1=2.8, 
                  fillcolor=chunk_colors[i], opacity=0.6, line=dict(color="black"))
    fig.add_annotation(x=x_start+1.25, y=2.4, text=f"Chunk {i+1}", showarrow=False, font=dict(size=10))

# Add overlap indicators
fig.add_annotation(x=1.75, y=1.5, text="Overlap", showarrow=False, font=dict(size=9, color="red"))
fig.add_annotation(x=3.25, y=1.5, text="Overlap", showarrow=False, font=dict(size=9, color="red"))
fig.add_annotation(x=4.75, y=1.5, text="Overlap", showarrow=False, font=dict(size=9, color="red"))
fig.add_annotation(x=6.25, y=1.5, text="Overlap", showarrow=False, font=dict(size=9, color="red"))

# Add arrows to show process flow
fig.add_annotation(x=5, y=7.5, text="▼", showarrow=False, font=dict(size=20))
fig.add_annotation(x=5, y=5.5, text="▼", showarrow=False, font=dict(size=20))
fig.add_annotation(x=5, y=3.5, text="▼", showarrow=False, font=dict(size=20))

# Add title and labels
fig.add_annotation(x=5, y=0.5, text="Overlapping Chunks with Context Continuity", 
                   showarrow=False, font=dict(size=12, color="gray"))

# Update layout
fig.update_layout(
    title="Document Chunking Process",
    xaxis=dict(range=[-0.5, 10.5], showgrid=False, showticklabels=False),
    yaxis=dict(range=[0, 10], showgrid=False, showticklabels=False),
    showlegend=False,
    plot_bgcolor="white"
)

# Save the chart
fig.write_image("chunking_process.png")
fig.write_image("chunking_process.svg", format="svg")

print("Chart saved successfully")