import plotly.graph_objects as go
import json

# Load the data
data = {
    "similarity_measures": [
        {
            "measure": "Dot Product", 
            "formula": "A · B = |A| × |B| × cos(θ)", 
            "considers": ["Magnitude", "Angle"], 
            "characteristics": "Measures projection magnitude", 
            "nlp_context": "Semantically richer content has higher magnitude", 
            "use_case": "When content richness matters"
        }, 
        {
            "measure": "Cosine Similarity", 
            "formula": "cos(θ) = (A · B) / (|A| × |B|)", 
            "considers": ["Angle only"], 
            "characteristics": "Normalized similarity measure", 
            "nlp_context": "Pure semantic similarity regardless of length", 
            "use_case": "When only meaning similarity matters"
        }
    ]
}

# Extract data for table
measures = data["similarity_measures"]

# Create table data with abbreviated text to meet 15-char limit
header = ["Property", "Dot Product", "Cosine Sim"]

# Abbreviate content to meet character limits
rows = [
    ["Formula", "A·B=|A||B|cos θ", "A·B/(|A||B|)"],
    ["Considers", "Mag + Angle", "Angle only"],
    ["Type", "Proj magnitude", "Normalized"],
    ["NLP Context", "Rich content", "Pure semantic"],
    ["Use Case", "Rich matters", "Meaning only"]
]

# Create the table
fig = go.Figure(data=[go.Table(
    header=dict(
        values=header,
        align="center",
        font=dict(size=14, color="white"),
        fill_color="#1FB8CD"
    ),
    cells=dict(
        values=[[row[i] for row in rows] for i in range(len(header))],
        align=["left", "center", "center"],
        font=dict(size=12),
        fill_color=[["#f0f0f0", "white"] * len(rows)],
        height=40
    )
)])

fig.update_layout(
    title="Vector Similarity Measures Comparison",
    font=dict(family="Arial")
)

# Save as both PNG and SVG
fig.write_image("comparison_table.png")
fig.write_image("comparison_table.svg", format="svg")