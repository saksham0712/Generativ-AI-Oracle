import plotly.graph_objects as go
import json

# Data for the comparison table
data = {
    "data_stores": [
        {
            "type": "Object Storage",
            "management": "Service-managed",
            "data_format": "PDF and TXT files",
            "file_limits": "1000 files, 100MB each",
            "setup_effort": "Low",
            "use_case": "Quick setup, standard documents",
            "ingestion": "Automatic",
            "search_type": "Hybrid (lexical + semantic)"
        },
        {
            "type": "OpenSearch",
            "management": "Customer-managed",
            "data_format": "Pre-indexed data",
            "file_limits": "Depends on OpenSearch setup",
            "setup_effort": "Medium",
            "use_case": "Existing search infrastructure",
            "ingestion": "Pre-ingested",
            "search_type": "Configurable"
        },
        {
            "type": "Oracle 23ai Vector",
            "management": "Customer-managed",
            "data_format": "Vector embeddings",
            "file_limits": "Database dependent",
            "setup_effort": "High",
            "use_case": "Advanced vector operations",
            "ingestion": "Custom functions",
            "search_type": "Vector similarity"
        }
    ]
}

# Extract data for the table
stores = data["data_stores"]

# Create headers and values for the table
headers = ["Feature", "Object Storage", "OpenSearch", "Oracle 23ai Vector"]

# Create rows of data
rows = [
    ["Management", stores[0]["management"], stores[1]["management"], stores[2]["management"]],
    ["Data Format", stores[0]["data_format"], stores[1]["data_format"], stores[2]["data_format"]],
    ["File Limits", stores[0]["file_limits"], stores[1]["file_limits"], stores[2]["file_limits"]],
    ["Setup Effort", stores[0]["setup_effort"], stores[1]["setup_effort"], stores[2]["setup_effort"]],
    ["Use Case", stores[0]["use_case"], stores[1]["use_case"], stores[2]["use_case"]],
    ["Ingestion", stores[0]["ingestion"], stores[1]["ingestion"], stores[2]["ingestion"]],
    ["Search Type", stores[0]["search_type"], stores[1]["search_type"], stores[2]["search_type"]]
]

# Create the table
fig = go.Figure(data=[go.Table(
    header=dict(
        values=headers,
        fill_color='#1FB8CD',
        align='left',
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=[[row[0] for row in rows],  # Feature column
                [row[1] for row in rows],  # Object Storage column
                [row[2] for row in rows],  # OpenSearch column  
                [row[3] for row in rows]], # Oracle 23ai Vector column
        fill_color=[['#E8F4F8']*len(rows), ['white']*len(rows), ['white']*len(rows), ['white']*len(rows)],
        align='left',
        font=dict(size=11),
        height=40
    )
)])

fig.update_layout(
    title="AI Agent Data Store Comparison",
    font=dict(family="Arial")
)

# Save as PNG and SVG
fig.write_image("comparison_table.png")
fig.write_image("comparison_table.svg", format="svg")