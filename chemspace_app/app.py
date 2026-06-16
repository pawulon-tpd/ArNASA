"""Interactive UMAP chemical space explorer for SDF / SMI / CSV(SMILES) files."""
import base64
import io

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, no_update
from rdkit import Chem
from rdkit.Chem import Draw

from chem_io import parse_molecules
from embedding import compute_clusters, compute_umap
from features import compute_descriptors, compute_fingerprints

# Module-level cache mapping SMILES -> RDKit Mol, used to render structure images on click.
MOL_CACHE = {}

app = Dash(__name__)
app.title = "Chemical Space Explorer"

app.layout = html.Div(
    [
        html.H2("Chemical Space Explorer (UMAP)"),
        html.P("Upload an SDF, SMI, or CSV (with a SMILES column) file to map its chemical space."),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or ", html.A("select a file")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(id="status-message", style={"marginTop": "10px", "color": "#a00"}),
        html.Div(
            [
                html.Label("Color by:"),
                dcc.Dropdown(id="color-by-dropdown", clearable=False),
            ],
            style={"width": "300px", "marginTop": "15px"},
        ),
        html.Div(
            [
                dcc.Graph(id="umap-scatter", style={"width": "65%", "display": "inline-block"}),
                html.Div(
                    id="structure-panel",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "10px",
                    },
                ),
            ]
        ),
        dcc.Store(id="embedding-store"),
    ]
)


def _decode_upload(contents):
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def _mol_to_image_src(mol):
    img = Draw.MolToImage(mol, size=(300, 300))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.callback(
    Output("embedding-store", "data"),
    Output("status-message", "children"),
    Output("color-by-dropdown", "options"),
    Output("color-by-dropdown", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def handle_upload(contents, filename):
    if contents is None:
        return no_update, no_update, no_update, no_update

    try:
        content_bytes = _decode_upload(contents)
        records = parse_molecules(filename, content_bytes)
        if not records:
            raise ValueError("No valid molecules could be parsed from this file.")

        mols = [r["mol"] for r in records]
        fingerprints = compute_fingerprints(mols)
        coords = compute_umap(fingerprints)
        clusters = compute_clusters(fingerprints)
        descriptors = compute_descriptors(mols)

        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "name": [r["name"] for r in records],
                "smiles": [r["smiles"] for r in records],
                "cluster": clusters.astype(str),
                **descriptors,
            }
        )
        extra_cols = sorted(
            {k for r in records for k in r if k not in ("mol", "name", "smiles")}
        )
        for col in extra_cols:
            df[col] = [r.get(col) for r in records]

        MOL_CACHE.clear()
        MOL_CACHE.update({r["smiles"]: r["mol"] for r in records})

        color_options = [{"label": "cluster", "value": "cluster"}] + [
            {"label": c, "value": c} for c in df.columns if c not in ("x", "y", "name", "smiles", "cluster")
        ]

        status = f"Loaded {len(df)} molecules from {filename}."
        return df.to_dict("records"), status, color_options, "cluster"

    except Exception as exc:  # surfaced to the user, not a real error-handling boundary
        return no_update, f"Failed to load {filename}: {exc}", no_update, no_update


@app.callback(
    Output("umap-scatter", "figure"),
    Input("embedding-store", "data"),
    Input("color-by-dropdown", "value"),
)
def update_scatter(data, color_by):
    if not data:
        return px.scatter(title="Upload a file to begin")

    df = pd.DataFrame(data)
    color_by = color_by if color_by in df.columns else None
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_by,
        hover_name="name",
        hover_data={"smiles": True, "x": False, "y": False},
        custom_data=["name", "smiles"],
        title="UMAP projection of chemical space",
    )
    fig.update_layout(clickmode="event+select")
    return fig


@app.callback(
    Output("structure-panel", "children"),
    Input("umap-scatter", "clickData"),
)
def show_structure(click_data):
    if not click_data:
        return html.P("Click a point to view its structure.")

    point = click_data["points"][0]
    smiles = point["customdata"][1]
    name = point["customdata"][0]
    mol = MOL_CACHE.get(smiles) or Chem.MolFromSmiles(smiles)
    if mol is None:
        return html.P(f"Could not render structure for {smiles}")

    return html.Div(
        [
            html.H4(name),
            html.Img(src=_mol_to_image_src(mol), style={"width": "100%"}),
            html.P(smiles, style={"wordBreak": "break-all", "fontSize": "12px"}),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
