from main import graph

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph_visualization2.png", "wb") as f:
    f.write(png_bytes)