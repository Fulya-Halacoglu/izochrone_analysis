import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union
import os
import pandas as pd

# -------------------------
# 1. Verileri Yükle
# -------------------------
border = gpd.read_file("15min_Urla_Veriler/border.shp")
edges = gpd.read_file("15min_Urla_Veriler/street_segment_urla.shp")
nodes = gpd.read_file("15min_Urla_Veriler/nodes.shp")
healthcare = gpd.read_file("15min_Urla_Veriler/healthcare_points.shp")
ek_hastane = gpd.read_file("15min_Urla_Veriler/hastane_nokta.shp")
buildings = gpd.read_file("15min_Urla_Veriler/bina.shp")

# -------------------------
# 2. CRS Ayarı
# -------------------------
target_crs = "EPSG:32635"
for gdf in [border, edges, nodes, healthcare, ek_hastane, buildings]:
    gdf.to_crs(target_crs, inplace=True)

# -------------------------
# 3. Temizlik
# -------------------------
edges = edges[edges.geometry.notnull()]
nodes = nodes[nodes.geometry.notnull()]
healthcare = healthcare[healthcare.geometry.notnull()]
ek_hastane = ek_hastane[ek_hastane.geometry.notnull()]
buildings = buildings[buildings.geometry.notnull()]

# -------------------------
# 4. Graph Oluştur
# -------------------------
G = nx.Graph()
for idx, row in nodes.iterrows():
    G.add_node(idx, coords=(row.geometry.x, row.geometry.y), geometry=row.geometry)
for idx, row in edges.iterrows():
    line = row.geometry
    if line.geom_type == "MultiLineString":
        line = list(line.geoms)[0]
    if line.geom_type == "LineString":
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        s_idx = nodes.distance(start).idxmin()
        e_idx = nodes.distance(end).idxmin()
        G.add_edge(s_idx, e_idx, weight=line.length)

# -------------------------
# 5. Hastane Noktalarını Birleştir
# -------------------------
healthcare = pd.concat([healthcare, ek_hastane], ignore_index=True)

# -------------------------
# 6. Nearest Node Eşle
# -------------------------
def find_nearest_node(point, nodes_gdf):
    nearest_geom = nearest_points(point, nodes_gdf.unary_union)[1]
    return nodes_gdf.distance(nearest_geom).idxmin()
healthcare["nearest_node"] = healthcare.geometry.apply(lambda x: find_nearest_node(x, nodes))

# -------------------------
# 7. Isochrone Zonları Oluştur (1200 m)
# -------------------------
iso_distance = 1200
iso_union = []
for node_id in healthcare["nearest_node"]:
    lengths = nx.single_source_dijkstra_path_length(G, node_id, cutoff=iso_distance, weight="weight")
    reachable_nodes = list(lengths.keys())
    reachable_geoms = [G.nodes[n]["geometry"] for n in reachable_nodes]
    if reachable_geoms:
        gdf = gpd.GeoDataFrame(geometry=reachable_geoms, crs=target_crs)
        hull = gdf.unary_union.convex_hull
        clipped = hull.intersection(border.unary_union)
        iso_union.append(clipped)

reachable_area = unary_union(iso_union)

# -------------------------
# 8. Bina Analizi
# -------------------------
buildings["accessible"] = buildings.geometry.apply(lambda geom: geom.intersects(reachable_area))

# -------------------------
# 9. Görselleştirme
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))
edges.plot(ax=ax, color="lightgray", linewidth=0.5)
border.boundary.plot(ax=ax, color="black", linewidth=1)
gpd.GeoSeries(reachable_area).plot(ax=ax, color="lightblue", alpha=0.4, label="15 Dakika Alanı")
healthcare.plot(ax=ax, color="red", markersize=20, label="Sağlık Noktaları")
buildings[buildings["accessible"]].plot(ax=ax, color="green", markersize=2, label="Erişilebilen Binalar")
buildings[~buildings["accessible"]].plot(ax=ax, color="gray", markersize=2, label="Erişilemeyen Binalar")

minx, miny, maxx, maxy = border.total_bounds
ax.set_xlim(minx - 200, maxx + 200)
ax.set_ylim(miny - 200, maxy + 200)

legend_handles = [
    mpatches.Patch(color="lightblue", label="15 Dakika Alanı"),
    mlines.Line2D([], [], color="red", marker='o', linestyle='None', markersize=8, label="Sağlık Noktaları"),
    mlines.Line2D([], [], color="green", marker='o', linestyle='None', markersize=5, label="Erişilebilen Binalar"),
    mlines.Line2D([], [], color="gray", marker='o', linestyle='None', markersize=5, label="Erişilemeyen Binalar"),
    mlines.Line2D([], [], color="black", linestyle='-', linewidth=1, label="Çalışma Alanı Sınırı"),
]
plt.legend(handles=legend_handles, loc="upper left")
plt.title("15 Dakikada Sağlık Hizmetine Erişim (Urla)")
plt.axis("off")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/hastane_15min_accessibility_map.png", dpi=300)
plt.show()
