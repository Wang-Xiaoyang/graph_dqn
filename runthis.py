import folium
from folium import plugins
import method
import random

# Create a map
Map = folium.Map(location=[51.46, -2.6],
                 zoom_start=14,
                 control_scale=True,
                 attr='default')


def lat():
    """
    Generate latitude
    :return: Latitude
    """
    global Lat
    Lat = random.uniform(51.4438, 51.4707)
    return Lat


def lng():
    """
    Generate longitude
    :return: Longitude
    """
    global Lng
    Lng = random.uniform(-2.6281, -2.5612)
    return Lng


# draw
# 标记一个实心圆
def DU(Lat, Lng):
    """
    Defining the DU symbol
    :param Lat: Latitude of DU
    :param Lng: Longitude of DU
    """
    folium.CircleMarker(
        location=[Lat, Lng],
        radius=6,
        popup='DU',
        color='green',  # 圈的颜色
        fill=True,
        fill_color='green'  # 填充颜色
    ).add_to(Map)


def RU(Lat, Lng):
    """
    Defining the RU symbol
    :param Lat: Latitude of RU
    :param Lng: Longitude of RU
    """
    folium.CircleMarker(
        location=[Lat, Lng],
        radius=4,
        popup='RU',
        color='blue',  # 圈的颜色
        fill=True,
        fill_color='blue'  # 填充颜色
    ).add_to(Map)


Loc_RU = []
Loc_DU = []
Loc_RU1 = []
Loc_DU1 = []
# Number of DU
for i in range(0, 2):
    Loc_DU.append(1)
    Loc_DU1.append(1)
    Lat = lat()
    Lng = lng()
    DU(Lat, Lng)
    Loc_DU[i] = [Lat, Lng]
    Loc_DU1[i] = [Lng, Lat]
# Number of RU
for i in range(0, 7):
    Loc_RU.append(1)
    Loc_RU1.append(1)
    Lat = lat()
    Lng = lng()
    RU(Lat, Lng)
    Loc_RU[i] = [Lat, Lng]
    Loc_RU1[i] = [Lng, Lat]


# Build the required parameters for drawing
times = [{"dates": ["2021-07-21T00:10:00", "2021-07-21T00:20:00"]},
         {"dates": ["2021-07-21T00:20:00", "2021-07-21T00:30:00"]},
         {"dates": ["2021-07-21T00:30:00", "2021-07-21T00:40:00"]},
         {"dates": ["2021-07-21T00:40:00", "2021-07-21T00:50:00"]},
         {"dates": ["2021-07-21T00:50:00", "2021-07-21T01:00:00"]},
         {"dates": ["2021-07-21T01:00:00", "2021-07-21T01:10:00"]},
         {"dates": ["2021-07-21T01:10:00", "2021-07-21T01:20:00"]},
         {"dates": ["2021-07-21T01:20:00", "2021-07-21T01:30:00"]},
         {"dates": ["2021-07-21T01:30:00", "2021-07-21T01:40:00"]}]
lines = []
line1 = []
line2 = []
for k in range(0, 9):
    # for each time step - give a solution
    lines.append(1)
    x, y, Num_1, Num_2 = method.solution(Loc_DU, Loc_RU)
    # print(x, y, Num_1, Num_2)
    M1 = x.copy()
    M2 = y.copy()
    print(M1, M2)
    m1 = len(x)
    m2 = len(y)
    for i in range(len(x) - 1, -1, -1):
        if x[i] == 0:
            x.remove(0)
    for i in range(len(y) - 1, -1, -1):
        if y[i] == 0:
            y.remove(0)
    l1 = len(x)
    l2 = len(y)
    for i in range(0, l1):
        # lines for DU1
        # t = 0
        # for j in range(0, m1):
        #     if M1[j] == 1:
        line1.append(1)
        num = Num_1[i]
        a = times[k].copy()
        a["coordinates"] = Loc_DU1[0], Loc_RU1[num-1]
        a["color"] = "red"
        a["weight"] = 5
        line1[i] = a.copy()
    for i in range(0, l2):
        # lines for DU2
        line2.append(1)
        num = Num_2[i]
        b = times[k].copy()
        b["coordinates"] = Loc_DU1[1], Loc_RU1[num-1]
        b["color"] = "blue"
        b["weight"] = 5
        line2[i] = b.copy()
    LINE = line1 + line2
    lines += LINE
    k += 1
    
for i in range(len(lines) - 1, -1, -1):
    if lines[i] == 1:
        lines.remove(1)
# print(lines)
features = [
    {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": line["coordinates"],
        },
        "properties": {
            "times": line["dates"],
            "style": {
                "color": line["color"],
                "weight": line["weight"],
            },
        },
    }
    for line in lines
]
# features.append(
#     {
#         "type": "Feature",
#         "geometry": {
#             "type": "LineString",
#             "coordinates": [Loc_RU1],
#         },
#         "properties": {
#             "popup": "Current address",
#             "times": [
#                 "2017-06-02",
#                 "2017-07-02",
#                 "2017-08-02",
#                 "2017-09-02",
#                 "2017-10-02",
#             ],
#             "icon": "circle",
#             "iconstyle": {
#                 "fillColor": "green",
#                 "fillOpacity": 0.6,
#                 "stroke": "false",
#                 "radius": 13,
#             },
#             "style": {"weight": 0},
#             "id": "man",
#         },
#     }
# )
plugins.TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1M",
    duration="PT20M",
    add_last_point=False,
    max_speed=1,
).add_to(Map)

Map.save("map3.html")
print("over")
