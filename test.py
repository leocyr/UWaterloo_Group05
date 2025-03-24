import requests
import pandas as pd

#Collision_url = 'https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Traffic_Collisions/FeatureServer/0/query?where=ACCIDENT_YEAR=2020&outFields=*&returnGeometry=false&outSR=4326&f=json'

base_url = "https://services1.arcgis.com/qAo1OsXi67t7XgmS/arcgis/rest/services/Traffic_Collisions/FeatureServer/0/query"
CollisionList = []
for year in range(2020, 2025):
    Collision_url = f"{base_url}?where=ACCIDENT_YEAR={year}&outFields=*&returnGeometry=false&outSR=4326&f=json"
    c = requests.get(Collision_url)
    y = c.json()
    for record in y["features"]:
        CollisionList.append(record["attributes"])

dfcollisions_raw = pd.DataFrame(CollisionList)
print(dfcollisions_raw)


# response = requests.get(url, params=params)

# if response.status_code == 200:
#     data = response.json()
#     print(data)
# else:
#     print("Failed to fetch data:", response.status_code)


