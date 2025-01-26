def df_to_geojson(df, lat_col, lon_col):
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row[lon_col], row[lat_col]]
            },
            "properties": {}
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}
