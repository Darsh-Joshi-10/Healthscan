import requests

def get_location():
    try:
        # Get IP address location
        response = requests.get('https://ipinfo.io')
        data = response.json()
        
        # Extract location details
        city = data.get('city')
        region = data.get('region')
        country = data.get('country')
        loc = data.get('loc')  # Latitude and Longitude
        
        return f"Location: {city}, {region}, {country} (Coordinates: {loc})"
    except Exception as e:
        return f"Error: {e}"

print(get_location())
