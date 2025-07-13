from typing import Optional
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import time
import os
import pytz
from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def search(query: str):
    """
    Search the web using Tavily.
    """
    tavily_tool = TavilySearchResults(
        tavily_api_key=os.environ["tavily_api_key"], max_results=5
    )
    results = tavily_tool.invoke(query)
    return results


@tool
def get_current_time(location: Optional[str] = None) -> str:
    """
    Get the current time:
    - If local time in current region is needed, put location as None or empty, and it will return the current local time.
    - If a specific location is provided, put the location in the input_args in it will return the current time in that location.
    """
    print(f"Location provided: {location}")
    if location:
        geolocator = Nominatim(user_agent="langgraph_timezone_agent")
        location = geolocator.geocode(location)
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude)
        utc_now = datetime.utcnow()
        target_timezone = pytz.timezone(timezone)
        local_time = utc_now.replace(tzinfo=pytz.utc).astimezone(target_timezone)
        return (
            f"Current time in {timezone} is {local_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    return (
        f"Current local time is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )

@tool
def use_google_map(position_1: str, position_2: str, mode: str, transit_mode: str = None) -> str:
    """
    Use Google Maps to find the distance between two places and estimate the time it takes to travel between two places.

    Args:
    position_1: Name of starting position.
    position_2: Name of destination position.
    mode: Transportation mode - "driving", "walking", "bicycling", or "transit".
    transit_mode: Optional. Only used when mode is 'transit'
                Optional: 'bus', 'subway', 'train', 'tram', 'rail', or combinations like 'bus,subway'
    """
    import googlemaps
    import os
    from datetime import datetime  
    gmaps = googlemaps.Client(key=os.environ["google_maps_api_key"])
    params = {
        "origins": position_1,
        "destinations": position_2,
        "mode": mode,
        "departure_time": datetime.now(),
    }
    if mode == "transit":
        departure_time = datetime.now()
        if transit_mode:
             if isinstance(transit_mode, str):
                transit_modes = [m.strip() for m in transit_mode.split(",")]
            else:
                transit_modes = [transit_mode]
            
            valid_modes = ['bus', 'subway', 'train', 'tram', 'rail']
            transit_modes = [m for m in transit_modes if m in valid_modes]

            params['departure_time'] = departure_time
            params['transit_mode'] = transit_modes
    elif mode == 'driving':
        params['departure_time'] = departure_time
    distance_result = gmaps.distance_matrix(**params)
    distance = distance_result['rows'][0]['elements'][0]['distance']['text']
    duration = distance_result['row'][0]['elements'][0]['duration']['text']
    return f"Distance from {position_1} to {position_2} is {distance} and it takes approximately {duration} to travel by {mode}."
