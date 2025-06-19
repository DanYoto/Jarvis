from typing import Optional

@tool
def search(query: str):
    import os
    from langchain_community.tools.tavily_search import TavilySearchResults
    """
    Search the web using Tavily.
    """
    tavily_tool = TavilySearchResults(tavily_api_key=os.environ["tavily_api_key"], max_results=5)
    results = tavily_tool.invoke(query)
    return results

@tool
def get_current_time(location: Optional[str] = None) -> str:
    from geopy.geocoders import Nominatim
    from timezonefinder import TimezoneFinder
    import time

    """
    Get the current time:
    - If local time is needed, put location as None or empty, and it will return the current local time.
    - If a specific location is provided, it will return the current time in that location.
    """
    if location:
        return f"Current time in {location} is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    else:
        geolocator = Nominatim(user_agent="langgraph_timezone_agent")
        location = geolocator.geocode(location)
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude)
        return f"Current time in {timezone} is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"