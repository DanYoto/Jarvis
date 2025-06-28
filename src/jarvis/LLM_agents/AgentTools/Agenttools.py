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
    tavily_tool = TavilySearchResults(tavily_api_key=os.environ["tavily_api_key"], max_results=5)
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
        return f"Current time in {timezone} is {local_time.strftime('%Y-%m-%d %H:%M:%S')}"    
    return f"Current time in {location} is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
