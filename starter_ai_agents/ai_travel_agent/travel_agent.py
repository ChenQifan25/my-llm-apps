from textwrap import dedent
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.serpapi import SerpApiTools
from agno.models.deepseek import DeepSeek
import streamlit as st
import re
from agno.models.openai import OpenAIChat
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import os

def generate_ics_content(plan_text:str, start_date: datetime = None) -> bytes:
    """
        Generate an ICS calendar file from a travel itinerary text.

        Args:
            plan_text: The travel itinerary text
            start_date: Optional start date for the itinerary (defaults to today)

        Returns:
            bytes: The ICS file content as bytes
        """
    cal = Calendar()
    cal.add('prodid','-//AI Travel Planner//github.com//' )
    cal.add('version', '2.0')

    if start_date is None:
        start_date = datetime.today()

    # Split the plan into days
    day_pattern = re.compile(r'Day (\d+)[:\s]+(.*?)(?=Day \d+|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)

    if not days: # If no day pattern found, create a single all-day event with the entire content
        event = Event()
        event.add('summary', "Travel Itinerary")
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)  
    else:
        # Process each day
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)
            
            # Create a single event for the entire day
            event = Event()
            event.add('summary', f"Day {day_num} Itinerary")
            event.add('description', day_content.strip())
            
            # Make it an all-day event
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)

    return cal.to_ical()

# Set up the Streamlit app
st.title("AI Travel Planner for chen")
st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using GPT-4o")

# Initialize session state to store the generated itinerary
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None

# ---------------------------------------------------------------
# API Keys: 从环境变量读取（必须是启动 streamlit 的那个进程能读到）
#   当前终端临时设置:  $env:DEEPSEEK_API_KEY="sk-..."; $env:SERP_API_KEY="..."
#   永久设置:          setx DEEPSEEK_API_KEY "sk-..."  -> 之后必须重开终端 / VS Code
# ---------------------------------------------------------------
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
serp_api_key = os.environ.get("SERP_API_KEY")

# 输入框无条件渲染：即使 Key 没配好，目的地 / 天数 也一定看得到
destination = st.text_input("Where do you want to go?")
num_days = st.number_input(
    "How many days do you want to travel for?", min_value=1, max_value=30, value=7
)

missing_keys = [
    name
    for name, value in (
        ("DEEPSEEK_API_KEY", deepseek_api_key),
        ("SERP_API_KEY", serp_api_key),
    )
    if not value
]
if missing_keys:
    st.warning(
        "未检测到环境变量：" + "、".join(missing_keys) + "。"
        "可以填写目的地和天数，但必须先配置 Key 才能生成行程。"
        "请在启动 streamlit 的同一终端执行：$env:DEEPSEEK_API_KEY='sk-...'; $env:SERP_API_KEY='...' ，然后重启应用；"
        "若用 setx 设置过，请重开终端 / VS Code（旧进程读不到新变量）。"
    )

col1, col2 = st.columns(2)

with col1:
    generate_clicked = st.button(
        "Generate Itinerary",
        disabled=bool(missing_keys) or not destination.strip(),
    )

if generate_clicked:
    researcher = Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations based on user preferences",
        # model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        model=DeepSeek(id="deepseek-chat", api_key=deepseek_api_key),
        description=dedent(
            """\
        You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
        ),
        instructions=[
            "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_context=True,
    )
    planner = Agent(
        name="Planner",
        role="Generates a draft itinerary based on user preferences and research results",
        # model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        model=DeepSeek(id="deepseek-chat", api_key=deepseek_api_key),
        description=dedent(
            """\
        You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_context=True,
    )

    with st.spinner("Researching your destination..."):
        try:
            research_results: RunOutput = researcher.run(
                f"Research {destination} for a {num_days} day trip", stream=False
            )
            st.write("Research completed")
        except Exception as e:
            st.error(f"研究阶段出错：{e}")
            st.stop()

    with st.spinner("Creating your personalized itinerary..."):
        prompt = f"""
        Destination: {destination}
        Duration: {num_days} days
        Research Results: {research_results.content}

        Please create a detailed itinerary based on this research.
        """
        try:
            response: RunOutput = planner.run(prompt, stream=False)
        except Exception as e:
            st.error(f"行程生成出错：{e}")
            st.stop()
        # Store the response in session state
        st.session_state.itinerary = response.content
        st.write(response.content)

# Only show download button if there's an itinerary
with col2:
    if st.session_state.itinerary:
        # Generate the ICS file
        ics_content = generate_ics_content(st.session_state.itinerary)
        
        # Provide the file for download
        st.download_button(
            label="Download Itinerary as Calendar (.ics)",
            data=ics_content,
            file_name="travel_itinerary.ics",
            mime="text/calendar"
        )
