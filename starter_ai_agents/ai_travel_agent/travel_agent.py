from textwrap import dedent
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.serpapi import SerpApiTools
from agno.models.deepseek import DeepSeek
import streamlit as st
import re
# 地图距离工具（地理编码 + 路网距离）拆分在同目录的 map_tools.py，本文件只负责 UI 与 Agent 编排
from map_tools import estimate_travel_distance
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


# ---------------------------------------------------------------
# Researcher 结果缓存：同一个 (目的地, 天数) 在 TTL 内只真正调一次搜索 API
#   1) st.cache_data 是 Streamlit 内置的，不需要任何外部依赖
#   2) 缓存 key 只由 destination 和 num_days 决定：参数名以 "_" 开头（_researcher）
#      时 Streamlit 不会把它算进缓存 key，所以 Agent 每次 rerun 重建也不会让缓存失效
#   3) ttl=24*60*60 秒 = 24 小时；show_spinner=False 是为了复用页面已有的 spinner，避免双圈
#   4) 函数体内只有 researcher.run，不含任何 st. 开头的 UI 代码（返回值是 str，可安全缓存）
# ---------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_research(_researcher, destination: str, num_days: int) -> str:
    """执行 Researcher 调研，返回正文文本；相同 destination + num_days 命中 24 小时缓存。"""
    research_results: RunOutput = _researcher.run(
        f"Research {destination} for a {num_days} day trip", stream=False
    )
    return research_results.content

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

# 输入框无条件渲染：即使 Key 没配好，目的地 / 天数 / 预算 / 偏好 也一定看得到
destination = st.text_input("Where do you want to go?")
num_days = st.number_input(
    "How many days do you want to travel for?", min_value=1, max_value=30, value=7
)

# 预算：金额 + 币种，会作为成本上限传给 Planner
budget_col, currency_col = st.columns(2)
with budget_col:
    budget = st.number_input(
        "What's your total budget for this trip?",
        min_value=0,
        max_value=1000000,
        value=5000,
        step=500,
    )
with currency_col:
    currency = st.selectbox(
        "Currency", ["CNY", "USD", "EUR", "GBP", "JPY"], index=0
    )

# 兴趣偏好：常用标签多选 + 自由文本补充，一起传给 Planner
INTEREST_OPTIONS = [
    "Food & Dining",
    "Culture & History",
    "Nature & Outdoors",
    "Museums & Art",
    "Shopping",
    "Nightlife",
    "Adventure & Sports",
    "Relaxation & Spa",
    "Photography",
    "Family-friendly",
]
interests = st.multiselect(
    "What are your interests and preferences?",
    INTEREST_OPTIONS,
    default=["Food & Dining", "Culture & History"],
)
extra_preferences = st.text_input(
    "Anything else we should know? (optional, e.g. vegetarian, no early mornings, wheelchair access)"
)

# 把预算和兴趣整理成 Planner 易读的文本
budget_text = (
    f"{int(budget)} {currency}" if budget > 0 else "No specific budget limit"
)
interests_text = ", ".join(interests) if interests else "No specific preference"
if extra_preferences.strip():
    interests_text = f"{interests_text}; {extra_preferences.strip()}"

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
        role="Generates a draft itinerary based on user preferences, budget, and research results",
        # model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        model=DeepSeek(id="deepseek-chat", api_key=deepseek_api_key),
        description=dedent(
            """\
        You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, a total budget,
        the traveller's interests and preferences, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, a total budget, the traveller's interests and preferences, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Never exceed the stated total budget: estimate the cost of transport, accommodation, food and activities, show a day-by-day cost breakdown plus the estimated total, and keep that total within the budget.",
            "Prioritise activities and accommodations that match the traveller's stated interests and preferences, and briefly explain how each day reflects them.",
            "Before you decide which attractions go on the same day, call `estimate_travel_distance` for each pair of neighbouring stops (pass the city name as `near` to disambiguate) and use the returned distance and travel time to group nearby places together; never make a day zig-zag back and forth across the city.",
            "Treat a one-way leg longer than about 60 minutes as a signal to move that stop to another day or replace it with something closer, and say so briefly when you keep it anyway.",
            "If a tool answers NOT_FOUND or ERROR, use a more complete place name (for example add the city or country) once; if it still fails, continue planning without that data and keep the day's route geographically sensible.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        # 挂载地图距离工具：Agno 会把函数名、docstring 和类型注解转成工具描述与参数 schema
        tools=[estimate_travel_distance],
        add_datetime_to_context=True,
    )

    with st.spinner("Researching your destination..."):
        try:
            # 替换原来的 researcher.run(...)：改调缓存函数
            # 相同 (destination, num_days) 在 24 小时内直接返回缓存，不再请求 API
            # destination.strip() + int(num_days) 让缓存 key 更稳定（忽略首尾空格、避免 int/float 差异）
            research_content = cached_research(
                researcher, destination.strip(), int(num_days)
            )
            st.write("Research completed")
        except Exception as e:
            st.error(f"研究阶段出错：{e}")
            st.stop()

    with st.spinner("Creating your personalized itinerary..."):
        # 预算 > 0 时才强制成本约束，否则只按兴趣偏好定制
        if budget > 0:
            closing_instruction = (
                "Please create a detailed itinerary based on this research. "
                "Keep the total estimated cost within the stated budget and tailor "
                "the activities and accommodations to the traveller's interests and preferences."
            )
        else:
            closing_instruction = (
                "Please create a detailed itinerary based on this research. "
                "Tailor the activities and accommodations to the traveller's interests and preferences."
            )
        prompt = f"""
        Destination: {destination}
        Duration: {num_days} days
        Budget: {budget_text}
        Interests & Preferences: {interests_text}
        Research Results: {research_content}

        {closing_instruction}
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
