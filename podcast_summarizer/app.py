import os
import streamlit as st
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from opik.integrations.crewai import track_crewai
from opik.integrations.openai import track_openai
import opik
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
import whisper
import yt_dlp

load_dotenv()

opik.configure(use_local=False, api_key=os.getenv("OPIK_API_KEY"))
openai_client = track_openai(OpenAI())
track_crewai(project_name="Podcast-Startup-Summarizer")

class WeaviateSearchInput(BaseModel):
    topic: str

class CustomWeaviateTool(BaseTool):
    name: str = "Weaviate Insights Search Tool"
    description: str = "Searches Weaviate for relevant insights related to a given topic."
    args_schema: Type[BaseModel] = WeaviateSearchInput

    @opik.track
    def _run(self, topic: str) -> str:
        try:
            client = weaviate.connect_to_wcs(
                cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
            )

            collection = client.collections.get(name="PodcastInsights")

            results = collection.query.near_text(
                query=topic,
                limit=3,
                return_properties=["insight"]
            )

            if not results.objects:
                return "No insights found in Weaviate."

            return "\n".join(f"💡 {obj.properties['insight']}" for obj in results.objects)

        except Exception as e:
            return f"❌ Error querying Weaviate: {e}"




class PodcastAgentsAndTasks:
    def __init__(self, transcript_text):
        self.transcript_text = transcript_text
        self.weaviate_tool = CustomWeaviateTool()
        self.insight_extractor_agent = self.create_insight_extractor_agent()
        self.resource_recommender_agent = self.create_resource_recommender_agent()

    def create_insight_extractor_agent(self) -> Agent:
        return Agent(
            role="Insight Extractor",
            goal="Extract startup ideas, trends, actionable insights, and investor mentions from the podcast transcript.",
            backstory="An AI agent skilled in analyzing podcast transcripts to identify key entrepreneurial insights.",
            verbose=True,
            model="gpt-4o-mini",
        )

    def create_resource_recommender_agent(self) -> Agent:
        return Agent(
            role="Resource Recommender",
            goal="Provide additional resources based on extracted insights.",
            backstory="An AI agent proficient in recommending relevant resources using semantic search.",
            tools=[self.weaviate_tool],
            verbose=True,
            model="gpt-4o-mini",
        )

    def create_insight_extraction_task(self) -> Task:
        return Task(
            name="Insight Extraction",
            description=f"Analyze the following transcript and identify startup ideas, trends, actionable insights, and investor mentions:\n\n{self.transcript_text[:3000]}",
            expected_output="A structured summary with categorized insights.",
            agent=self.insight_extractor_agent,
        )

    def create_resource_recommendation_task(self) -> Task:
        return Task(
            name="Resource Recommendation",
            description="Use the Weaviate tool to find resources related to the extracted insights.",
            expected_output="A list of recommended resources with brief descriptions.",
            agent=self.resource_recommender_agent,
        )

    def create_podcast_crew(self) -> Crew:
        return Crew(
            agents=[self.insight_extractor_agent, self.resource_recommender_agent],
            tasks=[self.create_insight_extraction_task(), self.create_resource_recommendation_task()],
            process=Process.sequential,
            verbose=True,
        )

st.set_page_config(page_title="Podcast to Startup Insights", layout="centered")
st.title("🎧 Podcast to Startup Insights")

youtube_url = st.text_input("Enter the YouTube URL of the podcast episode:")

if st.button("Analyze Podcast") and youtube_url:
    st.info("Downloading and transcribing podcast... ⏳")

    try:
        # Download YouTube audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloaded_podcast.%(ext)s',
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Transcribe audio
        model = whisper.load_model("base")
        result = model.transcribe("downloaded_podcast.mp3")
        transcript_text = result["text"]

        # Run CrewAI agents on transcript
        workflow = PodcastAgentsAndTasks(transcript_text)
        crew = workflow.create_podcast_crew()
        result = crew.kickoff()

        st.success("✅ Analysis complete!")
        st.markdown("### Extracted Insights")
        st.markdown(crew.tasks[0].output)

        st.markdown("### Recommended Resources")
        st.markdown(result)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")