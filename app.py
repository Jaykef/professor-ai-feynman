# Professor AI Feynman: A Multi-Agent Tool for Learning Anything the Feynman way.
# Jaward Sesay - Microsoft AI Agent Hackathon Submission April 2025
import os
import json
import re
import gradio as gr
import asyncio
import logging
import torch
import zipfile
import shutil
import datetime
from serpapi import GoogleSearch
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage, HandoffMessage, StructuredMessage
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
import traceback
import soundfile as sf
import tempfile
from pydub import AudioSegment
from TTS.api import TTS
import markdown

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lecture_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up environment
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Using output directory: {OUTPUT_DIR}")
os.environ["COQUI_TOS_AGREED"] = "1"

# Initialize TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
logger.info("TTS model initialized on %s", device)

# Define model for slide data
class Slide(BaseModel):
    title: str
    content: str

class SlidesOutput(BaseModel):
    slides: list[Slide]

# Search tool using SerpApi
def search_web(query: str, serpapi_key: str) -> str:
    try:
        params = {
            "q": query,
            "engine": "google",
            "api_key": serpapi_key,
            "num": 5
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            logger.error("SerpApi error: %s", results["error"])
            return None
        
        if "organic_results" not in results or not results["organic_results"]:
            logger.info("No search results found for query: %s", query)
            return None
        
        formatted_results = []
        for item in results["organic_results"][:5]:
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            link = item.get("link", "No link")
            formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
        
        formatted_output = "\n".join(formatted_results)
        logger.info("Successfully retrieved search results for query: %s", query)
        return formatted_output
    
    except Exception as e:
        logger.error("Unexpected error during search: %s", str(e))
        return None

# Custom function to render Markdown to HTML
def render_md_to_html(md_content: str) -> str:
    try:
        html_content = markdown.markdown(md_content, extensions=['extra', 'fenced_code', 'tables'])
        return html_content
    except Exception as e:
        logger.error("Failed to render Markdown to HTML: %s", str(e))
        return "<div>Error rendering content</div>"

# Define create_slides tool for generating HTML slides
def create_slides(slides: list[dict], title: str, output_dir: str = OUTPUT_DIR) -> list[str]:
    try:
        html_files = []
        template_file = os.path.join(os.getcwd(), "slide_template.html")
        with open(template_file, "r", encoding="utf-8") as f:
            template_content = f.read()
        
        for i, slide in enumerate(slides):
            slide_number = i + 1
            md_content = slide['content']
            html_content = render_md_to_html(md_content)
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Replace placeholders in the template
            slide_html = template_content.replace("<!--SLIDE_NUMBER-->", str(slide_number))
            slide_html = slide_html.replace("section title", f"{slide['title']}")
            slide_html = slide_html.replace("Lecture title", title)
            slide_html = slide_html.replace("<!--CONTENT-->", html_content)
            slide_html = slide_html.replace("speaker name", "Prof. AI Feynman")
            slide_html = slide_html.replace("date", date)
            
            html_file = os.path.join(output_dir, f"slide_{slide_number}.html")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(slide_html)
            logger.info("Generated HTML slide: %s", html_file)
            html_files.append(html_file)
        
        # Save slide content as Markdown files
        for i, slide in enumerate(slides):
            slide_number = i + 1
            md_file = os.path.join(output_dir, f"slide_{slide_number}_content.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(slide['content'])
            logger.info("Saved slide content to Markdown: %s", md_file)
        
        return html_files
    
    except Exception as e:
        logger.error("Failed to create HTML slides: %s", str(e))
        return []

# Define helper function for progress HTML
def html_with_progress(label, progress):
    return f"""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
        <div style="width: 70%; background-color: lightgrey; border-radius: 80px; overflow: hidden; margin-bottom: 20px;">
            <div style="width: {progress}%; height: 15px; background-color: #4CAF50; border-radius: 80px;"></div>
        </div>
        <h2 style="font-style: italic; color: #555;">{label}</h2>
    </div>
    """

# Get model client based on selected service
def get_model_client(service, api_key):
    if service == "OpenAI-gpt-4o-2024-08-06":
        return OpenAIChatCompletionClient(model="gpt-4o-2024-08-06", api_key=api_key)
    elif service == "Anthropic-claude-3-sonnet-20240229":
        return AnthropicChatCompletionClient(model="claude-3-sonnet-20240229", api_key=api_key)
    elif service == "Google-gemini-2.0-flash":
        return OpenAIChatCompletionClient(model="gemini-2.0-flash", api_key=api_key)
    elif service == "Ollama-llama3.2":
        return OllamaChatCompletionClient(model="llama3.2")
    elif service == "Azure AI Foundry":
        return AzureAIChatCompletionClient(
            model="phi-4",
            endpoint="https://models.inference.ai.azure.com",
            credential=AzureKeyCredential(os.environ.get("GITHUB_TOKEN", "")),
            model_info={
                "json_output": False,
                "function_calling": False,
                "vision": False,
                "family": "unknown",
                "structured_output": False,
            }
        )
    else:
        raise ValueError("Invalid service")

# Helper function to clean script text
def clean_script_text(script):
    if not script or not isinstance(script, str):
        logger.error("Invalid script input: %s", script)
        return None
    
    script = re.sub(r"\*\*Slide \d+:.*?\*\*", "", script)
    script = re.sub(r"\[.*?\]", "", script)
    script = re.sub(r"Title:.*?\n|Content:.*?\n", "", script)
    script = script.replace("humanlike", "human-like").replace("problemsolving", "problem-solving")
    script = re.sub(r"\s+", " ", script).strip()
    
    if len(script) < 10:
        logger.error("Cleaned script too short (%d characters): %s", len(script), script)
        return None
    
    logger.info("Cleaned script: %s", script)
    return script

# Helper function to validate and convert speaker audio
async def validate_and_convert_speaker_audio(speaker_audio):
    if not speaker_audio or not os.path.exists(speaker_audio):
        logger.warning("Speaker audio file does not exist: %s. Using default voice.", speaker_audio)
        default_voice = os.path.join(os.path.dirname(__file__), "feynman.mp3")
        if os.path.exists(default_voice):
            speaker_audio = default_voice
        else:
            logger.error("Default voice not found. Cannot proceed with TTS.")
            return None
    
    try:
        ext = os.path.splitext(speaker_audio)[1].lower()
        if ext == ".mp3":
            logger.info("Converting MP3 to WAV: %s", speaker_audio)
            audio = AudioSegment.from_mp3(speaker_audio)
            audio = audio.set_channels(1).set_frame_rate(22050)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUT_DIR) as temp_file:
                audio.export(temp_file.name, format="wav")
                speaker_wav = temp_file.name
        elif ext == ".wav":
            speaker_wav = speaker_audio
        else:
            logger.error("Unsupported audio format: %s", ext)
            return None
        
        data, samplerate = sf.read(speaker_wav)
        if samplerate < 16000 or samplerate > 48000:
            logger.error("Invalid sample rate for %s: %d Hz", speaker_wav, samplerate)
            return None
        if len(data) < 16000:
            logger.error("Speaker audio too short: %d frames", len(data))
            return None
        if data.ndim == 2:
            logger.info("Converting stereo WAV to mono: %s", speaker_wav)
            data = data.mean(axis=1)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUT_DIR) as temp_file:
                sf.write(temp_file.name, data, samplerate)
                speaker_wav = temp_file.name
        
        logger.info("Validated speaker audio: %s", speaker_wav)
        return speaker_wav
    
    except Exception as e:
        logger.error("Failed to validate or convert speaker audio %s: %s", speaker_audio, str(e))
        return None

# Helper function to generate audio using Coqui TTS API
def generate_xtts_audio(tts, text, speaker_wav, output_path):
    if not tts:
        logger.error("TTS model not initialized")
        return False
    try:
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_path)
        logger.info("Generated audio for %s", output_path)
        return True
    except Exception as e:
        logger.error("Failed to generate audio for %s: %s", output_path, str(e))
        return False

# Helper function to extract JSON from messages
def extract_json_from_message(message):
    if isinstance(message, TextMessage):
        content = message.content
        logger.debug("Extracting JSON from TextMessage: %s", content)
        if not isinstance(content, str):
            logger.warning("TextMessage content is not a string: %s", content)
            return None
        
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                logger.debug("Found JSON in code block: %s", json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from code block: %s", e)
        
        json_patterns = [
            r"\[\s*\{.*?\}\s*\]",
            r"\{\s*\".*?\"\s*:.*?\}",
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0).strip()
                    logger.debug("Found JSON with pattern %s: %s", pattern, json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON with pattern %s: %s", pattern, e)
        
        try:
            for i in range(len(content)):
                for j in range(len(content), i, -1):
                    substring = content[i:j].strip()
                    if (substring.startswith('{') and substring.endswith('}')) or \
                       (substring.startswith('[') and substring.endswith(']')):
                        try:
                            parsed = json.loads(substring)
                            if isinstance(parsed, (list, dict)):
                                logger.info("Found JSON in substring: %s", substring)
                                return parsed
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("Error in JSON substring search: %s", e)
        
        logger.warning("No JSON found in TextMessage content")
        return None
    
    elif isinstance(message, StructuredMessage):
        content = message.content
        logger.debug("Extracting JSON from StructuredMessage: %s", content)
        try:
            if isinstance(content, BaseModel):
                content_dict = content.dict()
                return content_dict.get("slides", content_dict)
            return content
        except Exception as e:
            logger.error("Failed to extract JSON from StructuredMessage: %s, Content: %s", e, content)
            return None
    
    elif isinstance(message, HandoffMessage):
        logger.debug("Extracting JSON from HandoffMessage context")
        for ctx_msg in message.context:
            if hasattr(ctx_msg, "content"):
                content = ctx_msg.content
                logger.debug("HandoffMessage context content: %s", content)
                if isinstance(content, str):
                    pattern = r"```json\s*(.*?)\s*```"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(1))
                        except json.JSONDecodeError as e:
                            logger.error("Failed to parse JSON from HandoffMessage: %s", e)
                    
                    json_patterns = [
                        r"\[\s*\{.*?\}\s*\]",
                        r"\{\s*\".*?\"\s*:.*?\}",
                    ]
                    
                    for pattern in json_patterns:
                        match = re.search(pattern, content, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(0))
                            except json.JSONDecodeError as e:
                                logger.error("Failed to parse JSON with pattern %s: %s", pattern, e)
                elif isinstance(content, dict):
                    return content.get("slides", content)
        
        logger.warning("No JSON found in HandoffMessage context")
        return None
    
    logger.warning("Unsupported message type for JSON extraction: %s", type(message))
    return None

# Async update audio preview
async def update_audio_preview(audio_file):
    if audio_file:
        logger.info("Updating audio preview for file: %s", audio_file)
        return audio_file
    return None

# Create a zip file of .md, .txt, and .mp3 files
def create_zip_of_files(file_paths):
    zip_path = os.path.join(OUTPUT_DIR, "all_lecture_materials.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                _, ext = os.path.splitext(file_path)
                if ext in ['.md', '.txt', '.mp3']:
                    zipf.write(file_path, os.path.basename(file_path))
                    logger.info("Added %s to zip", file_path)
    logger.info("Created zip file: %s", zip_path)
    return zip_path

# Access local files
def get_gradio_file_url(local_path):
    relative_path = os.path.relpath(local_path, os.getcwd())
    return f"/gradio_api/file={relative_path}"

# Async generate lecture materials and audio
async def on_generate(api_service, api_key, serpapi_key, title, lecture_content_description, lecture_type, speaker_audio, num_slides):
    model_client = get_model_client(api_service, api_key)

    if os.path.exists(OUTPUT_DIR):
        try:
            for item in os.listdir(OUTPUT_DIR):
                item_path = os.path.join(OUTPUT_DIR, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            logger.info("Cleared outputs directory: %s", OUTPUT_DIR)
        except Exception as e:
            logger.error("Failed to clear outputs directory: %s", str(e))
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Created outputs directory: %s", OUTPUT_DIR)
    
    # Total slides include user-specified content slides plus Introduction and Closing slides
    content_slides = num_slides
    total_slides = content_slides + 2 
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    research_agent = AssistantAgent(
        name="research_agent",
        model_client=model_client,
        handoffs=["slide_agent"],
        system_message="You are a Research Agent. Use the search_web tool to gather information on the topic and keywords from the initial message. Summarize the findings concisely in a single message, then use the handoff_to_slide_agent tool to pass the task to the Slide Agent. Do not produce any other output.",
        tools=[search_web]
    )
    slide_agent = AssistantAgent(
        name="slide_agent",
        model_client=model_client,
        handoffs=["script_agent"],
        system_message=f"""
You are a Slide Agent. Using the research from the conversation history and the specified number of content slides ({content_slides}), generate exactly {content_slides} content slides, plus an Introduction slide as the first slide and a Closing slide as the last slide, making a total of {total_slides} slides. 

- The Introduction slide (first slide) should have the title "{title}" and content containing only the lecture title, speaker name (Prof. AI Feynman), and date {date}, centered, in plain text.
- The Closing slide (last slide) should have the title "Closing" and content containing only "The End\nThank you", centered, in plain text.
- The remaining {content_slides} slides should be content slides based on the lecture description and audience type, with meaningful titles and content in valid Markdown format.

Output ONLY a JSON array wrapped in ```json ... ``` in a TextMessage, where each slide is an object with 'title' and 'content' keys. After generating the JSON, use the create_slides tool to produce HTML slides, then use the handoff_to_script_agent tool to pass the task to the Script Agent. Do not include any explanatory text or other messages.

Example output for 1 content slide (total 3 slides):
```json
[
    {{"title": "Introduction to AI Basics", "content": "AI Basics\nProf. AI Feynman\nMay 2nd, 2025"}},
    {{"title": "What is AI?", "content": "# What is AI?\n- Definition: Systems that mimic human intelligence\n- Key areas: ML, NLP, Robotics"}},
    {{"title": "Closing", "content": "The End\nThank you"}}
]
```""",
        tools=[create_slides],
        output_content_type=None,
        reflect_on_tool_use=False
    )
    script_agent = AssistantAgent(
        name="script_agent",
        model_client=model_client,
        handoffs=["feynman_agent"],
        system_message=f"""
You are a Script Agent model after Richard Feynman. Access the JSON array of {total_slides} slides from the conversation history, which includes an Introduction slide, {content_slides} content slides, and a Closing slide. Generate a narration script (1-2 sentences) for each of the {total_slides} slides, summarizing its content in a clear, academically inclined tone, with humour as professor feynman would deliver it. Ensure the lecture is engaging and covers the fundamental requirements of the topic. Overall keep lecture engaging yet highly informative, covering the fundamental requirements of the topic. Output ONLY a JSON array wrapped in ```json ... ``` with exactly {total_slides} strings, one script per slide, in the same order. Ensure the JSON is valid and complete. After outputting, use the handoff_to_feynman_agent tool. If scripts cannot be generated, retry once.

- For the Introduction slide, the script should be a welcoming message introducing the lecture.
- For the Closing slide, the script should be a brief farewell and thank you message.
- For the content slides, summarize the slide content academically.

Example for 3 slides (1 content slide):
```json
[
    "Welcome to the lecture on AI Basics. I am Professor AI Feynman, and today we will explore the fundamentals of artificial intelligence.",
    "Let us begin by defining artificial intelligence: it refers to systems that mimic human intelligence, spanning key areas such as machine learning, natural language processing, and robotics.",
    "That concludes our lecture on AI Basics. Thank you for your attention, and I hope you found this session insightful."
]
```""",
        output_content_type=None,
        reflect_on_tool_use=False
    )
    feynman_agent = AssistantAgent(
        name="feynman_agent",
        model_client=model_client,
        handoffs=[],
        system_message=f"""
You are Agent Feynman. Review the slides and scripts from the conversation history to ensure coherence, completeness, and that exactly {total_slides} slides and {total_slides} scripts are received, including the Introduction and Closing slides. Verify that HTML slide files exist in the outputs directory. Output a confirmation message summarizing the number of slides, scripts, and HTML files status. If slides, scripts, or HTML files are missing, invalid, or do not match the expected count ({total_slides}), report the issue clearly. Use 'TERMINATE' to signal completion.
Example: 'Received {total_slides} slides, {total_slides} scripts, and HTML files. Lecture is coherent. TERMINATE'
""")
    
    swarm = Swarm(
        participants=[research_agent, slide_agent, script_agent, feynman_agent],
        termination_condition=HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
    )
    
    progress = 0
    label = "Research: in progress..."
    yield (
        html_with_progress(label, progress),
        []
    )
    await asyncio.sleep(0.1)
    
    initial_message = f"""
    Lecture Title: {title}
    Lecture Content Description: {lecture_content_description}
    Audience: {lecture_type}
    Number of Content Slides: {content_slides}
    Please start by researching the topic, or proceed without research if search is unavailable.
    """
    logger.info("Starting lecture generation for title: %s with %d content slides (total %d slides)", title, content_slides, total_slides)
    
    slides = None
    scripts = None
    html_files = []
    error_html = """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
        <h2 style="color: #d9534f;">Failed to generate lecture materials</h2>
        <p style="margin-top: 20px;">Please try again with different parameters or a different model.</p>
    </div>
    """
    
    try:
        logger.info("Research Agent starting...")
        if serpapi_key:
            task_result = await Console(swarm.run_stream(task=initial_message))
        else:
            logger.warning("No SerpApi key provided, bypassing research phase")
            task_result = await Console(swarm.run_stream(task=f"{initial_message}\nNo search available, proceed with slide generation."))
        logger.info("Swarm execution completed")
        
        slide_retry_count = 0
        script_retry_count = 0
        max_retries = 2
        
        for message in task_result.messages:
            source = getattr(message, 'source', getattr(message, 'sender', None))
            logger.debug("Processing message from %s, type: %s", source, type(message))
            
            if isinstance(message, HandoffMessage):
                logger.info("Handoff from %s to %s", source, message.target)
                if source == "research_agent" and message.target == "slide_agent":
                    progress = 25
                    label = "Slides: generating..."
                    yield (
                        html_with_progress(label, progress),
                        []
                    )
                    await asyncio.sleep(0.1)
                elif source == "slide_agent" and message.target == "script_agent":
                    if slides is None:
                        logger.warning("Slide Agent handoff without slides JSON")
                        extracted_json = extract_json_from_message(message)
                        if extracted_json:
                            slides = extracted_json
                            logger.info("Extracted slides JSON from HandoffMessage context: %s", slides)
                    if slides is None or len(slides) != total_slides:
                        if slide_retry_count < max_retries:
                            slide_retry_count += 1
                            logger.info("Retrying slide generation (attempt %d/%d)", slide_retry_count, max_retries)
                            retry_message = TextMessage(
                                content=f"Please generate exactly {total_slides} slides (Introduction, {content_slides} content slides, and Closing) as per your instructions.",
                                source="user",
                                recipient="slide_agent"
                            )
                            task_result.messages.append(retry_message)
                            continue
                    progress = 50
                    label = "Scripts: generating..."
                    yield (
                        html_with_progress(label, progress),
                        []
                    )
                    await asyncio.sleep(0.1)
                elif source == "script_agent" and message.target == "feynman_agent":
                    if scripts is None:
                        logger.warning("Script Agent handoff without scripts JSON")
                        extracted_json = extract_json_from_message(message)
                        if extracted_json:
                            scripts = extracted_json
                            logger.info("Extracted scripts JSON from HandoffMessage context: %s", scripts)
                    progress = 75
                    label = "Review: in progress..."
                    yield (
                        html_with_progress(label, progress),
                        []
                    )
                    await asyncio.sleep(0.1)
            
            elif source == "research_agent" and isinstance(message, TextMessage) and "handoff_to_slide_agent" in message.content:
                logger.info("Research Agent completed research")
                progress = 25
                label = "Slides: generating..."
                yield (
                    html_with_progress(label, progress),
                    []
                )
                await asyncio.sleep(0.1)
            
            elif source == "slide_agent" and isinstance(message, (TextMessage, StructuredMessage)):
                logger.debug("Slide Agent message received")
                extracted_json = extract_json_from_message(message)
                if extracted_json:
                    slides = extracted_json
                    logger.info("Slide Agent generated %d slides: %s", len(slides), slides)
                    if len(slides) != total_slides:
                        if slide_retry_count < max_retries:
                            slide_retry_count += 1
                            logger.info("Retrying slide generation (attempt %d/%d)", slide_retry_count, max_retries)
                            retry_message = TextMessage(
                                content=f"Please generate exactly {total_slides} slides (Introduction, {content_slides} content slides, and Closing) as per your instructions.",
                                source="user",
                                recipient="slide_agent"
                            )
                            task_result.messages.append(retry_message)
                            continue
                    # Generate HTML slides
                    html_files = create_slides(slides, title)
                    if not html_files:
                        logger.error("Failed to generate HTML slides")
                    progress = 50
                    label = "Scripts: generating..."
                    yield (
                        html_with_progress(label, progress),
                        []
                    )
                    await asyncio.sleep(0.1)
                else:
                    logger.warning("No JSON extracted from slide_agent message")
                    if slide_retry_count < max_retries:
                        slide_retry_count += 1
                        logger.info("Retrying slide generation (attempt %d/%d)", slide_retry_count, max_retries)
                        retry_message = TextMessage(
                            content=f"Please generate exactly {total_slides} slides (Introduction, {content_slides} content slides, and Closing) as per your instructions.",
                            source="user",
                            recipient="slide_agent"
                        )
                        task_result.messages.append(retry_message)
                        continue
            
            elif source == "script_agent" and isinstance(message, (TextMessage, StructuredMessage)):
                logger.debug("Script Agent message received")
                extracted_json = extract_json_from_message(message)
                if extracted_json:
                    scripts = extracted_json
                    logger.info("Script Agent generated scripts for %d slides: %s", len(scripts), scripts)
                    for i, script in enumerate(scripts):
                        script_file = os.path.join(OUTPUT_DIR, f"slide_{i+1}_script.txt")
                        try:
                            with open(script_file, "w", encoding="utf-8") as f:
                                f.write(script)
                            logger.info("Saved script to %s", script_file)
                        except Exception as e:
                            logger.error("Error saving script to %s: %s", script_file, str(e))
                    progress = 75
                    label = "Scripts generated and saved. Reviewing..."
                    yield (
                        html_with_progress(label, progress),
                        []
                    )
                    await asyncio.sleep(0.1)
                else:
                    logger.warning("No JSON extracted from script_agent message")
                    if script_retry_count < max_retries:
                        script_retry_count += 1
                        logger.info("Retrying script generation (attempt %d/%d)", script_retry_count, max_retries)
                        retry_message = TextMessage(
                            content=f"Please generate exactly {total_slides} scripts for the {total_slides} slides as per your instructions.",
                            source="user",
                            recipient="script_agent"
                        )
                        task_result.messages.append(retry_message)
                        continue
            
            elif source == "feynman_agent" and isinstance(message, TextMessage) and "TERMINATE" in message.content:
                logger.info("Feynman Agent completed lecture review: %s", message.content)
                progress = 90
                label = "Lecture materials ready. Generating lecture speech..."
                file_paths = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.md', '.txt'))]
                file_paths.sort()
                file_paths = [os.path.join(OUTPUT_DIR, f) for f in file_paths]
                yield (
                    html_with_progress(label, progress),
                    file_paths
                )
                await asyncio.sleep(0.1)
        
        logger.info("Slides state: %s", "Generated" if slides else "None")
        logger.info("Scripts state: %s", "Generated" if scripts else "None")
        logger.info("HTML files state: %s", "Generated" if html_files else "None")
        if not slides or not scripts:
            error_message = f"Failed to generate {'slides and scripts' if not slides and not scripts else 'slides' if not slides else 'scripts'}"
            error_message += f". Received {len(slides) if slides else 0} slides and {len(scripts) if scripts else 0} scripts."
            logger.error("%s", error_message)
            logger.debug("Dumping all messages for debugging:")
            for msg in task_result.messages:
                source = getattr(msg, 'source', getattr(msg, 'sender', None))
                logger.debug("Message from %s, type: %s, content: %s", source, type(msg), msg.to_text() if hasattr(msg, 'to_text') else str(msg))
            yield (
                error_html,
                []
            )
            return
        
        if len(slides) != total_slides:
            logger.error("Expected %d slides, but received %d", total_slides, len(slides))
            yield (
                f"""
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #d9534f;">Incorrect number of slides</h2>
                    <p style="margin-top: 20px;">Expected {total_slides} slides, but generated {len(slides)}. Please try again.</p>
                </div>
                """,
                []
            )
            return
        
        if not isinstance(scripts, list) or not all(isinstance(s, str) for s in scripts):
            logger.error("Scripts are not a list of strings: %s", scripts)
            yield (
                f"""
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #d9534f;">Invalid script format</h2>
                    <p style="margin-top: 20px;">Scripts must be a list of strings. Please try again.</p>
                </div>
                """,
                []
            )
            return
        
        if len(scripts) != total_slides:
            logger.error("Mismatch between number of slides (%d) and scripts (%d)", len(slides), len(scripts))
            yield (
                f"""
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #d9534f;">Mismatch in slides and scripts</h2>
                    <p style="margin-top: 20px;">Generated {len(slides)} slides but {len(scripts)} scripts. Please try again.</p>
                </div>
                """,
                []
            )
            return
        
        # Access the generated HTML files
        html_file_urls = [get_gradio_file_url(html_file) for html_file in html_files]
        audio_urls = [None] * len(scripts)
        audio_timeline = ""
        for i in range(len(scripts)):
            audio_timeline += f'<audio id="audio-{i+1}" controls src="" style="display: inline-block; margin: 0 10px; width: 200px;"><span>Loading...</span></audio>'
        
        file_paths = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.md', '.txt'))]
        file_paths.sort()
        file_paths = [os.path.join(OUTPUT_DIR, f) for f in file_paths]
        
        audio_files = []
        validated_speaker_wav = await validate_and_convert_speaker_audio(speaker_audio)
        if not validated_speaker_wav:
            logger.error("Invalid speaker audio after conversion, skipping TTS")
            yield (
                f"""
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #d9534f;">Invalid speaker audio</h2>
                    <p style="margin-top: 20px;">Please upload a valid MP3 or WAV audio file and try again.</p>
                </div>
                """,
                []
            )
            return
        
        for i, script in enumerate(scripts):
            cleaned_script = clean_script_text(script)
            audio_file = os.path.join(OUTPUT_DIR, f"slide_{i+1}.mp3")
            script_file = os.path.join(OUTPUT_DIR, f"slide_{i+1}_script.txt")
            
            try:
                with open(script_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_script or "")
                logger.info("Saved script to %s: %s", script_file, cleaned_script)
            except Exception as e:
                logger.error("Error saving script to %s: %s", script_file, str(e))
            
            if not cleaned_script:
                logger.error("Skipping audio for slide %d due to empty or invalid script", i + 1)
                audio_files.append(None)
                audio_urls[i] = None
                progress = 90 + ((i + 1) / len(scripts)) * 10
                label = f"Generating speech for slide {i + 1}/{len(scripts)}..."
                yield (
                    html_with_progress(label, progress),
                    file_paths
                )
                await asyncio.sleep(0.1)
                continue
            
            max_audio_retries = 2
            for attempt in range(max_audio_retries + 1):
                try:
                    current_text = cleaned_script
                    if attempt > 0:
                        sentences = re.split(r"[.!?]+", cleaned_script)
                        sentences = [s.strip() for s in sentences if s.strip()][:2]
                        current_text = ". ".join(sentences) + "."
                        logger.info("Retry %d for slide %d with simplified text: %s", attempt, i + 1, current_text)
                    
                    success = generate_xtts_audio(tts, current_text, validated_speaker_wav, audio_file)
                    if not success:
                        raise RuntimeError("TTS generation failed")
                    
                    logger.info("Generated audio for slide %d: %s", i + 1, audio_file)
                    audio_files.append(audio_file)
                    audio_urls[i] = get_gradio_file_url(audio_file)
                    progress = 90 + ((i + 1) / len(scripts)) * 10
                    label = f"Generating speech for slide {i + 1}/{len(scripts)}..."
                    file_paths.append(audio_file)  
                    yield (
                        html_with_progress(label, progress),
                        file_paths
                    )
                    await asyncio.sleep(0.1)
                    break
                except Exception as e:
                    logger.error("Error generating audio for slide %d (attempt %d): %s\n%s", i + 1, attempt, str(e), traceback.format_exc())
                    if attempt == max_audio_retries:
                        logger.error("Max retries reached for slide %d, skipping", i + 1)
                        audio_files.append(None)
                        audio_urls[i] = None
                        progress = 90 + ((i + 1) / len(scripts)) * 10
                        label = f"Generating speech for slide {i + 1}/{len(scripts)}..."
                        yield (
                            html_with_progress(label, progress),
                            file_paths
                        )
                        await asyncio.sleep(0.1)
                        break
        
        # Create zip file with all materials except .html files
        zip_file = create_zip_of_files(file_paths)
        file_paths.append(zip_file)
        
        # Slide hack: Render the lecture container with iframe containing HTML slides
        audio_timeline = ""
        for j, url in enumerate(audio_urls):
            if url:
                audio_timeline += f'<audio id="audio-{j+1}" controls src="{url}" style="display: inline-block; margin: 0 10px; width: 200px;"></audio>'
            else:
                audio_timeline += f'<audio id="audio-{j+1}" controls src="" style="display: inline-block; margin: 0 10px; width: 200px;"><span>Audio unavailable</span></audio>'
        
        slides_info = json.dumps({"htmlFiles": html_file_urls, "audioFiles": audio_urls})
        html_output = f"""
        <div id="lecture-data" style="display: none;">{slides_info}</div>
        <div id="lecture-container" style="height: 700px; border: 1px solid #ddd; border-radius: 8px; display: flex; flex-direction: column; justify-content: space-between;">
            <div id="slide-content" style="flex: 1; overflow: auto; padding: 20px; text-align: center; background-color: #fff;">
                <iframe id="slide-iframe" style="width: 100%; height: 100%; border: none;"></iframe>
            </div>
            <div style="padding: 20px; text-align: center;">
                <div style="display: flex; justify-content: center; margin-bottom: 10px;">
                    {audio_timeline}
                </div>
                <div style="display: center; justify-content: center; margin-bottom: 10px;">
                    <button id="prev-btn" style="border-radius: 50%; width: 40px; height: 40px; margin: 0 5px; font-size: 1.2em; cursor: pointer; background-color: lightgrey"><i class="fas fa-step-backward" style="color: #000"></i></button>
                    <button id="play-btn" style="border-radius: 50%; width: 40px; height: 40px; margin: 0 5px; font-size: 1.2em; cursor: pointer; background-color: lightgrey"><i class="fas fa-play" style="color: #000"></i></button>
                    <button id="next-btn" style="border-radius: 50%; width: 40px; height: 40px; margin: 0 5px; font-size: 1.2em; cursor: pointer; background-color: lightgrey"><i class="fas fa-step-forward" style="color: #000"></i></button>
                    <button id="fullscreen-btn" style="border-radius: 50%; width: 40px; height: 40px; margin: 0 5px; font-size: 1.2em; cursor: pointer; background-color: lightgrey"><i style="color: #000" class="fas fa-expand"></i></button>
                </div>
            </div>
        </div>
        """
        logger.info("Yielding final lecture materials after audio generation")
        yield (
            html_output,
            file_paths
        )
        
        logger.info("Lecture generation completed successfully")
    
    except Exception as e:
        logger.error("Error during lecture generation: %s\n%s", str(e), traceback.format_exc())
        yield (
            f"""
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #000;">Error during lecture generation</h2>
                <p style="margin-top: 10px; font-size: 16px;color: #000;">{str(e)}</p>
                <p style="margin-top: 20px;">Please try again</p>
            </div>
            """,
            []
        )
        return

# custom js for lecture container features
js_code = """
() => {
    // Function to wait for an element to appear in the DOM
    function waitForElement(selector, callback, maxAttempts = 50, interval = 100) {
        let attempts = 0;
        const intervalId = setInterval(() => {
            const element = document.querySelector(selector);
            if (element) {
                clearInterval(intervalId);
                console.log(`Element ${selector} found after ${attempts} attempts`);
                callback(element);
            } else if (attempts >= maxAttempts) {
                clearInterval(intervalId);
                console.error(`Element ${selector} not found after ${maxAttempts} attempts`);
            }
            attempts++;
        }, interval);
    }

    // Main initialization function
    function initializeSlides() {
        console.log("Initializing slides...");

        // Wait for lecture-data to load the JSON data
        waitForElement('#lecture-data', (dataElement) => {
            if (!dataElement.textContent) {
                console.error("Lecture data element is empty");
                return;
            }
            let lectureData;
            try {
                lectureData = JSON.parse(dataElement.textContent);
                console.log("Lecture data parsed successfully:", lectureData);
            } catch (e) {
                console.error("Failed to parse lecture data:", e);
                return;
            }

            if (!lectureData.htmlFiles || lectureData.htmlFiles.length === 0) {
                console.error("No HTML files found in lecture data");
                return;
            }

            let currentSlide = 0;
            const totalSlides = lectureData.htmlFiles.length;
            let audioElements = [];
            let isPlaying = false;
            let hasNavigated = false; // Track if user has used prev/next buttons

            // Wait for slide-content element
            waitForElement('#slide-content', (slideContent) => {
                console.log("Slide content element found");

                // Initialize audio elements
                for (let i = 0; i < totalSlides; i++) {
                    const audio = document.getElementById(`audio-${i+1}`);
                    if (audio) {
                        audioElements.push(audio);
                        console.log(`Found audio element audio-${i+1}:`, audio);
                    } else {
                        console.error(`Audio element audio-${i+1} not found`);
                    }
                }

                function renderSlide() {
                    console.log("Rendering slide:", currentSlide + 1);
                    if (currentSlide >= 0 && currentSlide < totalSlides && lectureData.htmlFiles[currentSlide]) {
                        const iframe = document.getElementById('slide-iframe');
                        if (iframe) {
                            iframe.src = lectureData.htmlFiles[currentSlide];
                            console.log("Set iframe src to:", lectureData.htmlFiles[currentSlide]);
                            // Adjust font size based on content length and screen size
                            waitForElement('iframe', (iframe) => {
                                iframe.onload = () => {
                                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                                    const body = doc.body;
                                    if (body) {
                                        const textLength = body.textContent.length;
                                        const screenWidth = window.innerWidth;
                                        // Base font size: 12px max on large screens, scale down to 8px on small screens
                                        let baseFontSize = Math.min(12, Math.max(12, 16 * (screenWidth / 1920))); // Scale with screen width (1920px as reference)
                                        // Adjust inversely with content length
                                        const adjustedFontSize = Math.max(12, baseFontSize * (1000 / (textLength + 100))); // Minimum 8px, scale down with length
                                        const elements = body.getElementsByTagName('*');
                                        for (let elem of elements) {
                                            elem.style.fontSize = `${adjustedFontSize}px`;
                                        }
                                        console.log(`Adjusted font size to ${adjustedFontSize}px for ${textLength} characters on ${screenWidth}px width`);
                                    }
                                };
                            });
                        } else {
                            console.error("Iframe not found");
                        }
                    } else {
                        const iframe = document.getElementById('slide-iframe');
                        if (iframe) {
                            iframe.src = "about:blank";
                            console.log("No valid slide content for index:", currentSlide);
                        }
                    }
                }

                function updateSlide(callback) {
                    console.log("Updating slide to index:", currentSlide);
                    renderSlide();
                    // Pause and reset all audio elements
                    audioElements.forEach(audio => {
                        if (audio && audio.pause) {
                            audio.pause();
                            audio.currentTime = 0;
                            audio.style.border = 'none'; // Reset border
                            console.log("Paused and reset audio:", audio.id);
                        }
                    });
                    // Wait briefly to ensure pause completes before proceeding
                    setTimeout(() => {
                        if (callback) callback();
                    }, 100);
                }

                function updateAudioSources(audioUrls) {
                    console.log("Updating audio sources:", audioUrls);
                    audioUrls.forEach((url, index) => {
                        const audio = audioElements[index];
                        if (audio && url && audio.src !== url) {
                            audio.src = url;
                            audio.load();
                            console.log(`Updated audio-${index+1} src to:`, url);
                        } else if (!audio) {
                            console.error(`Audio element at index ${index} not found`);
                        }
                    });
                }

                function prevSlide() {
                    console.log("Previous button clicked, current slide:", currentSlide);
                    hasNavigated = true; // User has navigated
                    if (currentSlide > 0) {
                        currentSlide--;
                        updateSlide(() => {
                            const audio = audioElements[currentSlide];
                            if (audio && audio.play && isPlaying) {
                                audio.style.border = '5px solid #50f150';
                                audio.style.borderRadius = '30px';
                                audio.play().catch(e => console.error('Audio play failed:', e));
                            }
                        });
                    } else {
                        console.log("Already at first slide");
                    }
                }

                function nextSlide() {
                    console.log("Next button clicked, current slide:", currentSlide);
                    hasNavigated = true; // User has navigated
                    if (currentSlide < totalSlides - 1) {
                        currentSlide++;
                        updateSlide(() => {
                            const audio = audioElements[currentSlide];
                            if (audio && audio.play && isPlaying) {
                                audio.style.border = '5px solid #50f150';
                                audio.style.borderRadius = '30px';
                                audio.play().catch(e => console.error('Audio play failed:', e));
                            }
                        });
                    } else {
                        console.log("Already at last slide");
                    }
                }

                function playAll() {
                    console.log("Play button clicked, isPlaying:", isPlaying);
                    const playBtn = document.getElementById('play-btn');
                    if (!playBtn) {
                        console.error("Play button not found");
                        return;
                    }
                    const playIcon = playBtn.querySelector('i');
                    if (playIcon.className.includes('fa-pause')) {
                        // Pause playback
                        isPlaying = false;
                        audioElements.forEach(audio => {
                            if (audio && audio.pause) {
                                audio.pause();
                                audio.currentTime = 0;
                                audio.style.border = 'none';
                                console.log("Paused audio:", audio.id);
                            }
                        });
                        playIcon.className = 'fas fa-play';
                        return;
                    }
                    // Start playback
                    currentSlide = 0;
                    let index = 0;
                    isPlaying = true;
                    playIcon.className = 'fas fa-pause';
                    updateSlide(() => {
                        function playNext() {
                            if (index >= totalSlides || !isPlaying) {
                                isPlaying = false;
                                playIcon.className = 'fas fa-play';
                                audioElements.forEach(audio => {
                                    if (audio) audio.style.border = 'none';
                                });
                                console.log("Finished playing all slides or paused");
                                return;
                            }
                            currentSlide = index;
                            updateSlide(() => {
                                const audio = audioElements[index];
                                if (audio && audio.play) {
                                    // Highlight the current audio element
                                    audioElements.forEach(a => a.style.border = 'none');
                                    audio.style.border = '5px solid #16cd16';
                                    audio.style.borderRadius = '30px';
                                    console.log(`Attempting to play audio for slide ${index + 1}`);
                                    audio.play().then(() => {
                                        console.log(`Playing audio for slide ${index + 1}`);
                                        // Remove any existing ended listeners to prevent duplicates
                                        audio.onended = null;
                                        audio.addEventListener('ended', () => {
                                            console.log(`Audio ended for slide ${index + 1}`);
                                            index++;
                                            playNext();
                                        }, { once: true });
                                        // Fallback: Check if audio is stuck (e.g., duration not advancing)
                                        const checkDuration = setInterval(() => {
                                            if (!isPlaying) {
                                                clearInterval(checkDuration);
                                                return;
                                            }
                                            if (audio.duration && audio.currentTime >= audio.duration - 0.1) {
                                                console.log(`Fallback: Audio for slide ${index + 1} considered ended`);
                                                clearInterval(checkDuration);
                                                audio.onended = null; // Prevent duplicate triggers
                                                index++;
                                                playNext();
                                            }
                                        }, 1000);
                                    }).catch(e => {
                                        console.error(`Audio play failed for slide ${index + 1}:`, e);
                                        // Retry playing the same slide after a short delay
                                        setTimeout(() => {
                                            audio.play().then(() => {
                                                console.log(`Retry succeeded for slide ${index + 1}`);
                                                audio.onended = null;
                                                audio.addEventListener('ended', () => {
                                                    console.log(`Audio ended for slide ${index + 1}`);
                                                    index++;
                                                    playNext();
                                                }, { once: true });
                                                const checkDuration = setInterval(() => {
                                                    if (!isPlaying) {
                                                        clearInterval(checkDuration);
                                                        return;
                                                    }
                                                    if (audio.duration && audio.currentTime >= audio.duration - 0.1) {
                                                        console.log(`Fallback: Audio for slide ${index + 1} considered ended`);
                                                        clearInterval(checkDuration);
                                                        audio.onended = null;
                                                        index++;
                                                        playNext();
                                                    }
                                                }, 1000);
                                            }).catch(e => {
                                                console.error(`Retry failed for slide ${index + 1}:`, e);
                                                index++; // Move to next slide if retry fails
                                                playNext();
                                            });
                                        }, 500);
                                    });
                                } else {
                                    index++;
                                    playNext();
                                }
                            });
                        }
                        playNext();
                    });
                }

                function toggleFullScreen() {
                    console.log("Fullscreen button clicked");
                    const container = document.getElementById('lecture-container');
                    if (!container) {
                        console.error("Lecture container not found");
                        return;
                    }
                    if (!document.fullscreenElement) {
                        container.requestFullscreen().catch(err => {
                            console.error('Error enabling full-screen:', err);
                        });
                    } else {
                        document.exitFullscreen();
                        console.log("Exited fullscreen");
                    }
                }

                // Attach event listeners
                waitForElement('#prev-btn', (prevBtn) => {
                    prevBtn.addEventListener('click', prevSlide);
                    console.log("Attached event listener to prev-btn");
                });

                waitForElement('#play-btn', (playBtn) => {
                    playBtn.addEventListener('click', playAll);
                    console.log("Attached event listener to play-btn");
                });

                waitForElement('#next-btn', (nextBtn) => {
                    nextBtn.addEventListener('click', nextSlide);
                    console.log("Attached event listener to next-btn");
                });

                waitForElement('#fullscreen-btn', (fullscreenBtn) => {
                    fullscreenBtn.addEventListener('click', toggleFullScreen);
                    console.log("Attached event listener to fullscreen-btn");
                });

                // Initialize audio sources and render first slide
                updateAudioSources(lectureData.audioFiles);
                renderSlide();
                console.log("Initial slide rendered, starting at slide:", currentSlide + 1);
            });
        });
    }

    // Observe DOM changes to detect when lecture container is added
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                const lectureContainer = document.getElementById('lecture-container');
                if (lectureContainer) {
                    console.log("Lecture container detected in DOM");
                    observer.disconnect(); // Stop observing once found
                    initializeSlides();
                }
            }
        });
    });

    // Start observing the document body for changes
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("Started observing DOM for lecture container");
}
"""

# Gradio interface
with gr.Blocks(
    title="Agent Feynman",
    css="""
    h1 {text-align: center}
    h2 {text-align: center}
    #lecture-container {font-family: 'Times New Roman', Times, serif;}
    #slide-content {font-size: 48px; line-height: 1.2;}
    #form-group {box-shadow: 0 0 2rem rgba(0, 0, 0, .14) !important; border-radius: 30px; font-weight: 900; color: #000; background-color: white;}
    #download {box-shadow: 0 0 2rem rgba(0, 0, 0, .14) !important; border-radius: 30px;}
    #slide-display {box-shadow: 0 0 2rem rgba(0, 0, 0, .14) !important; border-radius: 30px; background-color: white;}
    button {transition: background-color 0.3s;}
    button:hover {background-color: #e0e0e0;}
    """,
    js=js_code,
    head='<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">'
) as demo:
    gr.Markdown("""
                # <center>Professor AI Feynman: A Multi-Agent Tool for Learning Anything the Feynman way.</center>
                ## <center>(Jaward Sesay - Microsoft AI Agent Hackathon Submission)</center>""")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_id="form-group"):
                title = gr.Textbox(label="Lecture Title", placeholder="e.g. Introduction to AI")
                lecture_content_description = gr.Textbox(label="Lecture Content Description", placeholder="e.g. Focus on recent advancements")
                lecture_type = gr.Dropdown(["Conference", "University", "High school"], label="Audience", value="University")
                api_service = gr.Dropdown(
                    choices=[
                        "Azure AI Foundry",
                        "OpenAI-gpt-4o-2024-08-06",
                        "Anthropic-claude-3-sonnet-20240229",
                        "Google-gemini-2.0-flash",
                        "Ollama-llama3.2",
                    ],
                    label="Model",
                    value="Google-gemini-2.0-flash"
                )
                api_key = gr.Textbox(label="Model Provider API Key", type="password", placeholder="Not required for Ollama or Azure AI Foundry (use GITHUB_TOKEN env var)")
                serpapi_key = gr.Textbox(label="SerpApi Key (For Research Agent)", type="password", placeholder="Enter your SerpApi key (optional)")
                num_slides = gr.Slider(1, 20, step=1, label="Number of Lecture Slides (will add intro and closing slides)", value=3)
                speaker_audio = gr.Audio(value="feynman.mp3", label="Speaker sample speech (MP3 or WAV)", type="filepath", elem_id="speaker-audio")
                generate_btn = gr.Button("Generate Lecture")
        with gr.Column(scale=2):
            default_slide_html = """
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 700px; padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 30px; box-shadow: 0 0 2rem rgba(0, 0, 0, .14) !important;">
                <h2 style="font-style: italic; color: #000;">Waiting for lecture content...</h2>
                <p style="margin-top: 10px; font-size: 16px;color: #000">Please Generate lecture content via the form on the left first before lecture begins</p>
            </div>
            """
            slide_display = gr.HTML(label="Lecture Slides", value=default_slide_html, elem_id="slide-display")
            file_output = gr.File(label="Download Lecture Materials", elem_id="download")
    
    speaker_audio.change(
        fn=update_audio_preview,
        inputs=speaker_audio,
        outputs=speaker_audio
    )
    
    generate_btn.click(
        fn=on_generate,
        inputs=[api_service, api_key, serpapi_key, title, lecture_content_description, lecture_type, speaker_audio, num_slides],
        outputs=[slide_display, file_output]
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=[OUTPUT_DIR])
