"""
This is the main script. For now, it will just generate a single person explainer podcast, but in the future i want
to build in the ability to have multiple hosts/ guests
It will:
- Take in a list of topics
- Search Google using google-search or news-api to find news articles on those topics
- download the text of those articles and clean them up
- Use Gpt-3 to summarize the articles
- Use Gpt-3 to generate a script for the podcast
- Use ElevenLabs to generate an audio file from the script(While testing, I will use built in text to speech, so I don't
    burn through my credits)
- Add a way to save the script and audio file
- incorporate ElevenLabs for text to voice
- Give the podcast a structure
    - Intro
    - News
    - Interview
    - Outro/Summary


"""

# imports
import os
import pyttsx3
import json
import datetime
import requests
from bs4 import BeautifulSoup
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import text_splitter
from newsapi import NewsApiClient

# set global variables
VERBOSE = True

# set API keys
with open('/Users/alexthe5th/Documents/API Keys/GoogleSearchAPI_key.txt', 'r') as f:
    os.environ["GOOGLE_API_KEY"] = f.read()

with open('/Users/alexthe5th/Documents/API Keys/news-api-key.txt', 'r') as f:
    os.environ["NEWS_API_KEY"] = f.read()

with open('/Users/alexthe5th/Documents/API Keys/GoogleSearch_ID.txt', 'r') as f:
    os.environ["GOOGLE_CSE_ID"] = f.read()

with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()

with open('/Users/alexthe5th/Documents/API Keys/eleven_voice_api_key.txt', 'r') as f:
    os.environ["ELEVENLABS_API_KEY"] = f.read()


# utility functions
def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def speak(text, voice):
    voice.say(text)
    voice.runAndWait()


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


# Initialize everything
def init_llm():
    llm = OpenAI(temperature=0.9, max_tokens=1024, top_p=1, frequency_penalty=0, presence_penalty=0.6)
    return llm


def init_python_tts():
    engine = pyttsx3.init()
    return engine


def init_google_search():
    google_search = GoogleSearchAPIWrapper()
    return google_search


def init_news_api():
    news_api = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    return news_api


def init_tools(llm):
    tools = load_tools(['news-api', 'google-search'], llm, news_api_key=os.environ["NEWS_API_KEY"])
    return tools


def init_eleven_labs():
    pass


def init_agent(tools, llm):
    agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=VERBOSE)
    return agent


def init():
    llm = init_llm()
    engine = init_python_tts()
    google_search = init_google_search()
    tools = init_tools(llm)
    news_api = init_news_api()
    return llm, engine, google_search, news_api, tools


def get_topics(news_api, llm):
    # get a list of the top 10 headlines from the news api
    headlines = []
    top_headlines = news_api.get_top_headlines(language='en',

                                               country='us',
                                               page_size=20)
    # use llm to get a list of 5 topics from the headline list
    template = """"
    The top headlines are:
    
    {headlines}
    
    Generate a list of 5 specific topics that are related to these headlines. 
    The list should be separated by new lines, and not numbered.
    
    Topics:
    """
    prompt = PromptTemplate(input_variables=['headlines'],
                            template=template)
    for headline in top_headlines["articles"]:
        headlines.append(headline["title"])
    topic_list = llm(prompt.format(headlines=headlines)).splitlines()
    for topic in topic_list:
        vprint(f"Topic: {topic}")
    return topic_list


def get_news(news_api, topic_list):
    news = {}
    for topic in topic_list:
        vprint(f"Getting news on {topic}")
        try:
            response = news_api.get_everything(q=topic,
                                               from_param=datetime.datetime.now() - datetime.timedelta(days=1),
                                               to=datetime.datetime.now(),
                                               language='en',
                                               sort_by='relevancy',
                                               page_size=1)
        except Exception as e:
            print(e)
            continue
        news[topic] = {"url": response["articles"][0]["url"],
                       "title": response["articles"][0]["title"],
                       "content": response["articles"][0]["content"],
                       "source": response["articles"][0]["source"]["name"]}
        for item in news[topic]:
            vprint(f"{item}: {news[topic][item]}")
    return news


def get_full_article_text(article_url, content):
    content = content[:100]
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = soup.get_text()
    start_index = article_text.find(content)
    article_text = article_text[start_index:]
    end_index = article_text.find("\n\n")

    article_text = article_text[:end_index]

    vprint(f"Article text: \n {article_text}")
    return article_text


def summarize_article_text(article_text, llm):
    # split article into list of chunks that are each 1024 words long or less
    article_text = article_text.split()
    article_text_chunks = []
    while len(article_text) > 1024:
        article_text_chunks.append(" ".join(article_text[:1024]))
        article_text = article_text[1024:]
    article_text_chunks.append(" ".join(article_text))
    summary = ""
    for chunk in article_text_chunks:
        template = """
        Summarize this text. be sure to include all of the relevant information, 
        and exclude any unnecessary information.
    
        {text}
    
        Summary:"""
        prompt = PromptTemplate(input_variables=["text"],
                                template=template)
        summary = llm(prompt=prompt.format(text=chunk))
    vprint(f"Summarized article text: \n {summary}")
    return summary


def get_podcast_intro(text, llm):
    template = """
    Generate an introduction script for a podcast given the following script. 
    The podcast is called "The Automated Podcast" and is about the news of the day.
    The Podcast is generated entirely by AI and it is important to be clear about this in the introduction.
    Make the text interesting and engaging in a conversational tone:
    
    Script:
    
    {text}

    Introduction:
    """
    prompt = PromptTemplate(input_variables=["text"],
                            template=template)
    try:
        output = llm(prompt=prompt.format(text=text))
    except Exception as e:
        vprint(e)
        output = llm(prompt=prompt.format(text=summarize_article_text(text)))
    vprint(f"Podcast intro: \n {output}")
    return output


def get_podcast_news(articles_dict, llm):
    output = ""
    for title in articles_dict:
        template = """
        Generate a news segment for a podcast given the following news article, This will not include an introduction
        or conclusion, but only the news segment for this article which may be one of many

        {title}

        {content}

        Podcast segment:
        """
        prompt = PromptTemplate(input_variables=["title", "content"],
                                template=template)
        try:
            output += llm(prompt=prompt.format(title=title,
                                               content=articles_dict[title]))
        except Exception as e:
            vprint(e)
            output += llm(prompt=prompt.format(title=title,
                                               content=summarize_article_text(articles_dict[title])))
    vprint(f"Podcast news: \n {output}")
    return output


def generate_interview_questions(podcast_news, llm):
    template = """   
    Generate a list of interview questions for a podcast given the following news segment:
    
    {podcast_news}
    
    Write the interview questions in a list separated by new lines, not as a bullet point list or numbered list.
    
    Interview Questions:
    """
    prompt = PromptTemplate(input_variables=["podcast_news"],
                            template=template)
    questions = llm(prompt=prompt.format(podcast_news=podcast_news))
    questions = questions.split("\n")
    # remove empty strings
    questions = [question for question in questions if question]
    vprint(f"Interview questions: \n {questions}")
    return questions


def get_podcast_interview(questions, llm):
    output = {}
    for question in questions:
        template = """
        You are a guest on a podcast and are being interviewed.
        The following is a question that is being asked of you:

        {question}

        Podcast Guest Response:
        """
        prompt = PromptTemplate(input_variables=["question"],
                                template=template)
        output[question] = llm(prompt=prompt.format(question=question))
    vprint(f"Podcast interview: \n {output}")
    return output


def get_podcast_conclusion(text, llm):
    template = """
    Generate a conclusion script for a podcast given the following script. 
    Make the text interesting and engaging in a conversational tone:

    {text}

    Conclusion:"""
    prompt = PromptTemplate(input_variables=["text"],
                            template=template)
    try:
        output = llm(prompt=prompt.format(text=text))
    except Exception as e:
        vprint(e)
        output = llm(prompt=prompt.format(text=summarize_article_text(text)))
    vprint(f"Podcast conclusion: \n {output}")
    return output


def get_podcast_script(intro, news, interview, conclusion):
    script = intro + "\n\n"
    script += news + "\n\n"
    for question in interview:
        script += question + "\n\n"
        script += interview[question] + "\n\n"
    script += conclusion
    vprint(f"Podcast script: \n {script}")
    return script


def speak_full_podcast(intro, news, interview, conclusion, engine):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(intro)
    engine.runAndWait()
    engine.say(news)
    engine.runAndWait()
    for question in interview:
        engine.setProperty('voice', voices[0].id)
        engine.say(question)
        engine.runAndWait()
        engine.setProperty('voice', voices[10].id)
        engine.say(interview[question])
        engine.runAndWait()
    engine.setProperty('voice', voices[0].id)
    engine.say(conclusion)
    engine.runAndWait()
    engine.stop()


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


def main():
    llm, engine, google_search, news_api, tools = init()
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(f"{date_today}podcast_news.json"):
        topic_list = get_topics(news_api=news_api, llm=llm)
        news = get_news(news_api=news_api, topic_list=topic_list)
        articles_dict = {}
        for item in news:
            article_text = get_full_article_text(news[item]["url"], news[item]["content"])
            articles_dict[news[item]["title"]] = article_text
        podcast_news = get_podcast_news(articles_dict, llm)
        save_json("podcast_news.json", podcast_news)
    else:
        podcast_news = load_json(f"{date_today}podcast_news.json")
    interview_questions = generate_interview_questions(podcast_news, llm)
    interview = get_podcast_interview(interview_questions, llm)
    for question in interview:
        print(question)
        print(interview[question])
    full_text = podcast_news
    for question in interview:
        full_text += f"\n\n{question}\n\n{interview[question]}"
    conclusion = get_podcast_conclusion(full_text, llm)
    full_text += f"\n\n{conclusion}"
    podcast_intro = get_podcast_intro(full_text, llm)
    podcast_script = get_podcast_script(podcast_intro, podcast_news, interview, conclusion)
    print(podcast_script)
    speak_full_podcast(podcast_intro, podcast_news, interview, conclusion, engine)


if __name__ == '__main__':
    main()
