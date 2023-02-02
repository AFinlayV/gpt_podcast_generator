"""
This is the main script. For now, it will just generate a single person explainer podcast, but in the future i want
to build in the ability to have multiple hosts/ guests
It will:
- Take in a list of topics
- Search Google using google-search or news-api to find news articles on those topics
- download the text of those articles and clean them up
- Use Gpt-3 to summarize the articles
- Use Gpt-3 to generate a script for the podcast
- Use ElevenLabs to generate an audio file from the script(While testing, I will use built in text to speech, so I don't burn through my credits)
    -(Maybe this in a separate script)



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
TOPIC_LIST = ['ChatGPT', 'AI Art', 'Python Programming']
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
    llm = OpenAI(temperature=0.5, max_tokens=1024, top_p=1, frequency_penalty=0, presence_penalty=0.6)
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


def init_elevenlabs():
    pass


def init_agent(tools, llm):
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=VERBOSE)
    return agent


def init():
    llm = init_llm()
    engine = init_python_tts()
    google_search = init_google_search()
    tools = init_tools(llm)
    agent = init_agent(tools=tools, llm=llm)
    news_api = init_news_api()
    return llm, engine, google_search, agent, news_api, tools


def get_news(news_api):
    news = {}
    for topic in TOPIC_LIST:
        vprint(f"Getting news on {topic}")
        news[topic] = news_api.get_everything(q=topic,
                                              from_param=datetime.datetime.now() - datetime.timedelta(days=1),
                                              to=datetime.datetime.now(),
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=1)
    return news


def get_news_article_urls(news):
    urls = []
    for topic in news:
        for article in news[topic]['articles']:
            urls.append(article['url'])
    return urls


def get_article_text(url):
    article_text = requests.get(url).text
    return article_text


def clean_article_text(article_text):
    # remove html tags, then return only the article text without all the extra stuff
    soup = BeautifulSoup(article_text, 'html.parser')
    text = soup.find_all('p')
    article_text = ''
    for paragraph in text:
        article_text += paragraph.text
    vprint(article_text)
    return article_text


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


def summarize_article_text(article_text, llm):
    article_text = article_text[:4096]
    if check_article_text(article_text, llm):
        transform_chain = TransformChain(input_variables=["text"],
                                         output_variables=["output_text"],
                                         transform=transform_func)
        template = """Summarize this text. be sure to include all of the relevant information
    
        {output_text}
    
        Summary:"""
        prompt = PromptTemplate(input_variables=["output_text"],
                                template=template)
        llm_chain = LLMChain(llm=llm,
                             prompt=prompt)
        sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
        output = sequential_chain.run(article_text)
        vprint(f"Summarized article text: \n {output}")
        return output
    else:
        vprint("Article text is not a news article")
        return None


def get_podcast_intro(text, llm):
    transform_chain = TransformChain(input_variables=["text"],
                                     output_variables=["output_text"],
                                     transform=transform_func)
    template = """
    Generate an introduction script for an engaging podcast given the following script:

    {output_text}

    Introduction:"""
    prompt = PromptTemplate(input_variables=["output_text"],
                            template=template)
    llm_chain = LLMChain(llm=llm,
                         prompt=prompt)
    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
    output = sequential_chain.run(text)
    vprint(f"Podcast intro: \n {output}")
    return output


def check_article_text(article_text, llm):
    template = """
    Check the following text. does it seem like it is a news article?
    
    {text}
    
    Is this a news article answer with only 'Yes' or 'No'?
    Answer:
    """
    prompt = PromptTemplate(input_variables=["text"],
                            template=template)
    output = llm(prompt.format(text=article_text))
    vprint(f"Article text check: {article_text} \n {output}")
    if "yes" in output.lower():
        return True
    else:
        return False


def get_podcast_script(summarized_article_text_list, llm):
    transform_chain = TransformChain(input_variables=["text"],
                                     output_variables=["output_text"],
                                     transform=transform_func)
    template = """
    Generate a long, detailed, verbose script of this article for a segment of a podcast:

    {text}

    Podcast segment:
    """
    prompt = PromptTemplate(input_variables=["text"],
                            template=template)
    llm_chain = LLMChain(llm=llm,
                         prompt=prompt)
    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
    output_list = []
    for item in summarized_article_text_list:
        vprint(item)
        output_list.append(sequential_chain.run(item))
        vprint(output_list[-1])
    return output_list


def main():
    llm, engine, google_search, agent, news_api, tools = init()
    news = get_news(news_api=news_api)
    url_list = get_news_article_urls(news=news)
    raw_article_text_list = []
    for url in url_list:
        raw_article_text_list.append(get_article_text(url=url))
    clean_article_text_list = []
    for item in raw_article_text_list:
        clean_article_text_list.append(clean_article_text(item))
    summarized_article_text_list = []
    for item in clean_article_text_list:
        summarized_article_text_list.append(summarize_article_text(item, llm))
    podcast_segment_list = get_podcast_script(summarized_article_text_list, llm)
    podcast_script = ''
    podcast_script.join(podcast_segment_list)
    podcast_intro = get_podcast_intro(podcast_script, llm)
    vprint('\n----------------Full Podcast script-------------\n', podcast_script, '\n')
    speak(podcast_intro + podcast_script, engine)


if __name__ == '__main__':
    main()
