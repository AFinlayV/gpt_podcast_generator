This is a project where I will be learning about integrating several AI tools to generate a podcast.
It is based on a tweet (https://twitter.com/jheitzeb/status/1620584063781462020) by @jheitzeb.
The basic idea will be to
-train a voice model on the user's voice from a sample of their voice
-look up current news articles about a set of topics
-feed the data from the articles into an LLM to summarize the articles
-feed the summaries into a GPT-3 model to generate a podcast script
-feed the script into a voice model to generate a podcast

I will be using the following tools:
-voice model: Eleven
-LLM: GPT-3
-search: news_api
