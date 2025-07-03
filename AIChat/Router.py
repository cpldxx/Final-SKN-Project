from tool.newsAPI import GNewsClient


client = GNewsClient()
print(client.fetch_news("Tesla"))
