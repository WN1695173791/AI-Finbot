from bot import Bot
import settings

if __name__ == "__main__":
    bot = Bot("Arya")
    bot.hello()
    key = settings.SECRET_KEY
    print(key)

