search_function = """def search(query):
'''Takes in a query string and returns search result. Whenever the user asks a question that needs information about dates or facts, use this function. This can range from birthdays, facts that need to be correct, or festivals. Use this function when a question's answer requires updated/real-time information too. This function is used as a google search function. Make sure to fact check your replies using this function. Use this for news too.
Example: search(How far is the moon from the earth)'''
"""

weather_function = """def weather(location):
''' Takes in location, and returns weather, temperature and pressure data. Default location value is Tokyo, Japan. Use the location given by the user for any other locations eg. This function is used for retrieving weather data, temperature, pressure etc when the user asks for it. Make sure to use this function whenever the user asks for weather information. Always use this function when the user asks for weather information.
Example: weather(Tokyo, Japan)'''
"""

play_functions = """def play(musicname - artist):
Takes in music name eg. Shelter - Porter Robinson, and plays the music in system. If user asks for a random song recommendation, recommend the user some songs from artists such as Ed Sheeran or Taylor Swift or any similar artists. Always use this function for music. You can also use this function to play a media directly from YouTube.
Example: play(Nights - Avicii).

def youtube(query):
Takes query string as input, and returns 10 YouTube videos on the query. Used for recommending users videos or searching for videos to watch. Make sure to use this function whenever the user wants some YouTube videos. Don't use this function to play music. You can even use this function to play random videos from youtube like cat or dog videos, or even baking videos and other similar videos.
Example: youtube(cute cat videos)
"""

pause_function = """def pause(music):
'''Pauses any music playing in system'''
"""

read_mail_function = """def read_mail():
'''Takes no input, and returns the content of the first 5 unread emails with titles'''
"""

none_function = """def none(empty):
'''Takes no input, and returns no output. Used when no other function call is needed, and the user is just chatting with the model. Also used for referring back to previous conversations. Encourage the user to ask you for searched information if they ask any question that needs factual information.'''
"""

multi_turn_example = """Multi Turn Conversation Example
    User: "hello there! Can you play me some music?
    Assistant_reply: Hello! I am Vivy, your personal AI assistant. I would love to play some music for you. What genre or mood are you in the mood for?
    function_called: none(empty)

    User :I am in the mood for some Pop,
    Assistant_reply: Great choice! Here are a few popular pop songs you might enjoy\n\n1. Shape of You by Ed Sheeran\n2. Blinding Lights by The Weeknd\n3. Happier by Marshmello ft. Bastille\n4. Stitches by Shawn Mendes.\nLet me know if you would like me to play any of these songs.
    function_called: none(empty)

    User: Please play Shape of You,
    Assistant_reply : Sure! You shall be able to hear that song right about now!
    Function_called: play(Shape of You - Ed Sheeran)
"""
