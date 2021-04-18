from transformermanager import TrasformerManager

def main():
    """Runs the program leveraging on the TransformerManager class, performs pre-processing on the input dataset of tweets
       and launches training and testing of Transformer model. After training, a list of tweets will be classified to verify the 
       program is performing as expected."""
       
    tm = TrasformerManager()
    tm.preprocess()
    tm.train()
    tweets = (
        "thought sleeping in was an option tomorrow but realizing that it now is not. evaluations in the morning and work in the afternoon!",
        "I hate everything and the world sucks",
        "I love you and the world is beautiful",
        "I fell in love with you",
        "Do you want to merry me?",
        "I'll kick your ass",
        "This water is tasty",
        "This food is amazing",
        "When I'm with you I feel like I'm complete.",
        "Studying all day makes me deeply satisfied",
        "I can't stand you any more, we better not see each others again.",
        "I think I like you")
    
    print()
    for tweet in tweets:
        print(tweet)
        print(tm.classify(tweet), '\n')


if __name__ == "__main__":
    main()
