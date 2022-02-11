import sys
sys.path.append("..")
from datasets import ChatBot


def test_chatbot():
    chatbot = ChatBot(inputs="fbank", outputs="labels")
    dataset = chatbot("all")
    print(dataset, len(dataset))


if __name__ == "__main__":
    test_chatbot()
