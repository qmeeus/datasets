import sys
sys.path.append("..")
from datasets import FluentSpeechCommands


def test_fluent():
    fluent = FluentSpeechCommands(inputs="mlm", outputs="tasks")
    valid = fluent("valid")
    print(valid, len(valid))


if __name__ == "__main__":
    test_fluent()
